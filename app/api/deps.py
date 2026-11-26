"""Common dependency injection providers for FastAPI route handlers."""

import hmac
from typing import Generator, List, Optional
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.core.security import (
    ExpiredTokenError,
    InvalidTokenError,
    decode_access_token,
)
from app.database.models.user import User, UserRole
from app.database.session import get_db
from app.intelligence.engine import ConversationEngine
from app.intelligence.llm_client import LocalLLMClient, MockLLMClient
from app.intelligence.ollama_client import OllamaLLMClient
from app.users.service import UserService

# Bearer token extractor (auto_error=False allows custom structured error reporting)
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> User:
    """Validate bearer token from Authorization header and return the authenticated User."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_access_token(credentials.credentials, settings.SECRET_KEY)
    except ExpiredTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token claims: missing subject identifier",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User referenced in token does not exist",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account has been deactivated",
        )

    return user


def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Optional[User]:
    """Extract authenticated user if bearer credentials are provided, else return None."""
    if not credentials or not credentials.credentials:
        return None
    try:
        return get_current_user(credentials, db, settings)
    except HTTPException:
        return None


def verify_user_access(
    target_user_id: int,
    current_user: User,
    db: Session,
) -> None:
    """Validate that the authenticated caller has authorization to access target_user_id's data.

    SECURITY INVARIANTS:
    1. A user can always access their own permitted resources (`current_user.id == target_user_id`).
    2. If target_user_id does not exist in the system, return 404 Not Found.
    3. A CAREGIVER can access a resident's resources ONLY if an explicit CaregiverResidentAssignment
       exists in the database linking this caregiver to target_user_id. Ambient caregiver authority
       across unassigned residents is strictly forbidden.
    4. Unassigned caregivers attempting to access another resident's data receive HTTP 403 Forbidden.
    5. Residents cannot access other residents' data (HTTP 403 Forbidden).
    6. FAMILY_VIEWER accounts have read-only access to their own permitted resources and cannot access
       other users' data without explicit authorization.
    """
    if current_user.id == target_user_id:
        return

    target_user = db.get(User, target_user_id)
    if target_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {target_user_id} not found",
        )

    if current_user.role == UserRole.CAREGIVER:
        if UserService.is_caregiver_assigned_to_resident(
            db, caregiver_id=current_user.id, resident_id=target_user_id
        ):
            return
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: Caregiver {current_user.id} is not assigned to resident {target_user_id}. Explicit caregiver-to-resident assignment is required.",
        )

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"Access denied: You do not have permission to access resources for user {target_user_id}",
    )


def require_role(allowed_roles: List[UserRole]):
    """Factory creating a role-enforcement dependency."""
    def _role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires one of roles: {[r.value for r in allowed_roles]}",
            )
        return current_user
    return _role_checker


def verify_device_auth(
    x_device_key: Optional[str] = Header(None, alias="X-Device-Key"),
    current_user: Optional[User] = Depends(get_optional_current_user),
    settings: Settings = Depends(get_settings),
) -> None:
    """Authenticate physical hardware assistant requests (e.g. NFC tap, emergency button).

    Requires either:
    1. A valid hardware pre-shared device key matching settings.DEVICE_API_KEY in X-Device-Key,
       verified using constant-time comparison (hmac.compare_digest), OR
    2. An authenticated CAREGIVER or RESIDENT session bearer token.
    """
    if (
        x_device_key
        and settings.DEVICE_API_KEY
        and hmac.compare_digest(x_device_key, settings.DEVICE_API_KEY)
    ):
        return

    if current_user and current_user.role in (UserRole.RESIDENT, UserRole.CAREGIVER):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized hardware or client: Valid X-Device-Key or user bearer token required",
    )


def get_llm_client(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> LocalLLMClient:
    """Dependency provider yielding the configured LocalLLMClient implementation.

    LIFECYCLE MANAGEMENT:
    When running under the FastAPI application lifecycle with LOCAL_LLM_PROVIDER='ollama',
    this reuses the persistent OllamaLLMClient stored on app.state.llm_client to preserve
    connection pooling and prevent creating new HTTP clients on every incoming request.

    If application lifespan initialization was bypassed or failed, lazily initializes
    and attaches a single managed client to app.state.llm_client so that subsequent requests
    reuse the same instance rather than leaking unmanaged HTTP clients.

    TESTING ISOLATION:
    Defaults to MockLLMClient for offline, deterministic test suite execution without
    requiring a physical GPU or active local Ollama daemon.
    """
    if settings.LOCAL_LLM_PROVIDER.lower() == "ollama":
        if request and hasattr(request, "app") and hasattr(request.app, "state"):
            client = getattr(request.app.state, "llm_client", None)
            if client is None:
                client = OllamaLLMClient(
                    base_url=settings.OLLAMA_BASE_URL,
                    model_name=settings.LOCAL_LLM_MODEL,
                    timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
                    temperature=settings.LLM_TEMPERATURE,
                )
                request.app.state.llm_client = client
            return client
        return OllamaLLMClient(
            base_url=settings.OLLAMA_BASE_URL,
            model_name=settings.LOCAL_LLM_MODEL,
            timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
            temperature=settings.LLM_TEMPERATURE,
        )
    return MockLLMClient()


def get_conversation_engine(
    llm_client: LocalLLMClient = Depends(get_llm_client),
) -> ConversationEngine:
    """Dependency provider for ConversationEngine with injected LocalLLMClient."""
    return ConversationEngine(llm_client=llm_client)

