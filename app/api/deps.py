"""Common dependency injection providers for FastAPI route handlers."""

from typing import Generator, List, Optional
from fastapi import Depends, Header, HTTPException, status
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
) -> None:
    """Validate that the authenticated caller has authorization to access target_user_id's data.

    Rules:
    1. A user can always access their own data (`current_user.id == target_user_id`).
    2. A CAREGIVER can manage and inspect resident profiles.
    3. All other cross-user access attempts are rejected with 403 Forbidden.
    """
    if current_user.id == target_user_id:
        return

    if current_user.role == UserRole.CAREGIVER:
        return

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
    1. A valid hardware pre-shared device key matching settings.DEVICE_API_KEY in X-Device-Key, OR
    2. An authenticated CAREGIVER or RESIDENT session.
    """
    if x_device_key and x_device_key == settings.DEVICE_API_KEY:
        return

    if current_user and current_user.role in (UserRole.RESIDENT, UserRole.CAREGIVER):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized hardware or client: Valid X-Device-Key or user bearer token required",
    )
