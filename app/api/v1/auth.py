"""Authentication REST endpoints for issuing signed access tokens."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, get_settings, Settings
from app.core.security import create_access_token, verify_password
from app.database.models.user import User
from app.users.schemas import UserResponse
from app.users.service import UserService

router = APIRouter(prefix="/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    """Credentials payload for token authentication."""
    username: str = Field(..., min_length=1, description="Registered username")
    password: str = Field(..., min_length=1, description="Account password")


class TokenResponse(BaseModel):
    """Cryptographic bearer token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: int
    username: str
    role: str


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate with username & password to receive signed access token",
)
def login(
    login_in: LoginRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """Validate credentials and issue an HMAC-SHA256 signed access token."""
    user = UserService.get_by_username(db, login_in.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account does not have a password configured. Contact a caregiver or admin.",
        )

    if not verify_password(login_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )

    expires_in_seconds = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    token = create_access_token(
        claims={
            "sub": str(user.id),
            "username": user.username,
            "role": user.role.value,
        },
        secret_key=settings.SECRET_KEY,
        expires_in_seconds=expires_in_seconds,
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in_seconds,
        user_id=user.id,
        username=user.username,
        role=user.role.value,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get verified identity and profile of current authenticated user",
)
def get_me(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """Return the profile of the caller validated from the Bearer token."""
    return UserResponse.model_validate(current_user)
