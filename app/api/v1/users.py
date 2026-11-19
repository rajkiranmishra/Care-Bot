"""User management and preference REST endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.database.models.user import OperatingMode, UserRole
from app.users.schemas import (
    UserCreate,
    UserPreferencesResponse,
    UserPreferencesUpdate,
    UserResponse,
    UserUpdate,
)
from app.users.service import (
    UserConflictError,
    UserNotFoundError,
    UserService,
)

router = APIRouter(prefix="/users", tags=["User Management"])


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user profile",
)
def create_user(
    user_in: UserCreate,
    db: Session = Depends(get_db),
) -> UserResponse:
    """Create a new user with individualized operating mode and default preferences."""
    try:
        user = UserService.create_user(db, user_in)
        return UserResponse.model_validate(user)
    except UserConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.get(
    "",
    response_model=List[UserResponse],
    summary="List registered users",
)
def list_users(
    role: Optional[UserRole] = Query(default=None, description="Filter by user role"),
    mode: Optional[OperatingMode] = Query(default=None, description="Filter by operating mode"),
    active_only: bool = Query(default=True, description="Filter only active users"),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
) -> List[UserResponse]:
    """Retrieve a paginated list of users."""
    users = UserService.list_users(
        db,
        role=role,
        mode=mode,
        active_only=active_only,
        skip=skip,
        limit=limit,
    )
    return [UserResponse.model_validate(u) for u in users]


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user profile by ID",
)
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
) -> UserResponse:
    """Retrieve complete user profile and interaction preferences."""
    try:
        user = UserService.get_by_id(db, user_id)
        return UserResponse.model_validate(user)
    except UserNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.put(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update user profile and operating mode",
)
def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
) -> UserResponse:
    """Update user's name, email, NFC card ID, or switch operating mode."""
    try:
        user = UserService.update_user(db, user_id, user_update)
        return UserResponse.model_validate(user)
    except UserNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except UserConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.put(
    "/{user_id}/preferences",
    response_model=UserPreferencesResponse,
    summary="Update interaction preferences",
)
def update_user_preferences(
    user_id: int,
    prefs_update: UserPreferencesUpdate,
    db: Session = Depends(get_db),
) -> UserPreferencesResponse:
    """Update language, speech speed, response style, or accessibility preferences."""
    try:
        prefs = UserService.update_preferences(db, user_id, prefs_update)
        return UserPreferencesResponse.model_validate(prefs)
    except UserNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a user profile",
)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
) -> None:
    """Delete a user profile and all associated data."""
    try:
        UserService.delete_user(db, user_id)
    except UserNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get(
    "/lookup/nfc/{nfc_card_id}",
    response_model=UserResponse,
    summary="Identify user by physical NFC/RFID card tap",
)
def lookup_user_by_nfc(
    nfc_card_id: str,
    db: Session = Depends(get_db),
) -> UserResponse:
    """Hardware assistant integration: resolves a physical NFC card to a registered user."""
    try:
        user = UserService.get_by_nfc_card_id(db, nfc_card_id)
        return UserResponse.model_validate(user)
    except UserNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
