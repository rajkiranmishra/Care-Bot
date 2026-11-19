"""User management and preference REST endpoints with authentication and RBAC."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_user,
    get_db,
    verify_device_auth,
    verify_user_access,
)
from app.database.models.user import OperatingMode, SupportedLanguage, User, UserRole
from app.users.schemas import (
    UserCreate,
    UserNfcLookupResponse,
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
    summary="List registered users (Authenticated)",
)
def list_users(
    role: Optional[UserRole] = Query(default=None, description="Filter by user role"),
    mode: Optional[OperatingMode] = Query(default=None, description="Filter by operating mode"),
    active_only: bool = Query(default=True, description="Filter only active users"),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[UserResponse]:
    """Retrieve registered users (accessible to authenticated users)."""
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserResponse:
    """Retrieve complete user profile and preferences with ownership verification."""
    verify_user_access(user_id, current_user)
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserResponse:
    """Update profile and switch operating mode. Family viewers are restricted."""
    verify_user_access(user_id, current_user)

    if current_user.role == UserRole.FAMILY_VIEWER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Family viewers have read-only access and cannot modify user profiles",
        )

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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserPreferencesResponse:
    """Update interaction preferences. Family viewers are restricted."""
    verify_user_access(user_id, current_user)

    if current_user.role == UserRole.FAMILY_VIEWER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Family viewers have read-only access and cannot modify user preferences",
        )

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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Delete a user profile. Family viewers are restricted."""
    verify_user_access(user_id, current_user)

    if current_user.role == UserRole.FAMILY_VIEWER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Family viewers cannot delete user profiles",
        )

    try:
        UserService.delete_user(db, user_id)
    except UserNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get(
    "/lookup/nfc/{nfc_card_id}",
    response_model=UserNfcLookupResponse,
    summary="Identify user by physical NFC/RFID card tap (Device Authenticated)",
)
def lookup_user_by_nfc(
    nfc_card_id: str,
    _: None = Depends(verify_device_auth),
    db: Session = Depends(get_db),
) -> UserNfcLookupResponse:
    """Hardware integration: resolves an NFC card tap to minimal non-sensitive user identity.

    Protected by device pre-shared key (X-Device-Key) or authenticated session.
    """
    try:
        user = UserService.get_by_nfc_card_id(db, nfc_card_id)
        pref_lang = user.preferences.preferred_language if user.preferences else SupportedLanguage.ENGLISH
        return UserNfcLookupResponse(
            user_id=user.id,
            username=user.username,
            full_name=user.full_name,
            operating_mode=user.operating_mode,
            preferred_language=pref_lang,
            nfc_card_id=user.nfc_card_id or nfc_card_id,
        )
    except UserNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
