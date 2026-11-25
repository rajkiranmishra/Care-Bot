"""User management and preference REST endpoints with authentication and RBAC."""

import hmac
import threading
from typing import List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_user,
    get_db,
    get_optional_current_user,
    get_settings,
    Settings,
    verify_device_auth,
    verify_user_access,
)
from app.database.models.user import OperatingMode, SupportedLanguage, SystemBootstrap, User, UserRole
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
    UserServiceError,
)

router = APIRouter(prefix="/users", tags=["User Management"])

# Process-level concurrency lock ensuring single-use atomic administrator bootstrap
_bootstrap_lock = threading.Lock()


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user profile",
)
def create_user(
    user_in: UserCreate,
    current_user: Optional[User] = Depends(get_optional_current_user),
    x_device_key: Optional[str] = Header(None, alias="X-Device-Key"),
    x_bootstrap_token: Optional[str] = Header(None, alias="X-Bootstrap-Token"),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UserResponse:
    """Create a new user with individualized operating mode and default preferences.

    SECURITY RULE (Privilege Escalation Prevention & Bootstrap Security):
    Public self-registration is strictly restricted to the 'resident' role.
    Elevated roles ('caregiver' and 'family_viewer') may ONLY be assigned by:
    1. An authenticated user possessing the CAREGIVER role.
    2. An authorized device administrator presenting a valid hardware X-Device-Key.
    3. The initial administrator bootstrap flow presenting a valid X-Bootstrap-Token
       BEFORE any administrator account exists in the system.
    """
    if user_in.role != UserRole.RESIDENT:
        is_caregiver = current_user is not None and current_user.role == UserRole.CAREGIVER
        is_device_admin = bool(
            x_device_key
            and settings.DEVICE_API_KEY
            and hmac.compare_digest(x_device_key, settings.DEVICE_API_KEY)
        )
        is_bootstrap_attempt = x_bootstrap_token is not None

        if is_caregiver or is_device_admin:
            # Authorized ongoing administrative provisioning
            pass
        elif is_bootstrap_attempt:
            # Validate server configuration for initial bootstrap
            if not settings.BOOTSTRAP_TOKEN:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Initial administrator bootstrap is disabled on this server.",
                )
            if not hmac.compare_digest(x_bootstrap_token, settings.BOOTSTRAP_TOKEN):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid bootstrap token.",
                )

            # Atomic concurrency protection: in-process mutex AND database-level unique constraint
            # to guarantee that multiple threads, processes, or Uvicorn workers cannot race.
            with _bootstrap_lock:
                db.expire_all()
                has_caregiver = (
                    db.query(User).filter(User.role == UserRole.CAREGIVER).first() is not None
                )
                has_bootstrap_marker = (
                    db.query(SystemBootstrap).filter(SystemBootstrap.id == 1).first() is not None
                )
                if has_caregiver or has_bootstrap_marker:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Initial administrator bootstrap is disabled: An administrator account has already been provisioned.",
                    )
                try:
                    # Database-level atomic latch: inserting SystemBootstrap(id=1) guarantees that
                    # even across multiple operating system processes or Uvicorn workers,
                    # the database engine's primary key constraint serializes and rejects concurrent attempts.
                    bootstrap_marker = SystemBootstrap(id=1, is_bootstrapped=True)
                    db.add(bootstrap_marker)
                    db.flush()

                    user = UserService.create_user(db, user_in, auto_commit=False)
                    bootstrap_marker.admin_user_id = user.id
                    db.commit()
                    db.refresh(user)
                    return UserResponse.model_validate(user)
                except IntegrityError:
                    db.rollback()
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Initial administrator bootstrap is disabled: An administrator account has already been provisioned.",
                    )
                except UserConflictError as exc:
                    db.rollback()
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
                except Exception:
                    db.rollback()
                    raise
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Privilege escalation prevented: Public self-registration is restricted to "
                    f"role '{UserRole.RESIDENT.value}'. Elevated role '{user_in.role.value}' must "
                    f"be provisioned by an authenticated Caregiver or authorized bootstrap."
                ),
            )

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


@router.post(
    "/{resident_id}/caregivers/{caregiver_id}",
    status_code=status.HTTP_200_OK,
    summary="Assign an authorized caregiver to a resident",
)
def assign_caregiver(
    resident_id: int,
    caregiver_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Assign an authorized caregiver to a resident. Only Caregivers can establish assignments."""
    if current_user.role != UserRole.CAREGIVER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only caregivers can assign caregiver relationships",
        )
    try:
        UserService.assign_caregiver_to_resident(db, caregiver_id=caregiver_id, resident_id=resident_id)
        return {"status": "assigned", "resident_id": resident_id, "caregiver_id": caregiver_id}
    except UserNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except UserServiceError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.delete(
    "/{resident_id}/caregivers/{caregiver_id}",
    status_code=status.HTTP_200_OK,
    summary="Remove a caregiver authorization for a resident",
)
def remove_caregiver(
    resident_id: int,
    caregiver_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove a caregiver authorization for a resident. Only Caregivers can revoke assignments."""
    if current_user.role != UserRole.CAREGIVER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only caregivers can revoke caregiver relationships",
        )
    removed = UserService.remove_caregiver_from_resident(db, caregiver_id=caregiver_id, resident_id=resident_id)
    return {"status": "removed" if removed else "not_found", "resident_id": resident_id, "caregiver_id": caregiver_id}

