"""Service layer for User and Preferences management."""

from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.database.models.user import (
    CaregiverResidentAssignment,
    OperatingMode,
    ResponseStyle,
    SupportedLanguage,
    User,
    UserPreferences,
    UserRole,
)
from app.users.schemas import UserCreate, UserPreferencesUpdate, UserUpdate


class UserServiceError(Exception):
    """Base exception for user service errors."""
    pass


class UserNotFoundError(UserServiceError):
    """User not found."""
    pass


class UserConflictError(UserServiceError):
    """User attribute already exists (e.g. username, email, NFC card)."""
    pass


class UserService:
    """Manages user registration, profile retrieval, and preference updates."""

    @staticmethod
    def create_user(db: Session, user_in: UserCreate, auto_commit: bool = True) -> User:
        """Register a new user with default preferences.

        Parameters:
            db: Active SQLAlchemy database session.
            user_in: Validated UserCreate payload.
            auto_commit: If True, commits the transaction immediately. If False, flushes
                         pending objects to assign primary keys while allowing callers to
                         participate in an enclosing atomic transaction.
        """
        # 1. Check for unique username
        existing_username = db.execute(
            select(User).where(User.username == user_in.username)
        ).scalar_one_or_none()
        if existing_username:
            raise UserConflictError(f"Username '{user_in.username}' is already registered")

        # 2. Check for unique email if provided
        if user_in.email:
            existing_email = db.execute(
                select(User).where(User.email == user_in.email)
            ).scalar_one_or_none()
            if existing_email:
                raise UserConflictError(f"Email '{user_in.email}' is already registered")

        # 3. Check for unique NFC card ID if provided
        if user_in.nfc_card_id:
            existing_nfc = db.execute(
                select(User).where(User.nfc_card_id == user_in.nfc_card_id)
            ).scalar_one_or_none()
            if existing_nfc:
                raise UserConflictError(f"NFC Card ID '{user_in.nfc_card_id}' is already assigned")

        # 4. Hash password securely if supplied
        hashed_pwd = hash_password(user_in.password) if user_in.password else None

        # 5. Create user record
        db_user = User(
            username=user_in.username,
            full_name=user_in.full_name,
            email=user_in.email,
            hashed_password=hashed_pwd,
            role=user_in.role,
            operating_mode=user_in.operating_mode,
            nfc_card_id=user_in.nfc_card_id,
        )
        db.add(db_user)
        db.flush()  # Populates db_user.id

        # 6. Initialize default preferences tailored to the user's operating mode
        default_lang = user_in.initial_language or SupportedLanguage.ENGLISH
        default_style = user_in.initial_response_style or (
            ResponseStyle.SIMPLE_AND_SLOW
            if user_in.operating_mode == OperatingMode.ELDERLY_CARE
            else ResponseStyle.CONCISE_AND_FAST
        )

        db_preferences = UserPreferences(
            user_id=db_user.id,
            preferred_language=default_lang,
            response_style=default_style,
            spoken_speed=0.9 if user_in.operating_mode == OperatingMode.ELDERLY_CARE else 1.0,
            voice_gender_preference="neutral",
            high_contrast_dashboard=(user_in.operating_mode == OperatingMode.ELDERLY_CARE),
            medication_reminders_enabled=True,
            sound_alerts_enabled=True,
        )
        db.add(db_preferences)
        if auto_commit:
            db.commit()
            db.refresh(db_user)
        else:
            db.flush()
        return db_user

    @staticmethod
    def get_by_id(db: Session, user_id: int) -> User:
        """Fetch user by primary key."""
        user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
        if not user:
            raise UserNotFoundError(f"User with ID {user_id} does not exist")
        return user

    @staticmethod
    def get_by_username(db: Session, username: str) -> Optional[User]:
        """Fetch user by username."""
        return db.execute(
            select(User).where(User.username == username.lower())
        ).scalar_one_or_none()

    @staticmethod
    def get_by_nfc_card_id(db: Session, nfc_card_id: str) -> User:
        """Fetch user mapped to a physical NFC card identifier."""
        user = db.execute(
            select(User).where(User.nfc_card_id == nfc_card_id)
        ).scalar_one_or_none()
        if not user:
            raise UserNotFoundError(f"No user found registered to NFC Card ID '{nfc_card_id}'")
        return user

    @staticmethod
    def list_users(
        db: Session,
        role: Optional[UserRole] = None,
        mode: Optional[OperatingMode] = None,
        active_only: bool = True,
        skip: int = 0,
        limit: int = 50,
    ) -> List[User]:
        """Retrieve paginated registered users with optional role and mode filtering."""
        query = select(User)
        if role:
            query = query.where(User.role == role)
        if mode:
            query = query.where(User.operating_mode == mode)
        if active_only:
            query = query.where(User.is_active.is_(True))

        query = query.order_by(User.id).offset(skip).limit(limit)
        return list(db.execute(query).scalars().all())

    @staticmethod
    def update_user(db: Session, user_id: int, user_update: UserUpdate) -> User:
        """Update user profile attributes and operating mode."""
        user = UserService.get_by_id(db, user_id)

        update_data = user_update.model_dump(exclude_unset=True)

        if "email" in update_data and update_data["email"] != user.email:
            conflict = db.execute(
                select(User).where(User.email == update_data["email"], User.id != user_id)
            ).scalar_one_or_none()
            if conflict:
                raise UserConflictError(f"Email '{update_data['email']}' is already in use")

        if "nfc_card_id" in update_data and update_data["nfc_card_id"] != user.nfc_card_id:
            conflict = db.execute(
                select(User).where(User.nfc_card_id == update_data["nfc_card_id"], User.id != user_id)
            ).scalar_one_or_none()
            if conflict:
                raise UserConflictError(f"NFC Card ID '{update_data['nfc_card_id']}' is already in use")

        for field, value in update_data.items():
            setattr(user, field, value)

        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def update_preferences(db: Session, user_id: int, prefs_update: UserPreferencesUpdate) -> UserPreferences:
        """Update personalized interaction preferences for a user."""
        user = UserService.get_by_id(db, user_id)
        preferences = user.preferences

        if not preferences:
            # Fallback creation if not present
            preferences = UserPreferences(user_id=user_id)
            db.add(preferences)

        update_data = prefs_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(preferences, field, value)

        db.commit()
        db.refresh(preferences)
        return preferences

    @staticmethod
    def delete_user(db: Session, user_id: int) -> None:
        """Soft-deactivate or remove user record."""
        user = UserService.get_by_id(db, user_id)
        db.delete(user)
        db.commit()

    @staticmethod
    def assign_caregiver_to_resident(
        db: Session,
        caregiver_id: int,
        resident_id: int,
    ) -> CaregiverResidentAssignment:
        """Explicitly authorize a Caregiver for a specific Resident."""
        caregiver = UserService.get_by_id(db, caregiver_id)
        resident = UserService.get_by_id(db, resident_id)

        if caregiver.role != UserRole.CAREGIVER:
            raise UserServiceError(f"User {caregiver_id} does not have the CAREGIVER role")
        if resident.role != UserRole.RESIDENT:
            raise UserServiceError(f"User {resident_id} is not a RESIDENT")

        existing = db.execute(
            select(CaregiverResidentAssignment).where(
                CaregiverResidentAssignment.caregiver_id == caregiver_id,
                CaregiverResidentAssignment.resident_id == resident_id,
            )
        ).scalar_one_or_none()
        if existing:
            return existing

        assignment = CaregiverResidentAssignment(
            caregiver_id=caregiver_id,
            resident_id=resident_id,
        )
        db.add(assignment)
        db.commit()
        db.refresh(assignment)
        return assignment

    @staticmethod
    def remove_caregiver_from_resident(
        db: Session,
        caregiver_id: int,
        resident_id: int,
    ) -> bool:
        """Revoke a Caregiver's authorization for a specific Resident."""
        assignment = db.execute(
            select(CaregiverResidentAssignment).where(
                CaregiverResidentAssignment.caregiver_id == caregiver_id,
                CaregiverResidentAssignment.resident_id == resident_id,
            )
        ).scalar_one_or_none()
        if not assignment:
            return False
        db.delete(assignment)
        db.commit()
        return True

    @staticmethod
    def is_caregiver_assigned_to_resident(
        db: Session,
        caregiver_id: int,
        resident_id: int,
    ) -> bool:
        """Verify whether a Caregiver is explicitly authorized for a Resident."""
        assignment = db.execute(
            select(CaregiverResidentAssignment).where(
                CaregiverResidentAssignment.caregiver_id == caregiver_id,
                CaregiverResidentAssignment.resident_id == resident_id,
            )
        ).scalar_one_or_none()
        return assignment is not None

    @staticmethod
    def list_assigned_residents_for_caregiver(
        db: Session,
        caregiver_id: int,
    ) -> List[int]:
        """List all resident user IDs assigned to a specific caregiver."""
        rows = db.execute(
            select(CaregiverResidentAssignment.resident_id).where(
                CaregiverResidentAssignment.caregiver_id == caregiver_id
            )
        ).scalars().all()
        return list(rows)
