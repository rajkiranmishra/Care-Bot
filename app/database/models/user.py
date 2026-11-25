"""SQLAlchemy models for User entities and User Preferences."""

import enum
from typing import TYPE_CHECKING, Optional
from sqlalchemy import Boolean, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.database.models.conversation import ConversationSession


class UserRole(str, enum.Enum):
    """User authorization roles."""
    RESIDENT = "resident"            # Primary voice user in the home
    CAREGIVER = "caregiver"          # Authorized family or professional caregiver
    FAMILY_VIEWER = "family_viewer"  # Read-only family dashboard access


class OperatingMode(str, enum.Enum):
    """CARE-BOT operating modes (individually configurable per user)."""
    ELDERLY_CARE = "elderly_care"                      # Simplified speech, high patience, health priority
    WORKING_PROFESSIONAL = "working_professional"      # Concise speech, productivity, scheduling, weather


class SupportedLanguage(str, enum.Enum):
    """Supported multilingual communication languages."""
    ENGLISH = "en"
    HINDI = "hi"
    BENGALI = "bn"


class ResponseStyle(str, enum.Enum):
    """Configurable AI response styles."""
    SIMPLE_AND_SLOW = "simple_slow"
    STANDARD = "standard"
    CONCISE_AND_FAST = "concise_fast"


class User(Base, TimestampMixin):
    """Registered CARE-BOT user."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(128), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255), unique=True, index=True, nullable=True)
    hashed_password: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role_enum", native_enum=False),
        default=UserRole.RESIDENT,
        nullable=False,
    )
    operating_mode: Mapped[OperatingMode] = mapped_column(
        Enum(OperatingMode, name="operating_mode_enum", native_enum=False),
        default=OperatingMode.ELDERLY_CARE,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Contactless RFID/NFC card identifier for hardware profile selection
    nfc_card_id: Mapped[Optional[str]] = mapped_column(String(64), unique=True, index=True, nullable=True)

    # Relationships
    preferences: Mapped["UserPreferences"] = relationship(
        "UserPreferences",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="joined",
    )
    sessions: Mapped[list["ConversationSession"]] = relationship(
        "ConversationSession",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username='{self.username}' mode='{self.operating_mode}' role='{self.role}'>"


class UserPreferences(Base, TimestampMixin):
    """Configurable interaction preferences associated with a User."""
    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    preferred_language: Mapped[SupportedLanguage] = mapped_column(
        Enum(SupportedLanguage, name="language_enum", native_enum=False),
        default=SupportedLanguage.ENGLISH,
        nullable=False,
    )
    response_style: Mapped[ResponseStyle] = mapped_column(
        Enum(ResponseStyle, name="response_style_enum", native_enum=False),
        default=ResponseStyle.STANDARD,
        nullable=False,
    )
    spoken_speed: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    voice_gender_preference: Mapped[str] = mapped_column(String(16), default="neutral", nullable=False)
    high_contrast_dashboard: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    medication_reminders_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    sound_alerts_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationship
    user: Mapped["User"] = relationship("User", back_populates="preferences")

    def __repr__(self) -> str:
        return f"<UserPreferences user_id={self.user_id} lang='{self.preferred_language}' style='{self.response_style}'>"


class SystemBootstrap(Base, TimestampMixin):
    """Database-level atomic tracking and locking model for initial administrator bootstrap.

    The single row with primary key `id=1` is created atomically during the initial
    administrator bootstrap. The database engine's primary key constraint enforces that
    even across multiple operating system processes or Uvicorn workers, only a single
    bootstrap operation can ever succeed.
    """
    __tablename__ = "system_bootstrap"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False, default=1)
    admin_user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_bootstrapped: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    def __repr__(self) -> str:
        return f"<SystemBootstrap id={self.id} is_bootstrapped={self.is_bootstrapped} admin_user_id={self.admin_user_id}>"
