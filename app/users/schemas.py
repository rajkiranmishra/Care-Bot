"""Pydantic validation schemas for User and Preference entities."""

import re
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from app.database.models.user import (
    OperatingMode,
    ResponseStyle,
    SupportedLanguage,
    UserRole,
)


class UserPreferencesBase(BaseModel):
    """Base schema for user preferences."""
    preferred_language: SupportedLanguage = Field(default=SupportedLanguage.ENGLISH, description="Preferred language")
    response_style: ResponseStyle = Field(default=ResponseStyle.STANDARD, description="Response verbosity and style")
    spoken_speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech synthesis rate multiplier")
    voice_gender_preference: str = Field(default="neutral", description="Voice persona: female, male, neutral")
    high_contrast_dashboard: bool = Field(default=False, description="Accessibility high contrast toggle")
    medication_reminders_enabled: bool = Field(default=True, description="Enable medication reminders")
    sound_alerts_enabled: bool = Field(default=True, description="Enable audio chime alerts")


class UserPreferencesUpdate(BaseModel):
    """Schema for updating interaction preferences."""
    preferred_language: Optional[SupportedLanguage] = None
    response_style: Optional[ResponseStyle] = None
    spoken_speed: Optional[float] = Field(default=None, ge=0.5, le=2.0)
    voice_gender_preference: Optional[str] = None
    high_contrast_dashboard: Optional[bool] = None
    medication_reminders_enabled: Optional[bool] = None
    sound_alerts_enabled: Optional[bool] = None


class UserPreferencesResponse(UserPreferencesBase):
    """Schema for serialized user preferences."""
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserBase(BaseModel):
    """Base schema for User attributes."""
    username: str = Field(..., min_length=3, max_length=64, description="Unique alphanumeric username")
    full_name: str = Field(..., min_length=1, max_length=128, description="User full display name")
    email: Optional[str] = Field(default=None, max_length=255, description="Optional email address")
    role: UserRole = Field(default=UserRole.RESIDENT, description="Role: resident, caregiver, family_viewer")
    operating_mode: OperatingMode = Field(default=OperatingMode.ELDERLY_CARE, description="Elderly Care or Working Professional")
    nfc_card_id: Optional[str] = Field(default=None, max_length=64, description="Hardware NFC card identifier")

    @field_validator("username")
    @classmethod
    def validate_username_format(cls, value: str) -> str:
        """Enforce clean alphanumeric and hyphen/underscore usernames."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", value):
            raise ValueError("Username may only contain letters, numbers, underscores, and hyphens")
        return value.lower()

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: Optional[str]) -> Optional[str]:
        """Validate standard email format without requiring external email-validator package."""
        if value is None:
            return None
        value = value.strip().lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value):
            raise ValueError("Invalid email address format")
        return value


class UserCreate(UserBase):
    """Schema for registering a new user."""
    password: Optional[str] = Field(default=None, min_length=6, description="Optional password for dashboard login")
    # Optional initial preference overrides
    initial_language: Optional[SupportedLanguage] = None
    initial_response_style: Optional[ResponseStyle] = None


class UserUpdate(BaseModel):
    """Schema for updating basic user profile and operating mode."""
    full_name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    email: Optional[str] = None
    operating_mode: Optional[OperatingMode] = None
    is_active: Optional[bool] = None
    nfc_card_id: Optional[str] = Field(default=None, max_length=64)

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: Optional[str]) -> Optional[str]:
        """Validate email format if provided."""
        if value is None:
            return None
        value = value.strip().lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value):
            raise ValueError("Invalid email address format")
        return value


class UserResponse(UserBase):
    """Schema for serialized user profiles including preferences."""
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    preferences: Optional[UserPreferencesResponse] = None

    model_config = ConfigDict(from_attributes=True)
