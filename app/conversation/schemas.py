"""Pydantic validation schemas for Conversation Sessions and Messages."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.database.models.conversation import (
    ConversationState,
    MessageRole,
    SessionStatus,
)
from app.database.models.user import OperatingMode, ResponseStyle, SupportedLanguage


class MessageCreate(BaseModel):
    """Schema for posting an utterance or assistant response."""
    role: MessageRole = Field(..., description="Message origin: user, assistant, system")
    content: str = Field(..., min_length=1, description="Message text content")
    detected_language: Optional[str] = Field(default=None, max_length=8)
    intent_detected: Optional[str] = Field(default=None, max_length=64)


class MessageResponse(BaseModel):
    """Schema for serialized conversation messages."""
    id: int
    session_id: str
    user_id: int
    role: MessageRole
    content: str
    detected_language: Optional[str] = None
    intent_detected: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SessionCreate(BaseModel):
    """Schema for initializing a dialogue session."""
    title: Optional[str] = Field(default=None, max_length=128, description="Optional conversation topic title")
    initial_state: ConversationState = Field(
        default=ConversationState.IDLE,
        description="Initial FSM state",
    )


class SessionUpdate(BaseModel):
    """Schema for updating session state and context metadata."""
    title: Optional[str] = Field(default=None, max_length=128)
    status: Optional[SessionStatus] = None
    current_state: Optional[ConversationState] = None
    context_metadata: Optional[str] = Field(default=None, description="Serialized context/slot state JSON")


class SessionResponse(BaseModel):
    """Schema for serialized conversation sessions."""
    id: str
    user_id: int
    title: Optional[str] = None
    status: SessionStatus
    current_state: ConversationState
    context_metadata: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class UserContextPayload(BaseModel):
    """Structured context representation formatted for downstream LLM inference."""
    user_id: int
    username: str
    full_name: str
    operating_mode: OperatingMode
    preferred_language: SupportedLanguage
    response_style: ResponseStyle
    active_session_id: Optional[str] = None
    session_state: ConversationState
    system_instructions: str
    recent_messages: List[MessageResponse] = Field(default_factory=list)
