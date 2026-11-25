"""SQLAlchemy models for multi-turn conversation sessions and message history."""

import enum
import uuid
from typing import TYPE_CHECKING, Optional
from sqlalchemy import Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.database.models.user import User


class SessionStatus(str, enum.Enum):
    """Lifecycle state of a conversation session."""
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


class ConversationState(str, enum.Enum):
    """FSM states for multi-turn conversational interaction."""
    IDLE = "idle"
    ACTIVE_CONVERSATION = "active_conversation"
    WAITING_FOR_USER = "waiting_for_user"
    PROCESSING_REQUEST = "processing_request"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
    SESSION_ENDED = "session_ended"


class MessageRole(str, enum.Enum):
    """Origin of a conversation message."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


def generate_session_id() -> str:
    """Generate a standard UUID4 string for conversation session IDs."""
    return str(uuid.uuid4())


class ConversationSession(Base, TimestampMixin):
    """Persistent conversational dialogue session associated with a user."""
    __tablename__ = "conversation_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_session_id)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, name="session_status_enum", native_enum=False),
        default=SessionStatus.ACTIVE,
        nullable=False,
    )
    current_state: Mapped[ConversationState] = mapped_column(
        Enum(ConversationState, name="conversation_state_enum", native_enum=False),
        default=ConversationState.IDLE,
        nullable=False,
    )
    # Serialized JSON context metadata (e.g. pending slot parameters, active intent)
    context_metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="sessions")
    messages: Mapped[list["ConversationMessage"]] = relationship(
        "ConversationMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.id",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<ConversationSession id='{self.id}' user_id={self.user_id} state='{self.current_state}'>"


class ConversationMessage(Base, TimestampMixin):
    """An individual utterance or response within a conversation session."""
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, name="message_role_enum", native_enum=False),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    detected_language: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    intent_detected: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Relationship
    session: Mapped["ConversationSession"] = relationship("ConversationSession", back_populates="messages")

    def __repr__(self) -> str:
        return f"<ConversationMessage id={self.id} role='{self.role}' user_id={self.user_id}>"
