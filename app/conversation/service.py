"""Service layer managing multi-turn dialogue sessions and message history."""

from typing import List, Optional
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.conversation.schemas import MessageCreate, SessionCreate, SessionUpdate
from app.database.models.conversation import (
    ConversationMessage,
    ConversationSession,
    ConversationState,
    SessionStatus,
)
from app.users.service import UserService, UserNotFoundError


class ConversationServiceError(Exception):
    """Base exception for conversation context errors."""
    pass


class SessionNotFoundError(ConversationServiceError):
    """Session not found or not owned by user."""
    pass


class SessionAccessDeniedError(ConversationServiceError):
    """Session exists but belongs to a different user."""
    pass


class ConversationService:
    """Provides isolated, user-scoped conversation session and history management."""

    @staticmethod
    def create_session(db: Session, user_id: int, session_in: SessionCreate) -> ConversationSession:
        """Create a new conversational dialogue session for a verified user."""
        # Ensure user exists
        UserService.get_by_id(db, user_id)

        session = ConversationSession(
            user_id=user_id,
            title=session_in.title,
            status=SessionStatus.ACTIVE,
            current_state=session_in.initial_state,
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    @staticmethod
    def get_session(db: Session, user_id: int, session_id: str) -> ConversationSession:
        """Fetch session ensuring strict user-ownership isolation."""
        session = db.execute(
            select(ConversationSession).where(ConversationSession.id == session_id)
        ).scalar_one_or_none()

        if not session:
            raise SessionNotFoundError(f"Session with ID '{session_id}' not found")

        # Security check: User ownership boundary
        if session.user_id != user_id:
            raise SessionAccessDeniedError(f"Access denied: Session '{session_id}' does not belong to user {user_id}")

        return session

    @staticmethod
    def list_sessions(
        db: Session,
        user_id: int,
        status: Optional[SessionStatus] = None,
        limit: int = 20,
    ) -> List[ConversationSession]:
        """List dialogue sessions exclusively belonging to the specified user."""
        query = select(ConversationSession).where(ConversationSession.user_id == user_id)
        if status:
            query = query.where(ConversationSession.status == status)

        query = query.order_by(desc(ConversationSession.updated_at)).limit(limit)
        return list(db.execute(query).scalars().all())

    @staticmethod
    def update_session(
        db: Session,
        user_id: int,
        session_id: str,
        session_update: SessionUpdate,
    ) -> ConversationSession:
        """Update session state, status, or context metadata with ownership enforcement."""
        session = ConversationService.get_session(db, user_id, session_id)

        update_data = session_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(session, field, value)

        db.commit()
        db.refresh(session)
        return session

    @staticmethod
    def add_message(
        db: Session,
        user_id: int,
        session_id: str,
        message_in: MessageCreate,
        commit: bool = True,
    ) -> ConversationMessage:
        """Append an utterance or response to a session with user ownership validation."""
        session = ConversationService.get_session(db, user_id, session_id)

        message = ConversationMessage(
            session_id=session.id,
            user_id=user_id,
            role=message_in.role,
            content=message_in.content,
            detected_language=message_in.detected_language,
            intent_detected=message_in.intent_detected,
        )
        db.add(message)

        # Advance session timestamp
        session.status = SessionStatus.ACTIVE
        if commit:
            db.commit()
            db.refresh(message)
        return message

    @staticmethod
    def get_messages(
        db: Session,
        user_id: int,
        session_id: str,
        limit: int = 50,
    ) -> List[ConversationMessage]:
        """Fetch chronological message history ensuring user ownership isolation."""
        # Enforces ownership check
        ConversationService.get_session(db, user_id, session_id)

        query = (
            select(ConversationMessage)
            .where(
                ConversationMessage.session_id == session_id,
                ConversationMessage.user_id == user_id,
            )
            .order_by(ConversationMessage.id.asc())
            .limit(limit)
        )
        return list(db.execute(query).scalars().all())

    @staticmethod
    def get_or_create_active_session(db: Session, user_id: int) -> ConversationSession:
        """Retrieve the most recent active session for a user or initialize a new one."""
        session = db.execute(
            select(ConversationSession)
            .where(
                ConversationSession.user_id == user_id,
                ConversationSession.status == SessionStatus.ACTIVE,
            )
            .order_by(desc(ConversationSession.updated_at))
            .limit(1)
        ).scalar_one_or_none()

        if not session:
            session = ConversationService.create_session(
                db,
                user_id,
                SessionCreate(title="Active Dialogue", initial_state=ConversationState.IDLE),
            )
        return session
