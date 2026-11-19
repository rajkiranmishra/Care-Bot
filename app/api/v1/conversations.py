"""Conversation context, multi-turn session, and message REST endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.conversation.schemas import (
    MessageCreate,
    MessageResponse,
    SessionCreate,
    SessionResponse,
    SessionUpdate,
    UserContextPayload,
)
from app.conversation.service import (
    ConversationService,
    SessionAccessDeniedError,
    SessionNotFoundError,
)
from app.core.context_engine import ContextEngine
from app.database.models.conversation import SessionStatus
from app.users.service import UserNotFoundError

router = APIRouter(prefix="/conversations", tags=["Conversation Context"])


@router.post(
    "/users/{user_id}/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new dialogue session for a user",
)
def create_session(
    user_id: int,
    session_in: SessionCreate,
    db: Session = Depends(get_db),
) -> SessionResponse:
    """Initialize a new persistent multi-turn conversation session."""
    try:
        session = ConversationService.create_session(db, user_id, session_in)
        return SessionResponse.model_validate(session)
    except UserNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get(
    "/users/{user_id}/sessions",
    response_model=List[SessionResponse],
    summary="List all dialogue sessions for a user",
)
def list_user_sessions(
    user_id: int,
    status_filter: Optional[SessionStatus] = Query(default=None, alias="status"),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> List[SessionResponse]:
    """Retrieve all conversation sessions belonging to the user."""
    sessions = ConversationService.list_sessions(db, user_id, status=status_filter, limit=limit)
    return [SessionResponse.model_validate(s) for s in sessions]


@router.get(
    "/users/{user_id}/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Get session details and messages",
)
def get_user_session(
    user_id: int,
    session_id: str,
    db: Session = Depends(get_db),
) -> SessionResponse:
    """Fetch session details, enforcing user data isolation."""
    try:
        session = ConversationService.get_session(db, user_id, session_id)
        return SessionResponse.model_validate(session)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except SessionAccessDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.put(
    "/users/{user_id}/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Update session status or state",
)
def update_user_session(
    user_id: int,
    session_id: str,
    session_update: SessionUpdate,
    db: Session = Depends(get_db),
) -> SessionResponse:
    """Update conversation state or metadata for an active session."""
    try:
        session = ConversationService.update_session(db, user_id, session_id, session_update)
        return SessionResponse.model_validate(session)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except SessionAccessDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.post(
    "/users/{user_id}/sessions/{session_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add an utterance to a session",
)
def post_message(
    user_id: int,
    session_id: str,
    message_in: MessageCreate,
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Append a user or assistant message to the persistent session history."""
    try:
        message = ConversationService.add_message(db, user_id, session_id, message_in)
        return MessageResponse.model_validate(message)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except SessionAccessDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.get(
    "/users/{user_id}/sessions/{session_id}/messages",
    response_model=List[MessageResponse],
    summary="Get chronological messages for a session",
)
def get_session_messages(
    user_id: int,
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> List[MessageResponse]:
    """Retrieve chronological dialogue history for a session with ownership checks."""
    try:
        messages = ConversationService.get_messages(db, user_id, session_id, limit=limit)
        return [MessageResponse.model_validate(m) for m in messages]
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except SessionAccessDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.get(
    "/users/{user_id}/context",
    response_model=UserContextPayload,
    summary="Synthesize complete LLM context payload for user",
)
def get_user_llm_context(
    user_id: int,
    history_window: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
) -> UserContextPayload:
    """Compile structured context (profile, operating mode, language, history) for local LLMs."""
    try:
        return ContextEngine.get_user_context(db, user_id, history_window=history_window)
    except UserNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
