"""Conversation context, multi-turn session, and message REST endpoints with security enforcement."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, verify_user_access
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
from app.database.models.conversation import MessageRole, SessionStatus
from app.database.models.user import User, UserRole
from app.intelligence import ChatTurnRequest, ChatTurnResponse, ConversationEngine
from app.intelligence.emergency_service import EmergencyService
from app.users.service import UserService, UserNotFoundError

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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SessionResponse:
    """Initialize a new persistent multi-turn conversation session."""
    verify_user_access(user_id, current_user)

    if current_user.role == UserRole.FAMILY_VIEWER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Family viewers have read-only access and cannot initialize conversation sessions",
        )

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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[SessionResponse]:
    """Retrieve conversation sessions belonging to the user with ownership verification."""
    verify_user_access(user_id, current_user)
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SessionResponse:
    """Fetch session details, enforcing user data isolation and ownership."""
    verify_user_access(user_id, current_user)
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SessionResponse:
    """Update conversation state or metadata for an active session."""
    verify_user_access(user_id, current_user)

    if current_user.role == UserRole.FAMILY_VIEWER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Family viewers have read-only access and cannot modify conversation sessions",
        )

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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Append a user utterance to the session history.

    SECURITY RULE: Clients are strictly prohibited from impersonating 'assistant' or 'system' roles.
    Only role='user' is permitted from external API clients.
    """
    verify_user_access(user_id, current_user)

    if current_user.role == UserRole.FAMILY_VIEWER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Family viewers have read-only access and cannot post conversation messages",
        )

    if message_in.role != MessageRole.USER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Client impersonation forbidden: Clients may only post messages with role 'user'. Assistant messages must be generated internally by the CARE-BOT intelligence engine.",
        )

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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[MessageResponse]:
    """Retrieve chronological dialogue history for a session with ownership checks."""
    verify_user_access(user_id, current_user)
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserContextPayload:
    """Compile structured context for local LLMs, enforcing user ownership."""
    verify_user_access(user_id, current_user)
    try:
        return ContextEngine.get_user_context(db, user_id, history_window=history_window)
    except UserNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post(
    "/users/{user_id}/chat",
    response_model=ChatTurnResponse,
    summary="Execute a conversational interaction turn with CARE-BOT",
)
async def chat_turn(
    user_id: int,
    request: ChatTurnRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatTurnResponse:
    """Execute a conversational interaction turn through the ConversationEngine.

    SECURITY INVARIANTS:
    1. Authenticated caller identity and verify_user_access determine permission.
    2. Any arbitrary user_id supplied in a request body is completely ignored.
    3. Family Viewers have read-only privileges and cannot execute chat turns (which mutate dialogue state).
    4. Deterministic emergency detection executes before any local LLM generation.
    """
    verify_user_access(user_id, current_user)

    if current_user.role == UserRole.FAMILY_VIEWER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Family viewers have read-only access and cannot execute conversational turns",
        )

    try:
        target_user = UserService.get_by_id(db, user_id)
    except UserNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    engine = ConversationEngine()
    try:
        return await engine.process_turn(db, target_user, request)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except SessionAccessDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.post(
    "/users/{user_id}/emergency/resolve",
    summary="Resolve an active emergency for a resident (Caregiver only)",
)
def resolve_resident_emergency(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Execute authorized caregiver resolution procedure to clear an active emergency.

    SECURITY RULE:
    Only authenticated Caregivers can resolve an active emergency.
    Residents and Family Viewers are strictly prohibited (403 Forbidden).
    """
    if current_user.role != UserRole.CAREGIVER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only authorized caregivers can resolve emergencies",
        )

    try:
        UserService.get_by_id(db, user_id)
    except UserNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    resolved = EmergencyService.resolve_emergency(db, user_id, current_user)
    if not resolved:
        return {"status": "none_active", "user_id": user_id, "detail": "No active emergency found for resident"}

    return {
        "status": "resolved",
        "emergency_id": resolved.id,
        "user_id": user_id,
        "severity": resolved.severity,
        "resolved_by": current_user.id,
        "resolved_at": resolved.resolved_at.isoformat() if resolved.resolved_at else None,
    }

