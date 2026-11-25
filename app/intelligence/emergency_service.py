"""Service managing persistent, session-independent active emergencies and caregiver resolution."""

from datetime import datetime, timezone
import json
import logging
from typing import Optional
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models.conversation import ConversationSession
from app.database.models.emergency import ActiveEmergency
from app.database.models.user import User, UserRole
from app.intelligence.schemas import EmergencySeverity
from app.users.service import UserService

logger = logging.getLogger("carebot.intelligence.emergency_service")


class UnauthorizedResolutionError(Exception):
    """Raised when a non-caregiver user attempts to resolve an emergency."""
    pass


class EmergencyService:
    """Service providing durable emergency lifecycle management.

    SAFETY INVARIANT:
    An active emergency is persistent across dialogue sessions, operating mode changes,
    unrelated conversation turns, and application restarts.
    Only an explicit caregiver resolution procedure can clear it.
    """

    @staticmethod
    def get_active_emergency(db: Session, user_id: int) -> Optional[ActiveEmergency]:
        """Query currently active, unresolved emergency for a resident."""
        return db.execute(
            select(ActiveEmergency)
            .where(
                ActiveEmergency.user_id == user_id,
                ActiveEmergency.is_resolved == False,
            )
            .order_by(ActiveEmergency.id.desc())
            .limit(1)
        ).scalar_one_or_none()

    @staticmethod
    def record_active_emergency(
        db: Session,
        user_id: int,
        severity: EmergencySeverity,
        trigger_phrase: Optional[str],
        reasoning: str,
        user_affirmation: Optional[str] = None,
        pending_resolution: bool = False,
        commit: bool = True,
    ) -> ActiveEmergency:
        """Create or update durable active emergency state for a resident."""
        existing = EmergencyService.get_active_emergency(db, user_id)

        status_str = "pending_resolution" if pending_resolution else "active"

        if existing:
            # Update existing unresolved emergency
            if severity in (EmergencySeverity.CRITICAL, EmergencySeverity.URGENT):
                existing.severity = severity.value
            if trigger_phrase:
                existing.trigger_phrase = trigger_phrase
            existing.reasoning = reasoning
            existing.status = status_str
            if user_affirmation:
                existing.user_affirmation = user_affirmation
            emergency_record = existing
            if commit:
                try:
                    db.commit()
                    db.refresh(emergency_record)
                except Exception:
                    db.rollback()
                    raise
        else:
            # Use a savepoint (begin_nested) to isolate the insert.
            # If a concurrent transaction inserted an unresolved record for this resident in the
            # interim, the resulting IntegrityError from uq_active_emergency_user_unresolved
            # rolls back ONLY the savepoint, leaving the outer transaction clean and allowing
            # safe query and update of the existing record.
            try:
                with db.begin_nested():
                    emergency_record = ActiveEmergency(
                        user_id=user_id,
                        severity=severity.value,
                        trigger_phrase=trigger_phrase,
                        reasoning=reasoning,
                        status=status_str,
                        user_affirmation=user_affirmation,
                        is_resolved=False,
                    )
                    db.add(emergency_record)
                    db.flush()
            except IntegrityError:
                # Savepoint was automatically rolled back; outer transaction is healthy and intact.
                existing = EmergencyService.get_active_emergency(db, user_id)
                if existing:
                    if severity in (EmergencySeverity.CRITICAL, EmergencySeverity.URGENT):
                        existing.severity = severity.value
                    if trigger_phrase:
                        existing.trigger_phrase = trigger_phrase
                    existing.reasoning = reasoning
                    existing.status = status_str
                    if user_affirmation:
                        existing.user_affirmation = user_affirmation
                    emergency_record = existing
                else:
                    raise

            if commit:
                try:
                    db.commit()
                    db.refresh(emergency_record)
                except Exception:
                    db.rollback()
                    raise

        return emergency_record

    @staticmethod
    def resolve_emergency(
        db: Session,
        user_id: int,
        resolver: User,
        notes: Optional[str] = None,
    ) -> Optional[ActiveEmergency]:
        """Execute authorized resolution procedure to clear an active emergency.

        SECURITY RULE:
        1. Only an authorized caregiver may resolve an emergency.
        2. The caregiver must be explicitly assigned to the specific resident identified by user_id.
           A caregiver cannot resolve another caregiver's resident's emergency.
        """
        if resolver.role != UserRole.CAREGIVER:
            raise UnauthorizedResolutionError(
                f"User {resolver.id} with role '{resolver.role}' is not authorized to resolve emergencies. "
                "Only Caregivers can clear active emergencies."
            )

        if not UserService.is_caregiver_assigned_to_resident(db, caregiver_id=resolver.id, resident_id=user_id):
            raise UnauthorizedResolutionError(
                f"Caregiver {resolver.id} is not authorized to resolve emergencies for resident {user_id}. "
                "Explicit caregiver-to-resident assignment is required."
            )

        record = EmergencyService.get_active_emergency(db, user_id)
        if not record:
            return None

        record.is_resolved = True
        record.status = "resolved"
        record.resolved_at = datetime.now(timezone.utc)
        record.resolved_by_user_id = resolver.id
        record.resolution_notes = notes

        # Clean up cached active emergency from any active sessions belonging to this resident
        sessions = db.execute(
            select(ConversationSession).where(ConversationSession.user_id == user_id)
        ).scalars().all()
        for sess in sessions:
            if sess.context_metadata and "active_emergency" in sess.context_metadata:
                try:
                    meta = json.loads(sess.context_metadata)
                    if "active_emergency" in meta:
                        meta.pop("active_emergency", None)
                        sess.context_metadata = json.dumps(meta)
                except Exception:
                    pass

        try:
            db.commit()
            db.refresh(record)
        except Exception:
            db.rollback()
            raise

        logger.info(
            "Emergency %d for resident %d resolved by caregiver %d",
            record.id,
            user_id,
            resolver.id,
        )
        return record
