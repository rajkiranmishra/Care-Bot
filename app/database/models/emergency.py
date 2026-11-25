"""SQLAlchemy model for persistent, session-independent active emergencies."""

from datetime import datetime
from typing import Optional
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.database.base import Base, TimestampMixin


class ActiveEmergency(Base, TimestampMixin):
    """Database model persisting active critical/urgent emergencies independently of conversation sessions.

    SAFETY & ARCHITECTURAL INVARIANT:
    An active emergency is anchored to a resident (user_id), NOT a single conversation session.
    Starting a new session, changing operating mode, restarting the host application, or sending
    unrelated messages CANNOT silently clear or reset an active emergency.
    Only an explicit, authorized caregiver resolution procedure can clear it.
    """
    __tablename__ = "active_emergencies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    trigger_phrase: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        default="active",
        nullable=False,
    )  # "active", "pending_resolution", "resolved"
    user_affirmation: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by_user_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<ActiveEmergency id={self.id} user_id={self.user_id} severity='{self.severity}' "
            f"status='{self.status}' is_resolved={self.is_resolved}>"
        )
