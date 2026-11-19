"""Database persistence layer for CARE-BOT."""

from app.database.base import Base
from app.database.session import engine, get_db, SessionLocal
import app.database.models  # Ensures all ORM models are registered with Base.metadata

__all__ = ["Base", "engine", "get_db", "SessionLocal"]
