"""Common dependency injection providers for FastAPI route handlers."""

from typing import Generator
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.config import get_settings, Settings

__all__ = ["get_db", "get_settings", "Session", "Settings"]
