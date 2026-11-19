"""Database engine, session factory, and connection lifecycle."""

from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from app.config import get_settings

settings = get_settings()

connect_args = {}
if settings.is_sqlite:
    # Disable thread checks for SQLite in asynchronous/threaded FastAPI requests
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=settings.DEBUG and settings.ENVIRONMENT == "development",
)

# Apply SQLite-specific pragmas for concurrency and data integrity
if settings.is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        # Enable Write-Ahead Logging (WAL) to permit concurrent reads during writes
        cursor.execute("PRAGMA journal_mode=WAL;")
        # Enable foreign key constraint enforcement in SQLite
        cursor.execute("PRAGMA foreign_keys=ON;")
        # Synchronous NORMAL provides sufficient durability while optimizing disk I/O
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.close()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """Dependency provider yielding a scoped database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
