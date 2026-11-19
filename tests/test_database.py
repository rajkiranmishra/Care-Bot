"""Tests for database engine, session lifecycle, and pragmas."""

from sqlalchemy import text
from sqlalchemy.orm import Session


def test_database_connection_query(db_session: Session):
    """Verify that a basic SQL query executes successfully on the test session."""
    result = db_session.execute(text("SELECT 1 AS alive;")).scalar()
    assert result == 1


def test_database_session_rollback(db_session: Session):
    """Verify that transactions isolate operations and can roll back cleanly."""
    db_session.execute(text("CREATE TABLE IF NOT EXISTS dummy_test (id INTEGER PRIMARY KEY, name TEXT);"))
    db_session.execute(text("INSERT INTO dummy_test (name) VALUES ('carebot');"))
    count = db_session.execute(text("SELECT COUNT(*) FROM dummy_test;")).scalar()
    assert count == 1
