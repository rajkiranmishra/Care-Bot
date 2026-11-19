"""Health check and diagnostic endpoints."""

import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict
from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.api.deps import get_db, get_settings, Settings

router = APIRouter(tags=["Health & Diagnostics"])


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="System Health & Diagnostics",
    response_description="Current service health, database connectivity, and platform status",
)
def check_health(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """Execute a comprehensive system diagnostic check.

    Verifies:
    1. FastAPI application state.
    2. Active database connection via ping query.
    3. Host environment and Python runtime compatibility.
    """
    db_status = "connected"
    db_latency_ms: float = 0.0

    try:
        start_time = time.perf_counter()
        db.execute(text("SELECT 1;"))
        db_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    except Exception as exc:
        db_status = f"unhealthy: {str(exc)}"

    is_healthy = db_status == "connected"

    return {
        "status": "healthy" if is_healthy else "degraded",
        "service": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": {
            "status": db_status,
            "engine": "sqlite",
            "latency_ms": db_latency_ms,
        },
        "system": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
