"""Central API router aggregating all v1 sub-routers."""

from fastapi import APIRouter
from app.api.v1 import health

api_router = APIRouter()

# Include health routes
api_router.include_router(health.router)
