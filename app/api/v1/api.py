"""Central API router aggregating all v1 sub-routers."""

from fastapi import APIRouter
from app.api.v1 import conversations, health, users

api_router = APIRouter()

# Register core v1 sub-routers
api_router.include_router(health.router)
api_router.include_router(users.router)
api_router.include_router(conversations.router)
