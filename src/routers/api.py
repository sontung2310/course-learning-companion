from fastapi import APIRouter
from src.routers import agents, users

api_router = APIRouter()
api_router.include_router(users.router, prefix="/users", tags=["User Management"])
api_router.include_router(agents.router, prefix="/agents", tags=["Agent Orchestrator"])
