from src.services.users import Users
from fastapi import Request


def get_users_service(request: Request) -> Users:
    """Dependency to get the Users service from the FastAPI request context."""
    return request.app.state.users_service
