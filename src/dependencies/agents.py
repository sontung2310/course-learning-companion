from fastapi import Request
from src.services.agents import LearningOrchestrator


def get_learning_orchestrator(request: Request) -> LearningOrchestrator:
    return request.app.state.learning_orchestrator
