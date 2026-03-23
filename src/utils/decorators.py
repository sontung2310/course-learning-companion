"""Decorators for cross-cutting concerns (e.g. timing)."""

import asyncio
import time
from functools import wraps
from typing import Any, Callable, TypeVar

from src.utils.logger import logger

F = TypeVar("F", bound=Callable[..., Any])


def agent_response_time(func: F) -> F:
    """Decorator that measures total time for the agent's response to the user's question.

    Works with async functions. Logs the elapsed time and, if the return value is a dict,
    adds a 'agent_response_time_seconds' key so the API can expose it.
    """

    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(
            "Agent response completed in %.3fs | %s",
            elapsed,
            func.__qualname__,
        )
        return _attach_duration(result, elapsed)

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(
            "Agent response completed in %.3fs | %s",
            elapsed,
            func.__qualname__,
        )
        return _attach_duration(result, elapsed)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # type: ignore[return-value]
    return sync_wrapper  # type: ignore[return-value]


def _attach_duration(result: Any, elapsed_seconds: float) -> Any:
    if isinstance(result, dict):
        result = {**result, "agent_response_time_seconds": round(elapsed_seconds, 3)}
    return result
