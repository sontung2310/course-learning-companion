from crewai.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from src.services.memory import ShortTermMemoryService
import json
import asyncio
import logging
import concurrent.futures

logger = logging.getLogger(__name__)

# Global memory service instance
memory_service = ShortTermMemoryService()

class ShortTermMemoryInput(BaseModel):
    """Input schema for short-term memory tool."""

    action: str = Field(
        ...,
        description="Action to perform: get_conversation_context, get_user_session, get_recommendations",
    )
    session_id: Optional[str] = Field(None, description="Session ID (optional)")
    user_id: Optional[str] = Field(None, description="User ID (optional)")

class ShortTermMemoryTool(BaseTool):
    """Tool for accessing short-term memory (Redis)."""

    name: str = "short_term_memory"
    description: str = "Access temporary session data, conversation context, and agent states from short-term memory"
    args_schema: Type[BaseModel] = ShortTermMemoryInput

    def _run(
        self,
        action: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Run short-term memory operations."""
        try:
            # Check if we're already in an async context
            try:
                asyncio.get_running_loop()
                # We're in an async context, need to use a different approach
                return self._run_in_thread(action, session_id, user_id)
            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                return self._run_async(action, session_id, user_id)

        except Exception as e:
            logger.error(f"Error in short-term memory tool: {e}")
            return f"Error accessing short-term memory: {str(e)}"
    
    def _run_async(
        self,
        action: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Run async operations when no event loop is running."""
        return asyncio.run(self._async_operation(action, session_id, user_id))

    def _run_in_thread(
        self,
        action: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Run operations in a thread when in async context."""

        def _sync_operation():
            # Create a new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    self._async_operation(action, session_id, user_id)
                )
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_sync_operation)
            return future.result(timeout=30)  # 30 second timeout

    async def _async_operation(
        self,
        action: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Actual async operations."""
        if action == "get_conversation_context" and session_id and user_id:
            context = await memory_service.get_conversation_context(session_id, user_id)
            return json.dumps(context) if context else "No context found"

        elif action == "get_user_session" and user_id:
            session = await memory_service.get_user_session(user_id)
            return json.dumps(session) if session else "No session found"

        else:
            return f"Unknown action or missing parameters: {action}"