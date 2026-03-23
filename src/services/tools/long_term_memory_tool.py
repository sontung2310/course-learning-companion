from crewai.tools import BaseTool
from src.services.memory import LongTermMemoryService
from typing import Type
from pydantic import BaseModel, Field
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

# Global memory service instance
memory_service = LongTermMemoryService()

class LongTermMemoryInput(BaseModel):
    """Input schema for long-term memory tool."""

    action: str = Field(
        ...,
        description="Action to perform: get_profile",
    )
    user_id: str = Field(..., description="User ID for the memory operation")
    limit: int = Field(10, description="Limit for history queries (optional)")

class LongTermMemoryTool(BaseTool):
    """Tool for accessing long-term memory (PostgreSQL)."""

    name: str = "long_term_memory"
    description: str = "Access user's learning history, profile, and knowledge graph from long-term memory"
    args_schema: Type[BaseModel] = LongTermMemoryInput

    def _run(self, action: str, user_id: str, limit: int = 10) -> str:
        """Run memory operations synchronously."""
        try:
            # Check if we're already in an async context
            try:
                asyncio.get_running_loop()
                # We're in an async context, need to use a different approach
                return self._run_in_thread(action, user_id, limit)
            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                return self._run_async(action, user_id, limit)

        except Exception as e:
            logger.error(f"Error in long-term memory tool: {e}")
            return f"Error accessing long-term memory: {str(e)}"
    
    def _run_async(self, action: str, user_id: str, limit: int = 10) -> str:
        """Run async operations when no event loop is running."""

        async def _async_operation():
            if action == "get_profile":
                profile = await memory_service.get_user_profile(user_id)
                return json.dumps(profile) if profile else "No profile found"

            else:
                return f"Unknown action: {action}"

        return asyncio.run(_async_operation())
    
    def _run_in_thread(self, action: str, user_id: str, limit: int = 10) -> str:
        """Run operations in a thread when in async context."""
        import concurrent.futures

        def _sync_operation():
            # Create a new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    self._async_operation(action, user_id, limit)
                )
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_sync_operation)
            return future.result(timeout=30)  # 30 second timeout
    
    async def _async_operation(self, action: str, user_id: str, limit: int = 10) -> str:
        """Actual async operations."""
        if action == "get_profile":
            profile = await memory_service.get_user_profile(user_id)
            return json.dumps(profile) if profile else "No profile found"


        else:
            return f"Unknown action: {action}"

