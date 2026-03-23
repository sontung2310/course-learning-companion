from src.services.memory import LongTermMemoryService, ShortTermMemoryService
from typing import Optional


class Users:
    def __init__(self):
        self.long_term_memory = LongTermMemoryService()
        self.short_term_memory = ShortTermMemoryService()

    async def create_user_profile(
        self,
        user_id: str,
        name: str,
        course_intake: list,
        interests: Optional[list] = None,
    ):
        """Create or update a user profile for personalized learning."""
        profile_data = {
            "name": name,
            "course_intake": course_intake,
            "interests": interests or [],
        }

        await self.long_term_memory.create_or_update_user_profile(user_id, profile_data)
        return profile_data

    async def get_user_profile(self, user_id: str):
        """Get user profile for personalization."""
        return await self.long_term_memory.get_user_profile(user_id)

    async def clear_user_session(self, user_id: str):
        """Clear user's short-term memory cache."""
        return await self.short_term_memory.clear_user_cache(user_id)
