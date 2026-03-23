"""Short-term memory service using Redis for temporary storage."""

import json
import asyncio
import redis.asyncio as redis
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from src.settings import SETTINGS


class ShortTermMemoryService:
    """Service for managing short-term memory using Redis."""

    def __init__(self):
        # Create Redis connection
        self.redis_client = redis.Redis(
            host=SETTINGS.REDIS_HOST,
            port=SETTINGS.REDIS_PORT,
            db=SETTINGS.REDIS_DB,
            password=SETTINGS.REDIS_PASSWORD.get_secret_value()
            if SETTINGS.REDIS_PASSWORD
            else None,
            decode_responses=True,
        )
    
    async def store_conversation_context(
        self, 
        session_id,
        user_id,
        context: Dict[str, Any],
        ttl_minutes: int = 60,
    ) -> None:
        """Store conversation context in Redis with ttl"""
        try:
            key = f"conversation:{session_id}:{user_id}"
            context_data = {
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "session_id": session_id,
            }
            await self.redis_client.setex(
                key, timedelta(minutes=ttl_minutes), json.dumps(context_data)
            )
            return True
        except Exception as e:
            print(f"Error storing conversation context: {e}")
            return False
    
    async def get_conversation_context(
        self,
        session_id,
        user_id,
    ) -> Optional[Dict[str, Any]]:
        """Get conversation context from Redis"""
        try:
            key = f"conversation:{session_id}:{user_id}"
            context_data = await self.redis_client.get(key)
            if context_data:
                return json.loads(context_data)
            return None
        
        except Exception as e:
            print(f"Error getting conversation context: {e}")
            return None

    
    async def store_user_session(
        self,
        user_id: str,
        session_data: Dict[str, Any],
        ttl_minutes: int = 60,
    ) -> None:
        """Store user session in Redis with ttl"""
        try:
            key = f"user_session:{user_id}"
            session_info = {
                "data": session_data,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
            }

            await self.redis_client.setex(
                key, timedelta(minutes=ttl_minutes), json.dumps(session_info)
            )
            return True
        except Exception as e:
            print(f"Error storing user session: {e}")
            return False
    
    async def get_user_session(
        self,
        user_id,
    ) -> Optional[Dict[str, Any]]:
        """Get user session from Redis"""
        try:
            key = f"user_session:{user_id}"
            session_info = await self.redis_client.get(key)
            if session_info:
                return json.loads(session_info)
            return None
        except Exception as e:
            print(f"Error getting user session: {e}")
            return None
    
    async def store_agent_state(
        self, agent_id: str, user_id: str, state: Dict[str, Any], ttl_minutes: int = 15
    ) -> bool:
        """Store agent state temporarily."""
        try:
            key = f"agent_state:{agent_id}:{user_id}"
            state_data = {
                "state": state,
                "timestamp": datetime.now().isoformat(),
                "agent_id": agent_id,
                "user_id": user_id,
            }

            await self.redis_client.setex(
                key, timedelta(minutes=ttl_minutes), json.dumps(state_data)
            )
            return True
        except Exception as e:
            print(f"Error storing agent state: {e}")
            return False

    async def get_agent_state(
        self, agent_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get agent state."""
        try:
            key = f"agent_state:{agent_id}:{user_id}"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"Error getting agent state: {e}")
            return None

    async def store_course_intake(
        self,
        user_id: str,
        course_name: str,
        ttl_minutes: int = 60,
    ) -> bool:
        """Store course intake in Redis with ttl"""
        try:
            key = f"course_intake:{user_id}"
            intake_data = {
                "course_name": course_name,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
            }

            await self.redis_client.setex(
                key, timedelta(minutes=ttl_minutes), json.dumps(intake_data)
            )
            return True
        except Exception as e:
            print(f"Error storing course intake: {e}")
            return False
    
    async def get_course_intake(
        self,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get course intake from Redis"""
        try:
            key = f"course_intake:{user_id}"
            intake_data = await self.redis_client.get(key)
            if intake_data:
                return json.loads(intake_data)
            return None
        except Exception as e:
            print(f"Error getting course intake: {e}")
            return None
    
    async def clear_user_cache(self, user_id: str) -> bool:
        """Clear all cached data for a user."""
        try:
            # Find all keys related to the user
            patterns = [
                f"conversation:*:{user_id}",
                f"user_session:{user_id}",
                f"agent_state:*:{user_id}",
                f"course_intake:{user_id}",
            ]

            for pattern in patterns:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)

            return True
        except Exception as e:
            print(f"Error clearing user cache: {e}")
            return False
    
    async def extend_session_ttl(
        self, session_id: str, user_id: str, additional_minutes: int = 30
    ) -> bool:
        """Extend the TTL of a session."""
        try:
            key = f"conversation:{session_id}:{user_id}"
            await self.redis_client.expire(key, timedelta(minutes=additional_minutes))
            return True
        except Exception as e:
            print(f"Error extending session TTL: {e}")
            return False

    async def get_active_sessions(self, user_id: str) -> List[str]:
        """Get all active sessions for a user."""
        try:
            pattern = f"conversation:*:{user_id}"
            keys = await self.redis_client.keys(pattern)
            # Extract session IDs from keys
            session_ids = []
            for key in keys:
                parts = key.split(":")
                if len(parts) >= 3:
                    session_ids.append(parts[1])
            return session_ids
        except Exception as e:
            print(f"Error getting active sessions: {e}")
            return []


async def main() -> None:
    """Basic self-test for Redis connectivity and core operations."""
    stm = ShortTermMemoryService()

    # Check Redis connectivity with PING
    try:
        pong = await stm.redis_client.ping()
        print(f"Redis connection OK: {pong}")
    except Exception as e:
        print(f"Redis connection FAILED: {e}")
        return

    test_user_id = "test_user"
    test_session_id = "test_session"

    # Test conversation context round-trip
    conversation_context = {"messages": ["hello", "world"], "meta": {"source": "self-test"}}
    print("\nTesting store_conversation_context / get_conversation_context ...")
    stored_conv = await stm.store_conversation_context(
        session_id=test_session_id,
        user_id=test_user_id,
        context=conversation_context,
        ttl_minutes=5,
    )
    print(f"store_conversation_context success: {stored_conv}")

    loaded_conv = await stm.get_conversation_context(
        session_id=test_session_id,
        user_id=test_user_id,
    )
    print(f"get_conversation_context returned: {loaded_conv}")

    # Test user session round-trip
    session_data = {"role": "tester", "last_login": datetime.now().isoformat()}
    print("\nTesting store_user_session / get_user_session ...")
    stored_session = await stm.store_user_session(
        user_id=test_user_id,
        session_data=session_data,
        ttl_minutes=5,
    )
    print(f"store_user_session success: {stored_session}")

    loaded_session = await stm.get_user_session(user_id=test_user_id)
    print(f"get_user_session returned: {loaded_session}")


if __name__ == "__main__":
    asyncio.run(main())