"""Long-term memory service using PostgreSQL for persistent storage."""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from sqlalchemy import Column, String, Text, Integer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.future import select
from src.settings import SETTINGS
from sqlalchemy.orm import mapped_column

Base = declarative_base()

class UserProfile(Base):
    """User profile table for storing learning preferences and history."""

    __tablename__ = "user_profiles"

    user_id = Column(String, primary_key=True)
    name = Column(String)
    course_intake = Column(JSONB)  # List of intaking courses
    interests = Column(JSONB)  # List of subjects/topics
    created_at = Column(
        TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )



class LongTermMemoryService:
    """Service for managing long-term memory using PostgreSQL."""

    def __init__(self):
        # Create async database URL
        password = SETTINGS.POSTGRES_PASSWORD.get_secret_value()
        self.database_url = (
            f"postgresql+asyncpg://{SETTINGS.POSTGRES_USER}:{password}@"
            f"{SETTINGS.POSTGRES_HOST}:{SETTINGS.POSTGRES_PORT}/{SETTINGS.POSTGRES_DB}"
        )
        # self.database_url = (
        #     f"postgresql://{SETTINGS.POSTGRES_USER}:{password}@"
        #     f"{SETTINGS.POSTGRES_HOST}:{SETTINGS.POSTGRES_PORT}/{SETTINGS.POSTGRES_DB}"
        # )
        print(f"Database URL: {self.database_url}")  # Debugging line
        # Create async engine and session
        self.engine = create_async_engine(self.database_url)
        self.async_session = async_sessionmaker(self.engine, class_=AsyncSession)
        print(f"Engine: {self.engine}")  # Debugging line

    async def create_tables(self):
        """Create tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def initialize(self):
        """Initialize the service by creating tables."""
        await self.create_tables()
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile by user_id."""
        async with self.async_session() as session:
            result = await session.execute(
                select(UserProfile).filter(UserProfile.user_id == user_id)
            )
            profile = result.scalar_one_or_none()
            if profile:
                return {
                    "user_id": profile.user_id,
                    "name": profile.name,
                    "course_intake": profile.course_intake,
                    "interests": profile.interests,
                }
            return None
    
    async def create_or_update_user_profile(
        self, user_id: str, profile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create or update a user profile."""
        async with self.async_session() as session:
            result = await session.execute(
                select(UserProfile).filter(UserProfile.user_id == user_id)
            )
            profile = result.scalar_one_or_none()

            if profile:
                for key, value in profile_data.items():
                    setattr(profile, key, value)
                profile.updated_at = datetime.now(timezone.utc)
            else:
                profile = UserProfile(user_id=user_id, **profile_data)
                session.add(profile)

            print(f"Profile data: {profile_data}")  # Debugging line

            await session.commit()
            await session.refresh(profile)
            return {
                "user_id": profile.user_id,
                "name": profile.name,
                "course_intake": profile.course_intake,
                "interests": profile.interests,
                "created_at": profile.created_at,
                "updated_at": profile.updated_at,
            }

if __name__ == "__main__":
    async def main() -> None:
        svc = LongTermMemoryService()

        # 1) Check DB connectivity
        try:
            async with svc.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            print("DB connectivity: OK")
        except Exception as e:
            print(f"DB connectivity: FAILED ({e})")
            return

        # 2) Check we can create tables
        try:
            await svc.create_tables()
            print("DB create tables: OK")
        except Exception as e:
            print(f"DB create tables: FAILED ({e})")
            return

        # 3) Check we can add a new user (insert/update)
        user_id = f"smoke_{uuid.uuid4().hex}"
        try:
            await svc.create_or_update_user_profile(
                user_id=user_id,
                profile_data={
                    "name": "Smoke Test User",
                    "course_intake": ["CS101"],
                    "interests": ["databases"],
                },
            )
            print(f"DB add user_profile: OK (user_id={user_id})")
        except Exception as e:
            print(f"DB add user_profile: FAILED ({e})")
            return

        # 4) Check get_user_profile
        try:
            profile = await svc.get_user_profile(user_id)
            print(f"DB get_user_profile: OK (result={profile})")
        except Exception as e:
            print(f"DB get_user_profile: FAILED ({e})")

    asyncio.run(main())
