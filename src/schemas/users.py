from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class UserProfileInput(BaseModel):
    user_id: str = Field(
        description="Unique user identifier", examples=["user123", "john_doe"]
    )
    name: str = Field(description="User's name", examples=["John Doe", "Alice Smith"])
    course_intake: List[str] = Field(
        description="User's preferred course intake",
        examples=[["CS101", "CS102"], ["MATH101", "MATH102"]],
    )
    interests: Optional[List[str]] = Field(
        description="User's learning interests and subjects",
        examples=[["programming", "data science"], ["mathematics", "physics"]],
    )


class UserProfileResponse(BaseModel):
    user_id: str
    name: str
    course_intake: List[str]
    interests: List[str]
    created_at: str
    updated_at: str
