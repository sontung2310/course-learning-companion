from pydantic import BaseModel, Field
from typing import Optional


class PersonalizedLearningInput(BaseModel):
    user_input: str = Field(
        description="User's question about the courses, lectures, or technical questions",
        examples=[
            "Explain how neural networks work",
            "What do we learn in week 7 of CS336?",
            "How does a RAG system work?",
        ],
    )
    session_id: Optional[str] = Field(
        description="Session ID for tracking conversation context",
        default=None,
    )
    user_id: Optional[str] = Field(
        description="User ID for personalization",
        default=None,
    )
