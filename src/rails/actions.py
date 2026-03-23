from typing import Optional
from nemoguardrails.actions import action
from src.services.agents import LearningOrchestrator

# Global orchestrator instance for handling user questions
orchestrator = LearningOrchestrator()


async def get_query_response(
    orchestrator: LearningOrchestrator,
    query: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
):
    """
    Use the learning orchestrator to answer the user's question.
    """
    result = await orchestrator.answer_question(
        query, session_id=session_id, user_id=user_id
    )
    # The orchestrator returns a dict like {"response": str, "use_rag": bool}
    return result.get("response", "")


@action(is_system_action=True)
async def user_query(context: Optional[dict] = None):
    """
    Function to invoke the QA chain to query user message.
    """
    if not context or "user_message" not in context:
        return "No user message provided in context."

    user_message = context.get("user_message")
    print("user_message is ", user_message)

    if not user_message:
        return "User message is empty."

    session_id = context.get("session_id") if context else None
    user_id = context.get("user_id") if context else None
    print(f"Session ID: {session_id}, User ID: {user_id} in user_query action")
    return await get_query_response(
        orchestrator, user_message, session_id=session_id, user_id=user_id
    )
