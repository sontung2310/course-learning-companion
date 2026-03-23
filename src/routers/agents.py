import json
from typing import Any, AsyncIterator, Dict

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse

from src.dependencies.agents import get_learning_orchestrator
from src.schemas.agents import PersonalizedLearningInput
from src.services.agents import LearningOrchestrator


router = APIRouter()


@router.post(
    "/personalized-learning",
    status_code=status.HTTP_200_OK,
    response_model=Dict[str, Any],
)
async def ask_question(
    input: PersonalizedLearningInput,
    request: Request,
    orchestrator: LearningOrchestrator = Depends(get_learning_orchestrator),
) -> Dict[str, Any]:
    """Main entrypoint to query the learning agent (with optional NeMo Guardrails)."""
    rails_service = getattr(request.app.state, "rails", None)

    question: str = input.user_input
    session_id: str = input.session_id or ""
    user_id: str = input.user_id or ""

    result = await orchestrator.generate(
        question=question,
        user_id=user_id,
        session_id=session_id,
        rails_service=rails_service,
    )
    return result


async def _sse_generator(event_stream: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[str]:
    """Convert agent events to Server-Sent Events (text/event-stream) format."""
    try:
        async for event in event_stream:
            yield f"data: {json.dumps(event)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


@router.post("/personalized-learning/stream")
async def ask_question_stream(
    input: PersonalizedLearningInput,
    request: Request,
    orchestrator: LearningOrchestrator = Depends(get_learning_orchestrator),
) -> StreamingResponse:
    """Streaming entrypoint — returns Server-Sent Events (SSE).

    Event shapes emitted over the stream:

    - ``data: {"type": "chunk",  "content": "..."}``
      One TEXT token/fragment yielded as it is generated.

    - ``data: {"type": "final",  "response": "...", "use_rag": true|false|null, "from_cache": bool}``
      Emitted once after the last chunk; contains the full assembled response.

    - ``data: {"type": "error",  "message": "..."}``
      Emitted if an unexpected error occurs during generation.

    - ``data: [DONE]``
      Always the last line; signals the stream is closed.
    """
    # Intentionally bypass guardrails for streaming to ensure token/chunk events
    # come directly from CrewAI stream-enabled crews.
    rails_service = None

    event_stream = orchestrator.generate_stream(
        question=input.user_input,
        user_id=input.user_id or "",
        session_id=input.session_id or "",
        rails_service=rails_service,
    )

    return StreamingResponse(
        _sse_generator(event_stream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx proxy buffering
        },
    )




