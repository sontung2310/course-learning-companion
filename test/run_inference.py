"""
Simple script to run the LearningOrchestrator pipeline on a user question.

Usage (from project root):

  uv run python -m test.run_inference "What is gradient descent?"
  # or
  .venv/bin/python -m test.run_inference "What is gradient descent?"

Requires: `.env` configured, LiteLLM/proxy on OPENAI_BASE_URL (or LITELLM_*),
PostgreSQL, Redis, and Chroma env vars if you hit RAG — same as the API app.

If you omit the question argument, the script will prompt you to type one.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict


def _ensure_project_root_on_path() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


async def _run(question: str) -> Dict[str, Any]:
    """Run the full agent pipeline (with guardrails) for a single user question."""
    print("Importing nemoguardrails")
    from nemoguardrails import LLMRails, RailsConfig

    print("Importing orchestrator")

    from src.services.agents import LearningOrchestrator  # Imported after sys.path setup

    print("Creating orchestrator")
    orchestrator = LearningOrchestrator()
    config = RailsConfig.from_path("src/rails")
    rails_service = LLMRails(config)
    print(f"Rails service: {rails_service}")
    result = await orchestrator.generate(
        question=question,
        user_id="cli-user",
        session_id="cli-session",
        rails_service=rails_service,
    )
    print(f"Result: {result}")
    # generate returns a dict like {"response": "...", "use_rag": bool | None}
    if isinstance(result, dict) and "response" in result:
        return result
    return {"response": str(result)}


def main() -> None:
    _ensure_project_root_on_path()

    # Question from CLI arg or prompt
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    else:
        question = input("Enter your question: ").strip()

    if not question:
        print("No question provided, exiting.")
        return

    print(f"Running agent pipeline for question:\n  {question}\n")

    # Run the async pipeline
    try:
        result = asyncio.run(_run(question))
    except KeyboardInterrupt:
        print("Interrupted.")
        return

    answer = result.get("response", "")
    print("=== Agent pipeline answer ===")
    print(answer)


if __name__ == "__main__":
    main()

