from __future__ import annotations

import os
from crewai import LLM

# LiteLLM proxy (gateway)
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-llmops")


def get_qwen_llm() -> LLM:
    """
    LiteLLM client pinned to the local Qwen3-4B model (`vllm-qwen` in config.yaml).
    """
    return LLM(
        model="vllm-qwen",
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
    )


def get_gpt_api_llm() -> LLM:
    """
    LiteLLM client pinned to the OpenAI load-balanced group (`gpt-api` in config.yaml).
    """
    return LLM(
        model="gpt-api",
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
    )
