"""
Standalone pipeline to test semantic cache: input text → check semantic cache → if miss, get answer and save (redis_cache decorator + semantic cache).

- Uses litellm RedisSemanticCache to check if the question hits the cache.
- If miss: calls a dummy generate_response (decorated with redis_cache.cache(ttl=10) from redis_cache.py), then saves the answer to the semantic cache.
- No agents, no pytest. Run from project root: python test/test_semantic_cache.py

Why "semantic cache MISS" every time (analysis):
  LiteLLM's async_set_cache and async_get_cache catch all exceptions and only call
  print_verbose(...), so errors are invisible unless verbose is on. Likely causes:
  1) async_set_cache fails silently (e.g. embedding call fails, or RedisVL astore fails)
     → nothing is stored → every get is a miss.
  2) Embedding API: wrong key, base URL, or model name so _get_async_embedding fails.
  3) Vector/schema mismatch: index was built with different dimensions than the embedding.
  4) Redis: wrong URL (e.g. missing DB, auth) or async client issue.

  To see the real error: run with LITELLM_LOG=DEBUG or set LITELLM_VERBOSE=1 below.

Log analysis (when you see MISS every time and no exception):
  - No "Error in async_set_cache" or "Error in async_get_cache" in the log means
    neither path is raising: set and get are completing without throwing.
  - So either (1) data is not actually being written (RedisVL astore doesn't persist),
    or (2) data is written but acheck doesn't find it (e.g. wrong index/prefix,
    vector dtype/dims, or distance threshold).
  - The line "SYNC kwargs[caching]: False; litellm.cache: None; ..." is from the
    embedding call (cache options), so embeddings are being called.
  - Run the script again and check the [diagnostic] line: if "Redis keys
    matching litellm_semantic_cache_index*: 0" after the first call, the write
    is not persisting; if count > 0 but still MISS, the read/query path is wrong.
  - If writes are not persisting (0 keys), run with SURFACE_SEMANTIC_CACHE_ERROR=1
    to re-raise the real error from async_set_cache (e.g. embedding or RedisVL).
"""
import asyncio
import json
import os
import sys
from typing import Optional

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Surface errors that LiteLLM otherwise swallows (run: LITELLM_VERBOSE=1 python test/test_semantic_cache.py)
if os.environ.get("LITELLM_VERBOSE") or os.environ.get("LITELLM_LOG") == "DEBUG":
    import litellm
    litellm.set_verbose = True

from litellm.caching.redis_semantic_cache import RedisSemanticCache
from litellm.litellm_core_utils.prompt_templates.common_utils import get_str_from_messages

from src.settings import SETTINGS
from src.services.redis_cache import redis_cache


def _redis_url() -> str:
    password = (
        SETTINGS.REDIS_PASSWORD.get_secret_value()
        if SETTINGS.REDIS_PASSWORD is not None
        else ""
    )
    return f"redis://:{password}@{SETTINGS.REDIS_HOST}:{SETTINGS.REDIS_PORT}"


def _make_semantic_cache() -> RedisSemanticCache:
    return RedisSemanticCache(
        similarity_threshold=0.8,
        redis_url=_redis_url(),
        embedding_model="text-embedding-3-small",
    )


# Single semantic cache instance for check + save
_semantic_cache = _make_semantic_cache()
count = sum(1 for _ in redis_cache.client.scan_iter(match=f"mlops:{SETTINGS.ENVIRONMENT}:*"))
print("mlops cache items:", count)

# Optional: re-raise errors so we see why writes don't persist (run with SURFACE_SEMANTIC_CACHE_ERROR=1)
if os.environ.get("SURFACE_SEMANTIC_CACHE_ERROR"):

    async def _async_set_cache_raise(self: RedisSemanticCache, key: str, value: object, **kwargs: object) -> None:
        messages = kwargs.get("messages", [])
        if not messages:
            return
        prompt = get_str_from_messages(messages)
        value_str = str(value)
        prompt_embedding = await self._get_async_embedding(prompt, **kwargs)
        ttl = self._get_ttl(**kwargs)
        await self.llmcache.astore(
            prompt, value_str, vector=prompt_embedding, ttl=ttl
        )

    _semantic_cache.async_set_cache = lambda key, value, **kwargs: _async_set_cache_raise(
        _semantic_cache, key, value, **kwargs
    )


@redis_cache.cache(ttl=100)
async def generate_response(
    question: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """
    Dummy responder: no agents, no LLM. Result is cached by redis_cache (exact key).
    """
    return {
        "answer": f"[Cached response for: {question}]",
        "source": "test_semantic_cache",
    }


async def run_pipeline(
    question: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """
    1) Check semantic cache (RedisSemanticCache).
    2) If hit → return cached answer.
    3) If miss → call generate_response (uses redis_cache.cache), then save answer to semantic cache.
    """
    messages = [{"role": "user", "content": question}]
    metadata = {"user_id": user_id, "session_id": session_id}
    cache_key = "semantic-test"

    # 1) Check semantic cache (LiteLLM swallows exceptions → wrap to surface errors)
    try:
        cached = await _semantic_cache.async_get_cache(
            cache_key,
            messages=messages,
            metadata=metadata,
        )
    except Exception as e:
        print(f"ERROR during semantic cache GET: {e}")
        raise

    if cached is not None:
        print("Semantic cache HIT.")
        if isinstance(cached, dict):
            return cached
        try:
            return json.loads(cached) if isinstance(cached, str) else cached
        except Exception:
            return {"answer": str(cached), "source": "semantic_cache"}

    # 2) Miss: get answer (decorator saves to redis_cache by exact key)
    print("Semantic cache MISS. Calling generate_response (redis_cache will cache result).")
    result = await generate_response(
        question=question,
        session_id=session_id,
        user_id=user_id,
    )

    # 3) Save to semantic cache (LiteLLM swallows exceptions → wrap to surface errors)
    value_to_store = json.dumps(result)
    try:
        await _semantic_cache.async_set_cache(
            cache_key,
            value_to_store,
            messages=messages,
            metadata=metadata,
            ttl=10,
        )
    except Exception as e:
        print(f"ERROR during semantic cache SET: {e}")
        raise
    print("Saved answer to semantic cache (ttl=10).")
    count = sum(1 for _ in redis_cache.client.scan_iter(match=f"mlops:{SETTINGS.ENVIRONMENT}:*"))
    print("mlops cache items after save:", count)
    return result


def _diagnose_redis_semantic_keys() -> None:
    """Print how many Redis keys exist for the semantic cache (semantic cache uses db=0)."""
    try:
        import redis
        password = SETTINGS.REDIS_PASSWORD.get_secret_value() if SETTINGS.REDIS_PASSWORD else None
        # Semantic cache redis_url has no /N → db=0. Scan db 0 and 1 to be sure.
        for db in (0, getattr(SETTINGS, "REDIS_DB_CACHE", 1)):
            r = redis.Redis(
                host=SETTINGS.REDIS_HOST,
                port=SETTINGS.REDIS_PORT,
                password=password or "",
                db=db,
                decode_responses=True,
            )
            keys = r.keys("litellm_semantic_cache_index*")
            print(f"  [diagnostic] Redis db={db} keys litellm_semantic_cache_index*: {len(keys)}")
            if keys:
                print(f"  [diagnostic] Example key: {keys[0]!r}")
                break
    except Exception as e:
        print(f"  [diagnostic] Could not list Redis keys: {e}")


async def main() -> None:
    question = "What's the capital of France?"
    print(f"Input: {question}\n")

    print("--- First call (expect MISS, then save) ---")
    out1 = await run_pipeline(question=question, user_id="u1", session_id="s1")
    print(f"Result: {out1}")
    _diagnose_redis_semantic_keys()
    print()

    print("--- Second call same question (expect semantic HIT) ---")
    out2 = await run_pipeline(question=question, user_id="u1", session_id="s1")
    print(f"Result: {out2}")
    _diagnose_redis_semantic_keys()
    print()

    print("--- Third call similar question (expect semantic HIT) ---")
    similar = "What's the capital of France?"
    out3 = await run_pipeline(question=similar, user_id="u1", session_id="s1")
    print(f"Result: {out3}\n")

    print("--- Third call similar question (expect semantic HIT) ---")
    similar = "Where Melbourne is located?"
    out4 = await run_pipeline(question=similar, user_id="u1", session_id="s1")
    print(f"Result: {out4}\n")

    print("--- Third call similar question (expect semantic HIT) ---")
    similar = "Where Melbourne is located?"
    out5 = await run_pipeline(question=similar, user_id="u1", session_id="s1")
    print(f"Result: {out5}\n")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
