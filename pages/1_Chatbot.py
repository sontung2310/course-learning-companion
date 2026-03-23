from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, Generator

import requests
import streamlit as st


st.set_page_config(page_title="Chatbot", page_icon="💬", layout="wide")


if not st.session_state.get("user_profile"):
    st.error("You must log in first on the **Course Learning Assistant** home page.")
    st.stop()


def _default_api_base_url() -> str:
    # FastAPI defaults from src/settings.py: PORT=8055, API_V1_STR="/v1"
    return os.getenv("AIDE_API_BASE_URL", "http://localhost:8055/v1").rstrip("/")


def _post_chat(
    api_base_url: str, user_input: str, session_id: str, user_id: str
) -> Dict[str, Any]:
    url = f"{api_base_url}/agents/personalized-learning"
    payload = {
        "user_input": user_input,
        "session_id": session_id or None,
        "user_id": user_id or None,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def _post_chat_stream(
    api_base_url: str, user_input: str, session_id: str, user_id: str
) -> Generator[str, None, None]:
    """Call SSE streaming endpoint and yield text chunks for st.write_stream()."""
    url = f"{api_base_url}/agents/personalized-learning/stream"
    payload = {
        "user_input": user_input,
        "session_id": session_id or None,
        "user_id": user_id or None,
    }
    st.session_state["_last_stream_meta"] = {}

    with requests.post(url, json=payload, stream=True, timeout=180) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")
            if event_type == "chunk":
                content = event.get("content", "")
                if content:
                    yield content
            elif event_type == "final":
                st.session_state["_last_stream_meta"] = {
                    "use_rag": event.get("use_rag"),
                    "from_cache": event.get("from_cache", False),
                }
                # Cache hits only emit ``final`` (no token chunks). ``write_stream`` only
                # renders yielded strings, so surface the full cached answer here.
                if event.get("from_cache"):
                    full = (event.get("response") or "").strip()
                    if full:
                        yield full
            elif event_type == "error":
                raise RuntimeError(event.get("message", "Streaming error from server"))


st.title("Chatbot")

with st.sidebar:
    profile = st.session_state.get("user_profile", {})

    st.subheader("Connection")
    default_base = st.session_state.get("api_base_url", _default_api_base_url())
    api_base_url = st.text_input("API base URL", value=default_base)
    st.session_state["api_base_url"] = api_base_url
    response_mode = st.radio(
        "Response mode",
        options=["Non-streaming", "Streaming (SSE)"],
        index=1,
        help="Non-streaming calls generate(); Streaming (SSE) calls generate_stream().",
    )
    use_streaming = response_mode == "Streaming (SSE)"
    if use_streaming:
        st.caption("Endpoint: `POST /v1/agents/personalized-learning/stream`")
    else:
        st.caption("Endpoint: `POST /v1/agents/personalized-learning`")

    st.subheader("Identity")
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    session_id = st.text_input("session_id", value=st.session_state.session_id)
    user_id = profile.get("user_id", "")
    st.text_input("user_id", value=user_id, disabled=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("New session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    with col_b:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Type your question and press Enter…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if use_streaming:
            try:
                answer = st.write_stream(
                    _post_chat_stream(
                        api_base_url=api_base_url,
                        user_input=prompt,
                        session_id=session_id,
                        user_id=user_id,
                    )
                )
                meta = st.session_state.get("_last_stream_meta", {})
                use_rag = meta.get("use_rag")
                if meta.get("from_cache"):
                    st.caption("_Served from cache_")
                elif use_rag is not None:
                    st.caption(f"use_rag: `{use_rag}`")
            except (requests.RequestException, RuntimeError) as e:
                answer = f"Request failed: `{e}`"
                use_rag = None
                st.markdown(answer)
        else:
            with st.spinner("Thinking..."):
                try:
                    data = _post_chat(
                        api_base_url=api_base_url,
                        user_input=prompt,
                        session_id=session_id,
                        user_id=user_id,
                    )
                    answer = (data or {}).get("response") or "(No response)"
                    use_rag = (data or {}).get("use_rag")
                except requests.RequestException as e:
                    answer = f"Request failed: `{e}`"
                    use_rag = None

            st.markdown(answer)
            if use_rag is not None:
                st.caption(f"use_rag: `{use_rag}`")

    st.session_state.messages.append({"role": "assistant", "content": answer})

