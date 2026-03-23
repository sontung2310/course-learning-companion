from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
import streamlit as st


st.set_page_config(
    page_title="Course Learning Assistant",
    page_icon="💬",
    layout="wide",
)


def _default_api_base_url() -> str:
    # FastAPI defaults from src/settings.py: PORT=8055, API_V1_STR="/v1"
    return os.getenv("AIDE_API_BASE_URL", "http://localhost:8055/v1").rstrip("/")


def _get_api_base_url() -> str:
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = _default_api_base_url()
    return st.session_state.api_base_url


def _set_api_base_url(value: str) -> None:
    st.session_state.api_base_url = value.rstrip("/")


def _get_user_profile(api_base_url: str, user_id: str) -> Optional[Dict[str, Any]]:
    url = f"{api_base_url}/users/user-profile/{user_id}"
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    # FastAPI returns null or {} when not found, depending on implementation
    if not data:
        return None
    return data


def _create_user_profile(
    api_base_url: str,
    user_id: str,
    name: str,
    course_intake: list[str],
    interests: list[str],
) -> Dict[str, Any]:
    url = f"{api_base_url}/users/user-profile"
    payload = {
        "user_id": user_id,
        "name": name,
        "course_intake": course_intake,
        "interests": interests or None,
    }
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


st.title("Course Learning Assistant")
st.caption("Welcome to the Course Learning Assistant.")

st.subheader("What you can do in this app")
col_chat, col_summary = st.columns(2)

with col_chat:
    st.markdown(
        """
        ### First page: Chatbot
        This page provides an AI agent that answers questions based on lecture
        materials. You can also ask technical questions outside the lecture
        scope, and the assistant is happy to help.
        """
    )

with col_summary:
    st.markdown(
        """
        ### Second page: Summaries
        Instead of only watching videos, learners can read clear explanations
        that summarize the main lecture concepts. Each report includes key
        discussions with the corresponding time they occur in the video, helping
        learners study the course more efficiently.
        """
    )

st.write(
    "Please **log in or register** to use the chatbot and browse lecture summaries."
)

with st.sidebar:
    st.subheader("Backend connection")
    api_base = st.text_input(
        "API base URL",
        value=_get_api_base_url(),
        help="Base URL of your FastAPI service, e.g. http://localhost:8055/v1",
    )
    _set_api_base_url(api_base)

    if st.session_state.get("user_profile"):
        profile = st.session_state["user_profile"]
        st.markdown(
            f"**Logged in as:** `{profile.get('user_id', '')}` – {profile.get('name', '')}"
        )
        if st.button("Log out"):
            # Clear login-related state; individual pages will handle their own UI state
            st.session_state.pop("user_profile", None)
            st.session_state.pop("messages", None)
            st.session_state.pop("session_id", None)
            st.success("Logged out.")


tab_login, tab_register = st.tabs(["Login", "Register"])

with tab_login:
    st.subheader("Login with existing user ID")
    login_user_id = st.text_input("User ID", key="login_user_id")

    if st.button("Login"):
        if not login_user_id.strip():
            st.error("Please enter a user ID.")
        else:
            try:
                profile = _get_user_profile(_get_api_base_url(), login_user_id.strip())
                if not profile:
                    st.error(
                        "User ID not found. Please check the ID or register a new account."
                    )
                else:
                    st.session_state["user_profile"] = profile
                    st.success(
                        f"Logged in as {profile.get('name', '')} (`{profile.get('user_id', '')}`)."
                    )
            except requests.RequestException as e:
                st.error(f"Login request failed: `{e}`")

with tab_register:
    st.subheader("Register / update profile")
    reg_user_id = st.text_input("User ID", key="reg_user_id")
    reg_name = st.text_input("Name", key="reg_name")
    reg_courses = st.text_input(
        "Course intake (comma-separated)",
        help="Example: CS336, CS229",
        key="reg_courses",
    )
    reg_interests = st.text_input(
        "Interests (comma-separated, optional)",
        help="Example: language models, optimization, deep learning",
        key="reg_interests",
    )

    if st.button("Register / Save profile"):
        if not reg_user_id.strip() or not reg_name.strip():
            st.error("User ID and Name are required.")
        else:
            courses = [c.strip() for c in reg_courses.split(",") if c.strip()]
            interests = [i.strip() for i in reg_interests.split(",") if i.strip()]
            try:
                profile = _create_user_profile(
                    _get_api_base_url(),
                    reg_user_id.strip(),
                    reg_name.strip(),
                    courses,
                    interests,
                )
                st.session_state["user_profile"] = profile
                st.success(
                    f"Profile saved for {profile.get('name', '')} (`{profile.get('user_id', '')}`)."
                )
            except requests.RequestException as e:
                st.error(f"Register request failed: `{e}`")

st.info(
    "Once logged in, use the **Chatbot** and **Summaries** pages from the sidebar."
)

