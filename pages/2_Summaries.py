from __future__ import annotations

from pathlib import Path
import base64
import re

import streamlit as st
import streamlit.components.v1 as components

from streamlit_utils import discover_lecture_summaries, group_by_course


st.set_page_config(page_title="Summaries", page_icon="📚", layout="wide")


if not st.session_state.get("user_profile"):
    st.error("You must log in first on the **Course Learning Assistant** home page.")
    st.stop()

st.title("Summarized documents")

output_dir = Path(st.text_input("Output directory", value="reports")).resolve()

items = discover_lecture_summaries(output_dir)
if not items:
    st.info(
        "No lecture summary HTML files found. Expected: `output/*/*_lecture_summary.html`."
    )
    st.stop()

grouped = group_by_course(items)

st.subheader("Index")
for course, lectures in grouped.items():
    st.markdown(f"**{course}**")
    for lec in lectures:
        label = (
            f"Lecture {lec.lecture_number}: {lec.lecture_title}"
            if lec.lecture_number is not None
            else lec.lecture_title
        )
        cols = st.columns([6, 1, 1])
        cols[0].write(label)
        if cols[1].button("View", key=f"view:{lec.file_path}"):
            st.session_state.selected_summary_path = str(lec.file_path)
        with cols[2]:
            try:
                cols[2].download_button(
                    "Download",
                    data=lec.file_path.read_bytes(),
                    file_name=lec.file_path.name,
                    mime="text/html",
                    key=f"dl:{lec.file_path}",
                )
            except Exception:
                cols[2].write("")

st.divider()

selected = st.session_state.get("selected_summary_path")
if selected:
    p = Path(selected)
    st.subheader(f"Preview: `{p.name}`")
    try:
        html = p.read_text(encoding="utf-8", errors="ignore")

        # The original HTML uses relative image paths like src="frames/xxx.jpg",
        # which do not resolve inside Streamlit. Rewrite them to inline
        # base64 data URLs so images render correctly in the browser.
        def _replace_img(match: re.Match[str]) -> str:
            src = match.group("src")
            alt = match.group("alt") or ""
            img_path = (p.parent / src).resolve()
            if not img_path.exists():
                # Keep the original tag if image is missing
                return match.group(0)
            try:
                data = img_path.read_bytes()
                b64 = base64.b64encode(data).decode("ascii")
                # Naively assume JPEG; your pipeline uses .jpg files.
                return f'<img src="data:image/jpeg;base64,{b64}" alt="{alt}">'
            except Exception:
                return match.group(0)

        img_pattern = re.compile(
            r'<img\s+src="(?P<src>[^"]+)"\s+alt="(?P<alt>[^"]*)"\s*/?>',
            re.IGNORECASE,
        )
        html_with_images = img_pattern.sub(_replace_img, html)

        components.html(html_with_images, height=900, scrolling=True)
    except Exception as e:
        st.error(f"Failed to read HTML: {e}")
else:
    st.caption("Click **View** to preview a summary here.")

