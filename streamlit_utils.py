from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class LectureSummary:
    course: str
    lecture_number: int | None
    lecture_title: str
    file_path: Path


_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)
# Trailing " – (2025)" (or similar) is optional — some generated HTML omits it.
_COURSE_RE = re.compile(
    r"^(?P<course>.*?)\s*–\s*Lecture\s*(?P<num>\d+)\s*:\s*(?P<title>.*?)(?:\s*–\s*\([^)]*\))?\s*$"
)


def _extract_html_title(html_text: str) -> str | None:
    m = _TITLE_RE.search(html_text)
    if not m:
        return None
    title = re.sub(r"\s+", " ", m.group(1)).strip()
    return title or None


def parse_summary_title(title: str, fallback_name: str) -> Tuple[str, int | None, str]:
    """
    Expected title pattern (from your generated HTML), with optional year suffix:
      '… – Lecture 3: Architectures, Hyperparameters – (2025)'
      '… – Lecture 2: Pytorch, Resource Accounting'
    """
    m = _COURSE_RE.match(title)
    if m:
        course = (m.group("course") or "").strip()
        lecture_number = int(m.group("num"))
        lecture_title = (m.group("title") or "").strip()
        return course, lecture_number, lecture_title

    # Fallback: best-effort parse from file/folder name
    course = "Summaries"
    lecture_title = title.strip() or fallback_name
    return course, None, lecture_title


def discover_lecture_summaries(output_dir: str | os.PathLike) -> List[LectureSummary]:
    base = Path(output_dir)
    if not base.exists():
        return []

    results: List[LectureSummary] = []
    for p in sorted(base.glob("*/*_lecture_summary.html")):
        try:
            html = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        title = _extract_html_title(html) or p.stem
        course, lecture_number, lecture_title = parse_summary_title(title, p.stem)
        results.append(
            LectureSummary(
                course=course,
                lecture_number=lecture_number,
                lecture_title=lecture_title,
                file_path=p,
            )
        )

    # Sort within course by lecture number (if present), else by title
    def sort_key(x: LectureSummary):
        num = x.lecture_number if x.lecture_number is not None else 10**9
        return (x.course.lower(), num, x.lecture_title.lower())

    return sorted(results, key=sort_key)


def group_by_course(items: List[LectureSummary]) -> Dict[str, List[LectureSummary]]:
    grouped: Dict[str, List[LectureSummary]] = {}
    for it in items:
        grouped.setdefault(it.course, []).append(it)
    return grouped

