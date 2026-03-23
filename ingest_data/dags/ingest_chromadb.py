#!/usr/bin/env python3
"""
Ingest `*_chunks.json` into ChromaDB for RAG.

Expected input JSON schema:
- Top-level keys: `video_id`, `metadata`, `segments`
- Main RAG text: `segments[*].content_plain`
- Stored as metadata in Chroma: `video_id`, `course_name`, `number_lecture`,
  `start_timestamp`, `end_timestamp`, `segment_index` (and a few helpful extras like `title`)

Automation-friendly:
- Pass an `output/<video_id>/` folder containing exactly one `*_chunks.json`.
- Exactly one `video_id` is ingested per run.
"""

import json
import os
import re
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterable

# ChromaDB needs SQLite >= 3.35. In some Airflow images, builtin sqlite3 is older.
# Prefer pysqlite3-binary when available, then import chromadb.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

import chromadb

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


# Default collection name for lecture segments
DEFAULT_COLLECTION_NAME = "lecture_segments"
# Persist directory relative to project or cwd
DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


def sanitize_for_id(s: str, max_len: int = 200) -> str:
    """Make a string safe for use as an ID (alphanumeric, underscore, hyphen)."""
    s = re.sub(r"[^\w\-]", "_", s)
    return s[:max_len].strip("_") or "chunk"


def _split_text(text: str, max_chars: int) -> List[str]:
    """Split text into chunks by paragraph then by sentence."""
    parts: List[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            parts.append(para)
            continue
        current: List[str] = []
        current_len = 0
        for sent in re.split(r"(?<=[.!?])\s+", para):
            sent = sent.strip()
            if not sent:
                continue
            if current_len + len(sent) + 1 <= max_chars:
                current.append(sent)
                current_len += len(sent) + 1
            else:
                if current:
                    parts.append(" ".join(current))
                current = [sent]
                current_len = len(sent)
        if current:
            parts.append(" ".join(current))
    return parts


def _chunk_text_by_tokens_with_overlap(
    text: str,
    max_tokens: int = 200,
    overlap_tokens: int = 40,
) -> List[str]:
    """
    Split text into sentence-based chunks with token limits and overlap.

    - Sentences are not split across chunks.
    - Each chunk aims for <= max_tokens tokens (approximate, using whitespace tokenization).
    - Consecutive chunks overlap by at least `overlap_tokens` tokens from the end of the previous chunk.
    """
    # First split into sentences.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []

    # Approximate tokens by whitespace splitting.
    token_lengths = [len(s.split()) for s in sentences]

    chunks: List[str] = []
    n = len(sentences)
    start = 0

    while start < n:
        end = start
        token_count = 0

        # Grow the chunk by adding whole sentences until we reach the limit.
        while end < n:
            next_len = token_lengths[end]
            # Always allow at least one sentence, even if it exceeds max_tokens.
            if token_count > 0 and token_count + next_len > max_tokens:
                break
            token_count += next_len
            end += 1
            if token_count >= max_tokens:
                break

        # Safety: ensure progress.
        if end == start:
            end = start + 1

        chunks.append(" ".join(sentences[start:end]))

        if end >= n:
            break

        # Determine new start index to ensure overlap of at least `overlap_tokens` tokens.
        overlap_sum = 0
        idx = end - 1
        # Walk backwards from the end of the current chunk.
        while idx > start and overlap_sum < overlap_tokens:
            overlap_sum += token_lengths[idx]
            idx -= 1

        # `idx + 1` is the first sentence to include in the overlap window.
        new_start = idx + 1
        # Ensure we always move forward (avoid infinite loops if overlap_tokens is large).
        if new_start >= end:
            new_start = end - 1

        start = new_start

    return chunks


def _flatten_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """ChromaDB accepts only str, int, float, bool for metadata values."""
    flat: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            flat[k] = str(v)
    return flat


def _load_chunks_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON object in {path}")
    return data


def _iter_segment_docs(
    data: Dict[str, Any],
    *,
    max_doc_chars: Optional[int],
) -> Iterable[Tuple[str, Dict[str, Any], str]]:
    """
    Yield (document_text, metadata, stable_id) for each segment (and optional sub-chunks).
    """
    top_video_id = data.get("video_id", "") or ""
    top_meta = data.get("metadata") or {}
    if not isinstance(top_meta, dict):
        top_meta = {}

    segments = data.get("segments") or []
    if not isinstance(segments, list):
        raise ValueError("Expected `segments` to be a list")

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        segment_index = seg.get("segment_index")
        content_plain = seg.get("content_plain") or ""
        if not isinstance(content_plain, str) or not content_plain.strip():
            continue

        seg_video_id = seg.get("video_id") or top_video_id
        title = seg.get("title")
        refined_title = seg.get("refined_title")
        start_timestamp = seg.get("start_timestamp")
        end_timestamp = seg.get("end_timestamp")

        # Include all top-level metadata (e.g., year) and override with segment fields.
        base_meta: Dict[str, Any] = {
            **top_meta,
            "video_id": seg_video_id,
            "segment_index": segment_index,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "title": title,
            "refined_title": refined_title,
        }

        # Interpret max_doc_chars as "max tokens per chunk" for token-based sentence chunking.
        max_tokens = max_doc_chars or 200
        chunks = _chunk_text_by_tokens_with_overlap(
            content_plain,
            max_tokens=max_tokens,
            overlap_tokens=40,
        )
        if not chunks:
            continue

        multiple_chunks = len(chunks) > 1
        for sub_index, sub in enumerate(chunks):
            meta = dict(base_meta)
            if multiple_chunks:
                meta["sub_index"] = sub_index
                stable_id = sanitize_for_id(f"{seg_video_id}_{segment_index}_{sub_index}")
            else:
                stable_id = sanitize_for_id(f"{seg_video_id}_{segment_index}")
            yield sub, _flatten_metadata(meta), stable_id


def _resolve_input(
    input_path: str,
    *,
    video_id: Optional[str],
) -> Tuple[str, str]:
    """
    Resolve to (resolved_video_id, resolved_json_path).

    `input_path` may be:
    - A folder `output/<video_id>/` containing exactly one `*_chunks.json`
    """
    input_path = os.path.join(input_path, video_id)
    p = Path(input_path)
    if not p.is_dir():
        raise FileNotFoundError(
            f"Expected a folder `output/<video_id>/` containing exactly one `*_chunks.json`: {input_path}"
        )

    matches = sorted(p.glob("*_chunks.json"))
    if not matches:
        raise FileNotFoundError(f"No *_chunks.json found in {input_path}")
    if len(matches) > 1:
        raise ValueError(
            f"Expected exactly one *_chunks.json in {input_path}, got: {[m.name for m in matches]}"
        )

    json_path = str(matches[0])
    data = _load_chunks_json(json_path)

    resolved_video_id = (data.get("video_id") or video_id or p.name or "").strip()
    if not resolved_video_id:
        raise ValueError("Could not resolve video_id; pass --video-id or ensure folder name/json has it")

    return resolved_video_id, json_path


def ingest_to_chromadb(
    input_path: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    embed_model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 64,
    max_doc_chars: Optional[int] = None,
    overwrite_video: bool = False,
    video_id: Optional[str] = None,
) -> int:
    """
    Load `*_chunks.json`, embed `content_plain`, add metadata, and add to ChromaDB.
    Returns number of documents added.
    """
    if chromadb is None:
        raise RuntimeError("Install chromadb: pip install chromadb")
    if SentenceTransformer is None:
        raise RuntimeError("Install sentence-transformers: pip install sentence-transformers")
    # input_path = os.path.join(input_path, video_id)
    resolved_video_id, json_path = _resolve_input(input_path, video_id=video_id)
    data = _load_chunks_json(json_path)

    # Connect to local Chroma server (e.g. via Docker on host port 2310).
    # If you change the exposed port in docker-compose, update `port` here.
    # client = chromadb.HttpClient(host="localhost", port=2310)
    
    # Use Chroma Cloud client for remote access.
    client = chromadb.CloudClient(
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
        api_key=os.environ["CHROMA_API_KEY"],
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Lecture segment chunks with video_id and timestamps"},
    )

    if overwrite_video:
        collection.delete(where={"video_id": resolved_video_id})

    items = list(_iter_segment_docs(data, max_doc_chars=max_doc_chars))
    if not items:
        raise ValueError(f"No segment documents found in {json_path}")

    ids = [it[2] for it in items]
    documents = [it[0] for it in items]
    metadatas = [it[1] for it in items]

    model = SentenceTransformer(embed_model)
    embeddings: List[List[float]] = []
    for i in range(0, len(documents), max(1, batch_size)):
        batch_docs = documents[i : i + batch_size]
        batch_emb = model.encode(batch_docs, normalize_embeddings=True)
        embeddings.extend(batch_emb.tolist() if hasattr(batch_emb, "tolist") else [list(map(float, v)) for v in batch_emb])

    if hasattr(collection, "upsert"):
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    else:
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    return len(documents)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest `*_chunks.json` into ChromaDB (SentenceTransformer embeddings)"
    )
    parser.add_argument(
        "--input_path",
        default="../../reports",
        help="Path to `reports/<video_id>/` folder containing exactly one `*_chunks.json`",
    )
    parser.add_argument(
        "--video_id",
        default=None,
        help="Optional override for video_id (defaults to JSON `video_id` or folder name)",
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="ChromaDB collection name")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="ChromaDB persist directory")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--max-doc-chars", type=int, default=None, help="Split long `content_plain` into sub-chunks")
    parser.add_argument(
        "--overwrite-video",
        action="store_true",
        help="Delete existing docs in collection for this video_id before ingesting",
    )
    args = parser.parse_args()
    n = ingest_to_chromadb(
        args.input_path,
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        embed_model=args.embed_model,
        batch_size=args.batch_size,
        max_doc_chars=args.max_doc_chars,
        overwrite_video=args.overwrite_video,
        video_id=args.video_id,
    )
    resolved_video_id, _ = _resolve_input(args.input_path, video_id=args.video_id)
    print(f"Ingested {n} chunks for video_id={resolved_video_id} into collection '{args.collection}'")


if __name__ == "__main__":
    main()
