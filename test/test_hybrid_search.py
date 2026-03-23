import os
from typing import List, Dict, Any, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

# Add path to ingest_chromadb.py
import sys
# Add path /Users/sontung/Desktop/3.Project/AIDE/Final Project/ingest_chromadb.py
sys.path.append("/Users/sontung/Desktop/3.Project/AIDE/Final Project")
from ingest_chromadb import DEFAULT_COLLECTION_NAME, DEFAULT_EMBED_MODEL  # type: ignore


def get_client():
    """
    Connect to ChromaDB.

    Preference order:
    1. CloudClient when CHROMA_TENANT / CHROMA_DATABASE / CHROMA_API_KEY are set
    2. Local HttpClient on localhost:2310 (Docker compose default)
    """
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")
    api_key = os.getenv("CHROMA_API_KEY")

    if tenant and database and api_key and hasattr(chromadb, "CloudClient"):
        print("Using Chroma Cloud Client")
        return chromadb.CloudClient(
            tenant=tenant,
            database=database,
            api_key=api_key,
        )

    # Fallback to local HTTP Chroma (see infra/docker-compose.yaml)
    return chromadb.HttpClient(host="localhost", port=2310)


def _keyword_score(query: str, document: str) -> float:
    """
    Very simple lexical signal: fraction of unique query tokens appearing in the document.
    """
    q_tokens = {t.lower() for t in query.split() if t.strip()}
    if not q_tokens:
        return 0.0
    doc_l = document.lower()
    hits = sum(1 for t in q_tokens if t in doc_l)
    return hits / len(q_tokens)


def hybrid_search(
    query: str,
    n_results: int = 5,
    initial_k: int = 25,
) -> List[Dict[str, Any]]:
    """
    Simple hybrid search:
    - Use dense vector search from ChromaDB to get top-k candidates.
    - Re-rank those candidates locally by combining vector distance and keyword overlap.
    """
    client = get_client()
    collection = client.get_collection(DEFAULT_COLLECTION_NAME)

    model = SentenceTransformer(DEFAULT_EMBED_MODEL)
    q_emb = model.encode([query], normalize_embeddings=True).tolist()

    res = collection.query(
        query_embeddings=q_emb,
        n_results=initial_k,
        include=["distances", "metadatas", "documents"],
    )

    ids: List[str] = res.get("ids", [[]])[0]
    docs: List[str] = res.get("documents", [[]])[0]
    metas: List[Dict[str, Any]] = res.get("metadatas", [[]])[0]
    dists: List[float] = res.get("distances", [[]])[0]

    if not ids:
        return []

    # Normalize distances into similarity in [0, 1] (rough heuristic)
    d_min, d_max = min(dists), max(dists)
    if d_max == d_min:
        sim_dense = [1.0 for _ in dists]
    else:
        sim_dense = [1.0 - (d - d_min) / (d_max - d_min) for d in dists]

    results: List[Tuple[float, Dict[str, Any]]] = []
    for i, (doc, meta, s_dense) in enumerate(zip(docs, metas, sim_dense)):
        s_lex = _keyword_score(query, doc)
        # Simple weighted sum (50% dense, 50% lexical)
        score = 0.5 * s_dense + 0.5 * s_lex
        results.append(
            (
                score,
                {
                    "id": ids[i],
                    "document": doc,
                    "metadata": meta,
                    "dense_similarity": s_dense,
                    "lexical_score": s_lex,
                },
            )
        )

    results.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in results[:n_results]]


def main():
    """
    Convenience CLI to manually inspect hybrid search behavior.

    Example:
      python -m test.test_hybrid_search "transformer architecture and RoPE"
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m test.test_hybrid_search \"your query here\"")
        raise SystemExit(1)

    query = " ".join(sys.argv[1:])
    print(f"Hybrid search query: {query!r}")

    results = hybrid_search(query, n_results=5)
    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, start=1):
        meta = r["metadata"]
        title = meta.get("title") or meta.get("refined_title") or ""
        seg_idx = meta.get("segment_index")
        vid = meta.get("video_id")
        print("=" * 80)
        print(f"Rank {i} | video_id={vid} segment_index={seg_idx}")
        if title:
            print(f"Title: {title}")
        print(f"Dense similarity: {r['dense_similarity']:.3f} | Lexical: {r['lexical_score']:.3f}")
        print("-" * 80)
        print(r["document"][:800])
        print()


if __name__ == "__main__":
    main()

