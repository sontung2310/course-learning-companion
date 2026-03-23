from sentence_transformers import SentenceTransformer
import chromadb
import os
from typing import List, Dict, Any, Tuple, Optional
import re

class RetrievalService:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.CloudClient(
            tenant=os.environ["CHROMA_TENANT"],
            database=os.environ["CHROMA_DATABASE"],
            api_key=os.environ["CHROMA_API_KEY"],
        )
        self.collection = self.client.get_collection("lecture_segments")
        print("Connected to ChromaDB successfully")
    
    def _tokenize(self, text: str) -> List[int]:
        tokens = re.findall(r"\w+", text.lower())
        return set(tokens)
        
    def _keyword_score(self, question: str, document: str) -> float:
        """
        Very simple lexical signal: fraction of unique query tokens appearing in the document.
        """
        q_tokens = self._tokenize(question)
        if not q_tokens:
            return 0.0
        doc_tokens = self._tokenize(document)
        hits = len(q_tokens.intersection(doc_tokens))
        return hits / len(q_tokens)

    @staticmethod
    def _infer_lecture_week_number(question: str) -> Optional[int]:
        """
        Parse a lecture/week index from the user question (e.g. "week 4", "lecture 4", "wk 4").
        Returns None if no confident match.
        """
        q = question.lower()
        patterns = [
            r"\bweek\s*(\d+)\b",
            r"\blecture\s*(\d+)\b",
            r"\bwk\.?\s*(\d+)\b",
            r"\blec\.?\s*(\d+)\b",
            r"\b(\d+)(?:st|nd|rd|th)\s+week\b",
        ]
        for pat in patterns:
            m = re.search(pat, q)
            if m:
                try:
                    n = int(m.group(1))
                    if 1 <= n <= 200:
                        return n
                except ValueError:
                    continue
        return None

    @staticmethod
    def _metadata_matches_lecture_week(meta: Optional[Dict[str, Any]], n: int) -> bool:
        """
        Aligns with ingested titles like "Lecture 4: ..." (see ingest_chromadb / *_chunks.json).
        Chroma cannot substring-filter string metadata, so we match in Python after fetch.
        """
        if not meta:
            return False
        raw = meta.get("number_lecture")
        if not isinstance(raw, str) or not raw.strip():
            return False
        s = raw.strip()
        if re.search(rf"(?i)^lecture\s+{n}\b", s):
            return True
        if re.search(rf"(?i)\bweek\s+{n}\b", s):
            return True
        return False

    def retrieve_vector(
        self,
        question: str,
        top_k: int = 5,
        initial_k: int = 25,
        alpha: float = 0.5,
        *,
        where: Optional[Dict[str, Any]] = None,
        metadata_boost: float = 0.2,
        use_lecture_metadata_boost: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid dense + lexical retrieval over Chroma.

        - ``where``: optional Chroma metadata filter (exact / $in / $and / $or). Use this when
          you know ``video_id``, exact ``number_lecture`` strings, etc.
        - When the question mentions a week/lecture number, chunks whose ``number_lecture``
          metadata matches that index get a score boost (metadata cannot be substring-filtered
          in Chroma for string fields).
        """
        q_emb = self.model.encode([question], normalize_embeddings=True).tolist()
        lecture_n = self._infer_lecture_week_number(question) if use_lecture_metadata_boost else None
        n_results = initial_k + 15 if lecture_n is not None else initial_k
        res = self.collection.query(
            query_embeddings=q_emb,
            n_results=n_results,
            include=["distances", "metadatas", "documents"],
            where=where,
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
            s_lex = self._keyword_score(question, doc)
            # Simple weighted sum (50% dense, 50% lexical)
            score = alpha * s_dense + (1 - alpha) * s_lex
            meta_match = (
                lecture_n is not None
                and self._metadata_matches_lecture_week(meta, lecture_n)
            )
            if meta_match:
                score += metadata_boost
            results.append(
                (
                    score,
                    {
                        "id": ids[i],
                        "document": doc,
                        "metadata": meta,
                        "dense_similarity": s_dense,
                        "lexical_score": s_lex,
                        "metadata_lecture_match": meta_match,
                    },
                )
            )

        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:top_k]]