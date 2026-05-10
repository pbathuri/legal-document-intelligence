"""Intra-document near-duplicate detection using local embedding cosine similarity."""

from __future__ import annotations

from typing import Any

from legal_intel.rag.embeddings import make_embedding_model
from legal_intel.rag.store import LegalVectorStore
from legal_intel.runtime.embedding_similarity import cosine_similarity_uvec


def list_chunks_for_doc_bounded(
    store: LegalVectorStore,
    doc_id: str,
    *,
    max_chunks: int,
) -> list[dict[str, Any]]:
    """Scroll all payloads for ``doc_id`` until exhaustion or ``max_chunks``."""
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    while len(out) < max_chunks:
        take = min(256, max_chunks - len(out))
        batch, cursor = store.scroll_chunks_for_document(doc_id, limit=take, cursor=cursor)
        out.extend(batch)
        if not batch or not cursor:
            break
    return out[:max_chunks]


def near_duplicate_chunk_pairs(
    *,
    doc_id: str,
    min_similarity: float,
    max_chunks: int,
    max_pairs: int,
) -> dict[str, Any]:
    """
    Encode chunk texts and find unordered pairs with cosine similarity >= ``min_similarity``.

    Bounded work: at most ``max_chunks`` chunks (default 48) are loaded and encoded once.
    """
    did = (doc_id or "").strip()
    if not did:
        raise ValueError("doc_id is required")

    store = LegalVectorStore()
    raw = list_chunks_for_doc_bounded(store, did, max_chunks=max_chunks)
    if len(raw) < 2:
        return {
            "doc_id": did,
            "chunks_scanned": len(raw),
            "pairs": [],
            "note": "Need at least two chunks for pairwise comparison.",
        }

    def sort_key(pl: dict[str, Any]) -> tuple[int, str]:
        ci = pl.get("chunk_index")
        if isinstance(ci, int):
            return (ci, str(pl.get("point_id", "")))
        return (10**9, str(pl.get("point_id", "")))

    rows = sorted(raw, key=sort_key)[:max_chunks]
    texts: list[str] = []
    for pl in rows:
        t = (pl.get("text") or "").strip()
        if not t:
            t = "[empty chunk text]"
        texts.append(t)

    model = make_embedding_model()
    vecs = model.encode(texts)
    if len(vecs) != len(texts):
        raise RuntimeError("Embedding backend returned unexpected vector count")

    pairs: list[dict[str, Any]] = []
    n = len(vecs)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity_uvec(vecs[i], vecs[j])
            if sim >= min_similarity:
                pi, pj = rows[i], rows[j]
                pairs.append(
                    {
                        "i": i,
                        "j": j,
                        "chunk_index_i": pi.get("chunk_index"),
                        "chunk_index_j": pj.get("chunk_index"),
                        "cosine_similarity": round(float(sim), 6),
                        "text_preview_i": texts[i][:400],
                        "text_preview_j": texts[j][:400],
                    }
                )

    pairs.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    pairs = pairs[:max_pairs]

    return {
        "doc_id": did,
        "embedding_dimension": model.dimension,
        "chunks_scanned": n,
        "pairs": pairs,
    }
