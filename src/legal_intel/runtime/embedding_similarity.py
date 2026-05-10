"""Cosine similarity over locally computed embedding vectors (Ollama or sentence-transformers)."""

from __future__ import annotations

import math
from typing import Any


def cosine_similarity_uvec(a: list[float], b: list[float]) -> float:
    """Cosine similarity for equal-length vectors."""
    if len(a) != len(b) or not a:
        raise ValueError("Vectors must be non-empty and equal length")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def similarity_for_text_pair(text_a: str, text_b: str) -> dict[str, Any]:
    """Encode both strings with the configured embedding backend and return cosine similarity."""
    from legal_intel.rag.embeddings import make_embedding_model

    ta = (text_a or "").strip()
    tb = (text_b or "").strip()
    if not ta or not tb:
        raise ValueError("Both texts must be non-empty")
    m = make_embedding_model()
    vecs = m.encode([ta, tb])
    if len(vecs) != 2:
        raise RuntimeError("Embedding backend returned unexpected vector count")
    sim = cosine_similarity_uvec(vecs[0], vecs[1])
    return {
        "cosine_similarity": round(float(sim), 6),
        "dimension": len(vecs[0]),
    }
