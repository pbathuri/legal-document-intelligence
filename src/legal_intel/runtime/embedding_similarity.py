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


def rank_candidates_by_query_embedding(*, query: str, candidates: list[str]) -> dict[str, Any]:
    """Encode query + candidates in one batch; return candidates sorted by cosine similarity to query (desc)."""
    from legal_intel.rag.embeddings import make_embedding_model

    q = (query or "").strip()
    cleaned = [(t or "").strip() for t in candidates]
    if not q:
        raise ValueError("query must be non-empty")
    if not cleaned:
        raise ValueError("Provide at least one candidate text")
    if any(not x for x in cleaned):
        raise ValueError("Each candidate must be non-empty")
    if len(cleaned) > 64:
        raise ValueError("At most 64 candidate texts")
    m = make_embedding_model()
    all_texts = [q] + cleaned
    vecs = m.encode(all_texts)
    qv = vecs[0]
    dim = len(qv)
    ranked_raw: list[tuple[float, int]] = []
    for i, tv in enumerate(vecs[1:], start=0):
        sim = cosine_similarity_uvec(qv, tv)
        ranked_raw.append((sim, i))
    ranked_raw.sort(key=lambda x: x[0], reverse=True)
    ranked: list[dict[str, Any]] = []
    for sim, i in ranked_raw:
        prev = cleaned[i][:120] + ("…" if len(cleaned[i]) > 120 else "")
        ranked.append(
            {
                "index": i,
                "cosine_similarity": round(float(sim), 8),
                "text_preview": prev,
            }
        )
    qp = q[:400] + ("…" if len(q) > 400 else "")
    return {
        "dimension": dim,
        "query_preview": qp,
        "ranked": ranked,
    }


def centroid_similarities_for_texts(texts: list[str]) -> dict[str, Any]:
    """Mean embedding vector (element-wise) + cosine similarity of each row to that centroid."""
    from legal_intel.rag.embeddings import make_embedding_model

    cleaned = [(t or "").strip() for t in texts]
    if len(cleaned) < 2:
        raise ValueError("Provide at least two non-empty texts")
    if any(not x for x in cleaned):
        raise ValueError("Each text must be non-empty")
    if len(cleaned) > 48:
        raise ValueError("At most 48 texts for centroid analytics")
    m = make_embedding_model()
    vecs = m.encode(cleaned)
    n = len(vecs)
    dim = len(vecs[0])
    centroid = [sum(vecs[i][j] for i in range(n)) / float(n) for j in range(dim)]
    cosines = [round(cosine_similarity_uvec(vecs[i], centroid), 8) for i in range(n)]
    previews = [s[:120] + ("…" if len(s) > 120 else "") for s in cleaned]
    return {
        "count": n,
        "dimension": dim,
        "centroid": [round(float(x), 8) for x in centroid],
        "cosine_to_centroid": cosines,
        "text_previews": previews,
    }


def pairwise_cosine_matrix_for_texts(texts: list[str]) -> dict[str, Any]:
    """Encode all strings once; return full cosine similarity matrix (symmetric, diagonal 1.0)."""
    from legal_intel.rag.embeddings import make_embedding_model

    cleaned = [(t or "").strip() for t in texts]
    if len(cleaned) < 2:
        raise ValueError("Provide at least two non-empty texts")
    if any(not x for x in cleaned):
        raise ValueError("Each text must be non-empty")
    if len(cleaned) > 24:
        raise ValueError("At most 24 texts for pairwise matrix")
    m = make_embedding_model()
    vecs = m.encode(cleaned)
    n = len(vecs)
    dim = len(vecs[0])
    matrix: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            row.append(round(cosine_similarity_uvec(vecs[i], vecs[j]), 8))
        matrix.append(row)
    previews = [s[:120] + ("…" if len(s) > 120 else "") for s in cleaned]
    return {
        "count": n,
        "dimension": dim,
        "matrix": matrix,
        "text_previews": previews,
    }


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
