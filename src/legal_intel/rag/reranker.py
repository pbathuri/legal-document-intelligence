"""Cross-encoder re-ranking for legal RAG (optional)."""
from __future__ import annotations

import logging
from functools import lru_cache

from legal_intel.config import get_settings
from legal_intel.errors import RerankerError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cross_encoder(model_id: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id)


def rerank_hits(
    query: str,
    hits: list[dict],
    *,
    top_k: int | None = None,
) -> list[dict]:
    """Re-score Qdrant hits with a cross-encoder. Preserves payload fields; updates score."""
    s = get_settings()
    if not s.rerank_enabled or not hits:
        return hits[: top_k or len(hits)]
    try:
        ce = _cross_encoder(s.rerank_model)
        pairs = [(query, str(h.get("text") or "")) for h in hits]
        scores = ce.predict(pairs, show_progress_bar=False)
        for h, sc in zip(hits, scores, strict=True):
            h["rerank_score"] = float(sc)
            h["score"] = float(sc)
        ranked = sorted(hits, key=lambda x: x.get(
            "rerank_score", 0.0), reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked
    except Exception as e:
        logger.warning("Re-ranking failed, using ANN order: %s", e)
        if s.rerank_strict:
            raise RerankerError(str(e)) from e
        return hits[: top_k or len(hits)]
