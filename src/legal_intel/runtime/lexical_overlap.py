"""Lexical overlap between indexed documents (token Jaccard — no LLM, device-local index)."""

from __future__ import annotations

import re
from typing import Any

from legal_intel.rag.store import LegalVectorStore

_WORD = re.compile(r"\w+", re.UNICODE)

_STOPWORDS = frozenset(
    """
    the a an and or to of in for is are was were be been being
    shall will may must should would could such any all each every
    this that these those with without by as at on from into onto
    not no nor if then else whether herein hereby thereof thereto
    party parties agreement section article exhibit schedule date
    """.split()
)


def _token_set(texts: list[str]) -> set[str]:
    out: set[str] = set()
    for t in texts:
        for m in _WORD.finditer((t or "").lower()):
            w = m.group()
            if len(w) < 2 or w in _STOPWORDS:
                continue
            out.add(w)
    return out


def document_lexical_jaccard(
    *,
    doc_id_a: str,
    doc_id_b: str,
    max_chunks_per_document: int = 64,
) -> dict[str, Any]:
    """Union token sets across chunks per document; Jaccard = |∩| / |∪|."""
    da = (doc_id_a or "").strip()
    db = (doc_id_b or "").strip()
    if not da or not db:
        raise ValueError("Both doc_ids must be non-empty")
    cap = max(4, min(max_chunks_per_document, 256))
    store = LegalVectorStore()
    texts_a = store.list_chunk_texts(da, max_chunks=cap)
    texts_b = store.list_chunk_texts(db, max_chunks=cap)
    if not texts_a or not texts_b:
        raise ValueError("Both documents must have indexed chunks")
    sa = _token_set(texts_a)
    sb = _token_set(texts_b)
    inter = sa & sb
    union = sa | sb
    ju = len(union)
    jac = (len(inter) / float(ju)) if ju else 0.0
    return {
        "chunks_used_a": len(texts_a),
        "chunks_used_b": len(texts_b),
        "unique_tokens_a": len(sa),
        "unique_tokens_b": len(sb),
        "intersection_token_count": len(inter),
        "union_token_count": ju,
        "jaccard_similarity": round(jac, 8),
    }
