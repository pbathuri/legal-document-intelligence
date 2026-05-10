"""Ollama /api/embed for vector RAG (device-local, no sentence-transformers load)."""

from __future__ import annotations

import math
from typing import Any

import httpx

from legal_intel.runtime.ollama_probe import ollama_origin_from_openai_base


def _l2_normalize(vec: list[float]) -> list[float]:
    s = math.sqrt(sum(x * x for x in vec))
    if s == 0:
        return vec
    return [x / s for x in vec]


def ollama_embed_texts(
    texts: list[str],
    *,
    model: str,
    openai_base_url: str,
    timeout_seconds: float = 120.0,
) -> list[list[float]]:
    if not texts:
        return []
    origin = ollama_origin_from_openai_base(openai_base_url)
    url = f"{origin.rstrip('/')}/api/embed"
    payload: dict[str, Any] = {"model": model, "input": texts}
    with httpx.Client(timeout=timeout_seconds) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    embs = data.get("embeddings")
    if not isinstance(embs, list):
        raise RuntimeError(f"Ollama embed response missing embeddings: {data!r}")
    out: list[list[float]] = []
    for row in embs:
        if isinstance(row, list):
            vec = [float(x) for x in row]
            out.append(_l2_normalize(vec))
        else:
            raise RuntimeError(f"Unexpected embedding row: {row!r}")
    if len(out) != len(texts):
        raise RuntimeError(f"Ollama embed count mismatch: expected {len(texts)}, got {len(out)}")
    return out
