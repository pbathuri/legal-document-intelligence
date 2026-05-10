"""Passthrough to Ollama native ``GET /api/tags`` (full daemon JSON)."""

from __future__ import annotations

from typing import Any

import httpx

from legal_intel.runtime.ollama_probe import ollama_origin_from_openai_base


def fetch_ollama_tags_raw(
    openai_base_url: str,
    *,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """Return the raw JSON from ``GET /api/tags`` (models with digests, sizes, etc.)."""
    origin = ollama_origin_from_openai_base(openai_base_url)
    url = f"{origin.rstrip('/')}/api/tags"
    with httpx.Client(timeout=timeout_seconds) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    return data if isinstance(data, dict) else {"models": [], "_raw": data}
