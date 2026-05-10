"""Pass-through to Ollama's native ``POST /api/embed`` — returns the full JSON response."""

from __future__ import annotations

from typing import Any

import httpx

from legal_intel.runtime.ollama_probe import ollama_origin_from_openai_base


def ollama_native_embed_raw(
    openai_base_url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """Forward JSON body to ``/api/embed`` and return parsed daemon response."""
    origin = ollama_origin_from_openai_base(openai_base_url)
    url = f"{origin.rstrip('/')}/api/embed"
    with httpx.Client(timeout=timeout_seconds) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()
