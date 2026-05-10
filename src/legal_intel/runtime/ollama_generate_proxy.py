"""Pass-through to Ollama's native ``POST /api/generate`` (non-OpenAI wire format)."""

from __future__ import annotations

from typing import Any

import httpx

from legal_intel.runtime.ollama_probe import ollama_origin_from_openai_base


def ollama_native_generate(
    openai_base_url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """JSON request/response when ``stream`` is false (default)."""
    if payload.get("stream"):
        raise ValueError("Streaming generate is not supported via this endpoint; set stream=false")
    origin = ollama_origin_from_openai_base(openai_base_url)
    url = f"{origin.rstrip('/')}/api/generate"
    with httpx.Client(timeout=timeout_seconds) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()
