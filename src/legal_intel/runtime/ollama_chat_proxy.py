"""Pass-through to Ollama's native ``POST /api/chat`` (multi-turn; non-streaming only)."""

from __future__ import annotations

from typing import Any

import httpx

from legal_intel.runtime.ollama_probe import ollama_origin_from_openai_base


def ollama_native_chat(
    openai_base_url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """JSON request/response when ``stream`` is false."""
    if payload.get("stream"):
        raise ValueError("Streaming chat is not supported via this endpoint; set stream=false")
    origin = ollama_origin_from_openai_base(openai_base_url)
    url = f"{origin.rstrip('/')}/api/chat"
    with httpx.Client(timeout=timeout_seconds) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()
