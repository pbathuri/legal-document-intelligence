"""Pass-through to Ollama's native ``POST /api/show`` (model manifest / parameters)."""

from __future__ import annotations

from typing import Any

import httpx

from legal_intel.runtime.ollama_probe import ollama_origin_from_openai_base


def ollama_native_show(
    openai_base_url: str,
    model: str,
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Return Ollama model details (layers, parameters, template, etc.)."""
    name = (model or "").strip()
    if not name:
        raise ValueError("model name is required")
    origin = ollama_origin_from_openai_base(openai_base_url)
    url = f"{origin.rstrip('/')}/api/show"
    with httpx.Client(timeout=timeout_seconds) as client:
        r = client.post(url, json={"name": name})
        r.raise_for_status()
        return r.json()
