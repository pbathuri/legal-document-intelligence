"""Lightweight GET /api/version from the local Ollama daemon."""

from __future__ import annotations

from typing import Any

import httpx

from legal_intel.runtime.ollama_probe import ollama_origin_from_openai_base


def ollama_native_version(
    openai_base_url: str,
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    origin = ollama_origin_from_openai_base(openai_base_url)
    url = f"{origin.rstrip('/')}/api/version"
    with httpx.Client(timeout=timeout_seconds) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()
