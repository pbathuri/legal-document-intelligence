"""Native Ollama ``/api/ps`` — models currently loaded / running on the daemon."""

from __future__ import annotations

from typing import Any

import httpx

from legal_intel.runtime.ollama_probe import ollama_origin_from_openai_base


def fetch_ollama_running_models(
    openai_base_url: str,
    *,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    """Returns JSON from ``POST /api/ps`` (fallback ``GET /api/ps``)."""
    origin = ollama_origin_from_openai_base(openai_base_url)
    base = origin.rstrip("/")
    with httpx.Client(timeout=timeout_seconds) as client:
        try:
            r = client.post(f"{base}/api/ps", json={})
            r.raise_for_status()
            return r.json()
        except Exception:
            r = client.get(f"{base}/api/ps")
            r.raise_for_status()
            return r.json()
