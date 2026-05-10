"""Extra Ollama daemon introspection (/api/version, /api/ps) for local ops dashboards."""

from __future__ import annotations

from typing import Any

import httpx

from legal_intel.runtime.ollama_probe import ollama_origin_from_openai_base


def gather_ollama_host_snapshot(
    openai_base_url: str,
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """
    Pull native Ollama REST state (not OpenAI /v1): version string + running model processes.

    Compatible with typical Ollama installs; failures are captured per sub-call.
    """
    origin = ollama_origin_from_openai_base(openai_base_url)
    base = origin.rstrip("/")
    out: dict[str, Any] = {
        "origin": origin,
        "version": None,
        "ps": None,
        "errors": [],
    }
    with httpx.Client(timeout=timeout_seconds) as client:
        try:
            r = client.get(f"{base}/api/version")
            r.raise_for_status()
            out["version"] = r.json()
        except Exception as e:
            out["errors"].append(f"GET /api/version: {e}")
        try:
            r = client.post(f"{base}/api/ps", json={})
            r.raise_for_status()
            out["ps"] = r.json()
        except Exception as e1:
            try:
                r = client.get(f"{base}/api/ps")
                r.raise_for_status()
                out["ps"] = r.json()
            except Exception as e2:
                out["errors"].append(f"POST/GET /api/ps: {e1}; {e2}")
    out["ok"] = len(out["errors"]) == 0
    return out
