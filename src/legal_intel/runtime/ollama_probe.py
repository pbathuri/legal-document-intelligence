"""Probe a local Ollama daemon for installed models (uses /api/tags, not OpenAI /v1)."""

from __future__ import annotations

from urllib.parse import urlparse

import httpx


def ollama_origin_from_openai_base(openai_base_url: str) -> str:
    """Strip /v1 suffix path from OpenAI-compatible base → REST origin for native Ollama APIs."""
    u = urlparse(openai_base_url.strip())
    if not u.scheme or not u.netloc:
        raise ValueError(f"Invalid Ollama/OpenAI base URL: {openai_base_url!r}")
    return f"{u.scheme}://{u.netloc}"


def fetch_ollama_model_names(
    openai_base_url: str,
    *,
    timeout_seconds: float = 2.0,
) -> tuple[list[str] | None, str | None]:
    """
    Returns (model_names, error_message).
    On success error_message is None; on failure model_names is None.
    """
    try:
        origin = ollama_origin_from_openai_base(openai_base_url)
        url = f"{origin.rstrip('/')}/api/tags"
        with httpx.Client(timeout=timeout_seconds) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
        models = data.get("models") or []
        names: list[str] = []
        for m in models:
            if isinstance(m, dict) and m.get("name"):
                names.append(str(m["name"]))
        names.sort()
        return names, None
    except Exception as e:
        return None, str(e)
