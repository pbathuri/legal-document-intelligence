"""Best-effort optional dependency versions on the API host (stdlib importlib — no network)."""

from __future__ import annotations

import importlib
from typing import Any


def _try_version(modname: str) -> dict[str, Any]:
    try:
        m = importlib.import_module(modname)
        ver = getattr(m, "__version__", None)
        if ver is None and modname == "fitz":
            ver = getattr(m, "fitz_version", None)
        return {
            "available": True,
            "version": str(ver) if ver is not None else None,
        }
    except Exception as e:
        return {"available": False, "error": str(e)[:400]}


def gather_optional_import_versions() -> dict[str, Any]:
    """Surface ML/RAG-related wheels present in the active interpreter (useful for agent sizing)."""
    modules = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("sentence_transformers", "sentence_transformers"),
        ("langchain_core", "langchain_core"),
        ("langgraph", "langgraph"),
        ("qdrant_client", "qdrant_client"),
        ("fitz", "fitz"),
        ("sklearn", "sklearn"),
        ("psutil", "psutil"),
        ("openai", "openai"),
    ]
    resolved: dict[str, Any] = {}
    for key, mod in modules:
        resolved[key] = _try_version(mod)
    return {"modules": resolved}
