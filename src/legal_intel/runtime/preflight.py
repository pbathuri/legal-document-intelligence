"""Aggregated readiness checks for ops dashboards and local CLI."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from legal_intel.config import get_settings
from legal_intel.rag.store import LegalVectorStore
from legal_intel.runtime.device_profile import gather_device_profile
from legal_intel.runtime.local_paths import parse_allow_prefixes
from legal_intel.runtime.ollama_probe import (
    fetch_ollama_model_names,
    ollama_origin_from_openai_base,
)
from legal_intel.runtime.ollama_warnings import build_ollama_model_warnings


def _qdrant_ping() -> dict[str, Any]:
    s = get_settings()
    try:
        store = LegalVectorStore()
        name = s.qdrant_collection
        exists = store._client.collection_exists(name)
        pts = None
        if exists:
            try:
                cnt = store._client.count(collection_name=name, exact=True)
                pts = getattr(cnt, "count", cnt)
            except Exception:
                pts = None
        return {"ok": True, "collection": name, "exists": exists, "points_count": pts}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _ollama_embed_probe() -> dict[str, Any]:
    s = get_settings()
    if s.embedding_provider != "ollama":
        return {
            "skipped": True,
            "ok": True,
            "reason": "embedding_provider is not ollama",
        }
    try:
        from legal_intel.rag.ollama_embed import ollama_embed_texts

        ollama_embed_texts(
            ["ping"],
            model=s.ollama_embedding_model,
            openai_base_url=s.ollama_base_url,
            timeout_seconds=max(5.0, s.ollama_probe_timeout_seconds * 5),
        )
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def ollama_embed_probe_result() -> dict[str, Any]:
    """Same probe as preflight / readiness — exposed for ``/v1/ollama/agent-stack`` diagnostics."""
    return _ollama_embed_probe()


def gather_preflight() -> dict[str, Any]:
    s = get_settings()
    checks: dict[str, Any] = {
        "mock_llm": s.legal_intel_mock_llm,
        "embedding_provider": s.embedding_provider,
        "embedding_model_st": s.embedding_model,
        "ollama_embedding_model": s.ollama_embedding_model,
        "qdrant_url": s.qdrant_url,
        "llm_provider": s.llm_provider,
        "local_ingest_allow_configured": bool((s.legal_intel_allow_local_paths or "").strip()),
        "local_ingest_prefixes": [
            str(p) for p in parse_allow_prefixes(s.legal_intel_allow_local_paths)
        ],
    }

    o_origin: str | None = None
    o_models: list[str] | None = None
    o_err: str | None = None
    if s.llm_provider == "ollama":
        try:
            o_origin = ollama_origin_from_openai_base(s.ollama_base_url)
            o_models, o_err = fetch_ollama_model_names(
                s.ollama_base_url,
                timeout_seconds=s.ollama_probe_timeout_seconds,
            )
        except Exception as e:
            o_err = str(e)

    warns = build_ollama_model_warnings(s, o_models or [])
    checks["ollama_origin"] = o_origin
    checks["ollama_tags_ok"] = o_err is None and o_models is not None
    checks["ollama_error"] = o_err
    checks["ollama_model_count"] = len(o_models or [])
    checks["warnings"] = warns

    # Disk for upload dir
    up = Path(s.upload_storage_dir)
    up.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(up)
    free_ratio = usage.free / usage.total if usage.total else 0.0
    checks["disk"] = {
        "upload_storage_dir": str(up.resolve()),
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "free_ratio": round(free_ratio, 4),
    }

    checks["qdrant"] = _qdrant_ping()
    checks["ollama_embed"] = ollama_embed_probe_result()
    checks["device"] = gather_device_profile()

    ready = bool(checks["qdrant"].get("ok"))
    oe = checks["ollama_embed"]
    if s.embedding_provider == "ollama":
        ready = ready and bool(oe.get("ok"))
    if s.llm_provider == "ollama":
        ready = ready and (o_err is None)

    return {"ready": ready, "checks": checks, "degraded": bool(warns)}
