"""Single-call snapshot of local Ollama + routing — for agent orchestrators and dashboards."""

from __future__ import annotations

import time
from typing import Any

from legal_intel.config import get_settings
from legal_intel.llm.client import resolve_model_for_task
from legal_intel.runtime.ollama_probe import (
    fetch_ollama_model_names,
    ollama_origin_from_openai_base,
)
from legal_intel.runtime.ollama_ps_proxy import fetch_ollama_running_models
from legal_intel.runtime.ollama_version_proxy import ollama_native_version
from legal_intel.runtime.ollama_warnings import build_ollama_model_warnings
from legal_intel.runtime.preflight import ollama_embed_probe_result


def gather_ollama_agent_stack(
    *,
    version_timeout: float | None = None,
    ps_timeout: float | None = None,
    tags_timeout: float | None = None,
) -> dict[str, Any]:
    """
    Aggregate daemon metadata, loaded-model RAM state, tag inventory, embed ping, and task routing.

    Individual probes are wrapped so one failure does not drop the whole payload.
    """
    s = get_settings()
    vo = (
        version_timeout
        if version_timeout is not None
        else max(10.0, s.ollama_probe_timeout_seconds * 5)
    )
    po = ps_timeout if ps_timeout is not None else max(15.0, s.ollama_probe_timeout_seconds * 8)
    to = tags_timeout if tags_timeout is not None else s.ollama_probe_timeout_seconds

    t0 = time.perf_counter()

    o_origin: str | None = None
    if s.llm_provider == "ollama":
        try:
            o_origin = ollama_origin_from_openai_base(s.ollama_base_url)
        except Exception as e:
            o_origin = None
            origin_err = str(e)
        else:
            origin_err = None
    else:
        origin_err = None

    tags_models: list[str] | None = None
    tags_err: str | None = None
    if s.llm_provider == "ollama":
        try:
            tags_models, tags_err = fetch_ollama_model_names(s.ollama_base_url, timeout_seconds=to)
        except Exception as e:
            tags_err = str(e)

    warns = build_ollama_model_warnings(s, tags_models or [])

    ver: dict[str, Any] | None = None
    ver_err: str | None = None
    if s.llm_provider == "ollama":
        try:
            ver = ollama_native_version(s.ollama_base_url, timeout_seconds=vo)
        except Exception as e:
            ver_err = str(e)

    ps: dict[str, Any] | None = None
    ps_err: str | None = None
    if s.llm_provider == "ollama":
        try:
            ps = fetch_ollama_running_models(s.ollama_base_url, timeout_seconds=po)
        except Exception as e:
            ps_err = str(e)

    embed_probe = ollama_embed_probe_result()

    elapsed = time.perf_counter() - t0

    return {
        "generated_at_epoch": time.time(),
        "elapsed_seconds": round(elapsed, 4),
        "mock_llm": s.legal_intel_mock_llm,
        "llm_provider": s.llm_provider,
        "embedding_provider": s.embedding_provider,
        "ollama_base_url": s.ollama_base_url,
        "ollama_origin": o_origin,
        "ollama_origin_error": origin_err,
        "model_routing": {
            "default": s.llm_model,
            "extraction": resolve_model_for_task("extraction"),
            "specialist": resolve_model_for_task("specialist"),
            "synthesis": resolve_model_for_task("synthesis"),
        },
        "ollama_embedding_model": s.ollama_embedding_model,
        "embedding_model_sentence_transformers": s.embedding_model,
        "api_tags": {
            "ok": tags_err is None and tags_models is not None,
            "models": tags_models,
            "error": tags_err,
            "count": len(tags_models or []),
        },
        "daemon_version": ver,
        "daemon_version_error": ver_err,
        "loaded_models_ps": ps,
        "loaded_models_ps_error": ps_err,
        "embed_probe": embed_probe,
        "warnings": warns,
    }
