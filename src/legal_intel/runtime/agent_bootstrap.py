"""Single JSON bundle for local agents: Ollama routing, preflight, device, platform."""

from __future__ import annotations

from typing import Any

from legal_intel.config import get_settings
from legal_intel.llm.client import resolve_model_for_task
from legal_intel.runtime.device_profile import gather_device_profile
from legal_intel.runtime.platform_detail import gather_platform_detail
from legal_intel.runtime.preflight import gather_preflight


def gather_agent_bootstrap_pack(*, api_version: str) -> dict[str, Any]:
    """Aggregate on-device signals so an agent can orient before calling RAG routes."""
    s = get_settings()
    routes_hint = {
        "rag_json_extraction_examples": [
            "/v1/rag/covenant-matrix",
            "/v1/rag/financial-terms-ledger",
            "/v1/rag/remedies-playbook",
            "/v1/rag/survival-schedule",
            "/v1/rag/assignment-coc",
            "/v1/rag/ip-assets-sweep",
            "/v1/rag/post-closing-covenants",
            "/v1/rag/earn-out-mechanics",
            "/v1/rag/representations-buckets",
            "/v1/rag/conditions-precedent",
            "/v1/rag/execution-formalities",
            "/v1/rag/issue-spotter",
            "/v1/rag/deal-thesis",
        ],
        "rag_agent_planner_examples": ["/v1/rag/retrieval-expand-plan"],
        "rag_synthesis_examples": ["/v1/rag/document-summary", "/v1/rag/bibliography-export"],
        "embeddings_examples": [
            "/v1/embeddings/nearest-to-query",
            "/v1/embeddings/farthest-pair",
            "/v1/embeddings/document-centroid-similarity",
            "/v1/embeddings/document-chunk-stats",
            "/v1/embeddings/document-lexical-jaccard",
        ],
    }
    return {
        "api_version": api_version,
        "legal_intel": {
            "mock_llm": s.legal_intel_mock_llm,
            "llm_provider": s.llm_provider,
            "ollama_base_url": s.ollama_base_url,
            "openai_compatible_base": s.openai_api_base,
            "embedding_provider": s.embedding_provider,
            "embedding_model_sentence_transformers": s.embedding_model,
            "ollama_embedding_model": s.ollama_embedding_model,
            "qdrant_url": s.qdrant_url,
            "retrieval_top_k_default": s.retrieval_top_k,
        },
        "model_routing": {
            "extraction": resolve_model_for_task("extraction"),
            "synthesis": resolve_model_for_task("synthesis"),
            "specialist": resolve_model_for_task("specialist"),
            "default": resolve_model_for_task(None),
        },
        "preflight": gather_preflight(),
        "platform_detail": gather_platform_detail(),
        "device": gather_device_profile(),
        "optional_imports_probe": "GET /v1/runtime/optional-imports",
        "route_hints": routes_hint,
    }
