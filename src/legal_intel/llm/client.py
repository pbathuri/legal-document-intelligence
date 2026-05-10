from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from legal_intel.config import LlmTaskKind, get_settings
from legal_intel.privacy import redact_all

logger = logging.getLogger(__name__)


def _maybe_redact(text: str) -> str:
    s = get_settings()
    if s.llm_redact_pii:
        return redact_all(text)
    return text


def _resolve_base_url() -> str:
    s = get_settings()
    if s.llm_provider == "ollama":
        return s.ollama_base_url.rstrip("/")
    return s.openai_api_base.rstrip("/")


def _resolve_api_key() -> str:
    s = get_settings()
    if s.llm_provider == "ollama":
        return s.openai_api_key if s.openai_api_key and s.openai_api_key != "EMPTY" else "ollama"
    return s.openai_api_key


def resolve_model_for_task(task: LlmTaskKind | None) -> str:
    """Pick model id for extraction vs specialist vs synthesis (multimodel routing)."""
    s = get_settings()
    if task == "extraction":
        return s.llm_model_extraction or s.llm_model
    if task == "synthesis":
        return s.llm_model_synthesis or s.llm_model
    if task == "specialist":
        return s.llm_model_specialist or s.llm_model
    return s.llm_model


def _make_llm(
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    model_kwargs: dict[str, Any] | None = None,
) -> ChatOpenAI:
    callbacks = _langfuse_callbacks()
    kw: dict[str, Any] = {
        "base_url": _resolve_base_url(),
        "api_key": _resolve_api_key(),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model_kwargs:
        kw["model_kwargs"] = model_kwargs
    llm = ChatOpenAI(**kw)
    if callbacks:
        # LangChain binds callbacks per-invoke in current code paths
        pass
    return llm


def _sync_langfuse_env_from_settings() -> None:
    """Langfuse v4 reads credentials from env; sync Pydantic settings when set."""
    import os

    s = get_settings()
    if s.langfuse_public_key:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", s.langfuse_public_key)
    if s.langfuse_secret_key:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", s.langfuse_secret_key)
    if s.langfuse_host:
        base = s.langfuse_host.rstrip("/")
        os.environ.setdefault("LANGFUSE_BASE_URL", base)


def _langfuse_callbacks() -> list[BaseCallbackHandler]:
    s = get_settings()
    if not s.langfuse_enabled:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        _sync_langfuse_env_from_settings()
        kw: dict[str, Any] = {}
        if s.langfuse_public_key:
            kw["public_key"] = s.langfuse_public_key
        return [CallbackHandler(**kw)]
    except ImportError as e:
        logger.warning(
            'Langfuse tracing disabled: install observability extras (pip install -e ".[observability]"): %s',
            e,
        )
        return []


def chat_complete(
    system: str,
    user: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    task: LlmTaskKind | None = None,
) -> str:
    """Call OpenAI-compatible server (vLLM or Ollama /v1). When LEGAL_INTEL_MOCK_LLM=1, return stub text."""
    s = get_settings()
    system = _maybe_redact(system)
    user = _maybe_redact(user)
    if s.legal_intel_mock_llm:
        return _mock_response(system, user)
    model = resolve_model_for_task(task)
    llm = _make_llm(model=model, temperature=temperature, max_tokens=max_tokens)
    callbacks = _langfuse_callbacks()
    cfg = {"callbacks": callbacks} if callbacks else {}
    resp = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)],
        config=cfg,
    )
    content = resp.content
    if isinstance(content, str):
        return content
    return "".join(str(p) for p in content)


def chat_stream(
    system: str,
    user: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    task: LlmTaskKind | None = None,
) -> Iterator[str]:
    """Token/chunk stream from vLLM or Ollama (OpenAI-compatible). Mock mode yields a single chunk."""
    s = get_settings()
    system = _maybe_redact(system)
    user = _maybe_redact(user)
    if s.legal_intel_mock_llm:
        yield _mock_response(system, user)
        return
    model = resolve_model_for_task(task)
    llm = _make_llm(model=model, temperature=temperature, max_tokens=max_tokens)
    callbacks = _langfuse_callbacks()
    cfg = {"callbacks": callbacks} if callbacks else {}
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    for chunk in llm.stream(messages, config=cfg):
        text = _flatten_chunk_content(chunk.content)
        if text:
            yield text


def _flatten_chunk_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def chat_complete_json(
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    task: LlmTaskKind | None = "extraction",
) -> str:
    """JSON-object mode for OpenAI-compatible servers (vLLM response_format). Ollama: retry without format if needed."""
    s = get_settings()
    system = _maybe_redact(system)
    user = _maybe_redact(user)
    if s.legal_intel_mock_llm:
        if "ref_index" in system or "citations" in system.lower():
            return json.dumps(
                {
                    "direct_answer": "[MOCK LLM] Grounded JSON answer placeholder.",
                    "citations": [{"ref_index": 1, "relevance": "high", "quote": "mock excerpt"}],
                    "limitations": "Mock LLM — disable LEGAL_INTEL_MOCK_LLM for real structured citations.",
                }
            )
        if "structured_extract_v1" in system:
            return json.dumps(
                {
                    "parties": ["[MOCK] Party A"],
                    "consideration": None,
                    "evidence_refs": [1],
                    "_mock": True,
                }
            )
        if "timeline_extract_v1" in system:
            return json.dumps(
                {
                    "events": [
                        {
                            "date_text": "[MOCK] January 15, 2026",
                            "event": "[MOCK] Illustrative milestone from excerpts.",
                            "confidence": "medium",
                            "evidence_refs": [1],
                        }
                    ],
                    "limitations": "Mock LLM — disable LEGAL_INTEL_MOCK_LLM for a real timeline.",
                    "_mock": True,
                }
            )
        if "risk_scan_v1" in system:
            return json.dumps(
                {
                    "risks": [
                        {
                            "title": "[MOCK] Illustrative risk",
                            "severity": "medium",
                            "summary": "[MOCK] Grounded-style summary placeholder.",
                            "evidence_refs": [1],
                            "mitigation_hint": "Human counsel review.",
                        }
                    ],
                    "limitations": "Mock LLM — disable LEGAL_INTEL_MOCK_LLM for a real risk scan.",
                    "_mock": True,
                }
            )
        if "glossary_extract_v1" in system:
            return json.dumps(
                {
                    "terms": [
                        {
                            "term": "[MOCK] Confidential Information",
                            "definition_or_scope": "[MOCK] Non-public technical or business information disclosed under the agreement.",
                            "confidence": "medium",
                            "evidence_refs": [1],
                        }
                    ],
                    "limitations": "Mock LLM — disable LEGAL_INTEL_MOCK_LLM for a real glossary.",
                    "_mock": True,
                }
            )
        if "contradictions_scan_v1" in system:
            return json.dumps(
                {
                    "tensions": [
                        {
                            "summary": "[MOCK] Illustrative cross-document tension.",
                            "severity": "informational",
                            "topic": "[MOCK] Illustrative topic",
                            "evidence_refs": [1, 2],
                        }
                    ],
                    "aligned_points": [{"summary": "[MOCK] Overlap placeholder.", "evidence_refs": [1]}],
                    "limitations": "Mock LLM — disable LEGAL_INTEL_MOCK_LLM for a real contradiction scan.",
                    "_mock": True,
                }
            )
        if "document_outline_v1" in system:
            return json.dumps(
                {
                    "sections": [
                        {
                            "heading": "[MOCK] Indemnification",
                            "summary_line": "[MOCK] Survival and scope of indemnity obligations.",
                            "confidence": "medium",
                            "evidence_refs": [1],
                        }
                    ],
                    "limitations": "Mock LLM — disable LEGAL_INTEL_MOCK_LLM for a real outline.",
                    "_mock": True,
                }
            )
        if "diligence_checklist_v1" in system:
            return json.dumps(
                {
                    "items": [
                        {
                            "category": "[MOCK] Commercial",
                            "check": "[MOCK] Confirm working capital mechanics against disclosed schedules.",
                            "priority": "P1",
                            "evidence_refs": [1],
                        }
                    ],
                    "limitations": "Mock LLM — disable LEGAL_INTEL_MOCK_LLM for a real diligence checklist.",
                    "_mock": True,
                }
            )
        if "issue_spotter_v1" in system:
            return json.dumps(
                {
                    "issues": [
                        {
                            "title": "[MOCK] Illustrative issue",
                            "severity": "informational",
                            "detail": "[MOCK] Example gap suggested by excerpts.",
                            "issue_type": "legal",
                            "evidence_refs": [1],
                        }
                    ],
                    "limitations": "Mock LLM — disable LEGAL_INTEL_MOCK_LLM for real issue spotting.",
                    "_mock": True,
                }
            )
        if "suggested_questions_v1" in system:
            return json.dumps(
                {
                    "questions": [
                        {
                            "question": "[MOCK] What approvals are required prior to assignment?",
                            "rationale": "[MOCK] Excerpts reference assignment restrictions.",
                            "evidence_refs": [1],
                        }
                    ],
                    "limitations": "Mock LLM — disable LEGAL_INTEL_MOCK_LLM for real suggested questions.",
                    "_mock": True,
                }
            )
        if "deal_thesis_v1" in system:
            return json.dumps(
                {
                    "thesis_headline": "[MOCK] Transaction exhibits standard protective covenants with identifiable carve-outs.",
                    "bull_points": [{"point": "[MOCK] Earn-out mechanics may align incentives.", "evidence_refs": [1]}],
                    "bear_points": [{"point": "[MOCK] Earn-out disputes possible if metrics contested.", "evidence_refs": [1]}],
                    "key_dependencies": ["[MOCK] Regulatory approvals if referenced in excerpts."],
                    "limitations": "Mock LLM — disable LEGAL_INTEL_MOCK_LLM for a real deal thesis.",
                    "_mock": True,
                }
            )
        if "covenant_matrix_v1" in system:
            return json.dumps(
                {
                    "rows": [
                        {
                            "obligation_summary": "[MOCK] Deliver audited financial statements annually.",
                            "party_subject": "Unknown",
                            "polarity": "affirmative",
                            "topic": "financial_reporting",
                            "trigger_or_timing": "post-Closing",
                            "carve_out_hint": None,
                            "evidence_refs": [1],
                        }
                    ],
                    "limitations": "Mock LLM — disable LEGAL_INTEL_MOCK_LLM for a real covenant matrix.",
                    "_mock": True,
                }
            )
        if "financial_terms_ledger_v1" in system:
            return json.dumps(
                {
                    "entries": [
                        {
                            "label": "[MOCK] Indemnity cap",
                            "amount_or_formula_text": "[MOCK] placeholder numeric reference from excerpts",
                            "currency_or_unit": "USD",
                            "context_note": "[MOCK] liability basket",
                            "evidence_refs": [1],
                        }
                    ],
                    "limitations": "Mock LLM — disable LEGAL_INTEL_MOCK_LLM for a real financial ledger.",
                    "_mock": True,
                }
            )
        if "remedies_playbook_v1" in system:
            return json.dumps(
                {
                    "governing_law": [{"text": "[MOCK] Illustrative governing law clause.", "evidence_refs": [1]}],
                    "forum_selection": [{"text": "[MOCK] Illustrative forum selection.", "evidence_refs": [1]}],
                    "notice_and_cure": [],
                    "specific_performance_or_injunction": [],
                    "fee_shifting_attorneys": [],
                    "limitations": "Mock LLM — disable LEGAL_INTEL_MOCK_LLM for a real remedies map.",
                    "_mock": True,
                }
            )
        return '{"doc_type":"unknown","seller_names":[],"buyer_names":[],"parcel_ids":[],"evidence":[],"mentions_dispute":false,"mentions_encumbrance":false}'
    model = resolve_model_for_task(task)
    model_kwargs = {"response_format": {"type": "json_object"}}
    llm = _make_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs=model_kwargs,
    )
    callbacks = _langfuse_callbacks()
    cfg = {"callbacks": callbacks} if callbacks else {}
    try:
        resp = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)],
            config=cfg,
        )
    except Exception as e:
        logger.warning("JSON-mode invoke failed (%s); retrying without response_format", e)
        llm2 = _make_llm(model=model, temperature=temperature, max_tokens=max_tokens)
        resp = llm2.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)],
            config=cfg,
        )
    content = resp.content
    if isinstance(content, str):
        return content
    return "".join(str(p) for p in content)


def _mock_response(system: str, user: str) -> str:
    return (
        "[MOCK LLM] Based only on the provided excerpts (no external citations):\n\n"
        f"- Context length: {len(user)} characters.\n"
        "- Recommend human verification of dollar amounts, dates, and party names.\n"
        f"- Specialist role (prefix): {system[:120].strip()}…\n"
    )
