from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from legal_intel.config import get_settings
from legal_intel.privacy import redact_all

logger = logging.getLogger(__name__)


def _maybe_redact(text: str) -> str:
    s = get_settings()
    if s.llm_redact_pii:
        return redact_all(text)
    return text


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
            "Langfuse tracing disabled: install observability extras (pip install -e \".[observability]\"): %s",
            e,
        )
        return []


def chat_complete(
    system: str,
    user: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    """Call OpenAI-compatible server (vLLM). When LEGAL_INTEL_MOCK_LLM=1, return stub text."""
    s = get_settings()
    system = _maybe_redact(system)
    user = _maybe_redact(user)
    if s.legal_intel_mock_llm:
        return _mock_response(system, user)
    llm = ChatOpenAI(
        base_url=s.openai_api_base,
        api_key=s.openai_api_key,
        model=s.llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
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


def chat_complete_json(
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """JSON-object mode for OpenAI-compatible servers (vLLM response_format)."""
    s = get_settings()
    system = _maybe_redact(system)
    user = _maybe_redact(user)
    if s.legal_intel_mock_llm:
        return '{"doc_type":"unknown","seller_names":[],"buyer_names":[],"parcel_ids":[],"evidence":[],"mentions_dispute":false,"mentions_encumbrance":false}'
    llm = ChatOpenAI(
        base_url=s.openai_api_base,
        api_key=s.openai_api_key,
        model=s.llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
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


def _mock_response(system: str, user: str) -> str:
    return (
        "[MOCK LLM] Based only on the provided excerpts (no external citations):\n\n"
        f"- Context length: {len(user)} characters.\n"
        "- Recommend human verification of dollar amounts, dates, and party names.\n"
        f"- Specialist role (prefix): {system[:120].strip()}…\n"
    )
