"""Integration-style tests for LLM client wiring (mock server by default)."""
from unittest.mock import MagicMock, patch

import pytest

from legal_intel.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_chat_complete_json_returns_object_string_when_mock():
    import os

    os.environ["LEGAL_INTEL_MOCK_LLM"] = "1"
    get_settings.cache_clear()
    from legal_intel.llm.client import chat_complete_json

    out = chat_complete_json("sys", "user", max_tokens=100)
    assert "doc_type" in out


@patch("legal_intel.llm.client.ChatOpenAI")
def test_chat_complete_redacts_pan_before_invoke(mock_llm_cls):
    import os

    os.environ["LEGAL_INTEL_MOCK_LLM"] = "0"
    os.environ["LLM_REDACT_PII"] = "true"
    get_settings.cache_clear()
    mock_inst = MagicMock()
    mock_inst.invoke.return_value = MagicMock(content="{}")
    mock_llm_cls.return_value = mock_inst

    from legal_intel.llm.client import chat_complete

    chat_complete("s", "PAN ABCDE1234F end", temperature=0.0, max_tokens=10)
    call_args = mock_inst.invoke.call_args
    messages = call_args[0][0]
    user_content = messages[1].content
    assert "ABCDE1234F" not in user_content
    assert "REDACTED_PAN" in user_content
