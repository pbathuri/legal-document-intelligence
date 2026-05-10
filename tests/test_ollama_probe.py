"""Ollama local probe (uses httpx; no live daemon required for unit test)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from legal_intel.runtime.ollama_probe import (
    fetch_ollama_model_names,
    ollama_origin_from_openai_base,
)


def test_ollama_origin_strips_v1_path():
    assert ollama_origin_from_openai_base("http://localhost:11434/v1") == "http://localhost:11434"


def test_fetch_models_parses_tags():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "models": [{"name": "llama3.2:latest", "size": 1}, {"name": "mistral:latest"}],
    }
    mock_resp.raise_for_status = MagicMock()
    inner = MagicMock()
    inner.get.return_value = mock_resp

    with patch("legal_intel.runtime.ollama_probe.httpx.Client") as mc:
        mc.return_value.__enter__.return_value = inner
        names, err = fetch_ollama_model_names("http://localhost:11434/v1", timeout_seconds=1.0)

    assert err is None
    assert names == ["llama3.2:latest", "mistral:latest"]
