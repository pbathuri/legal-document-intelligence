"""Ollama embedding helper contract."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from legal_intel.rag.ollama_embed import ollama_embed_texts


@patch("legal_intel.rag.ollama_embed.httpx.Client")
def test_ollama_embed_texts_parses_embeddings(mock_client_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "embeddings": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    }
    mock_inner = MagicMock()
    mock_inner.post.return_value = mock_resp
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_inner
    mock_cm.__exit__.return_value = None
    mock_client_cls.return_value = mock_cm

    out = ollama_embed_texts(
        ["a", "b"],
        model="nomic-embed-text",
        openai_base_url="http://127.0.0.1:11434/v1",
    )
    assert len(out) == 2
    assert len(out[0]) == 3
    assert abs(sum(x * x for x in out[0]) - 1.0) < 1e-6
