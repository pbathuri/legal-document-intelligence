from __future__ import annotations

import functools
from typing import Any

from sentence_transformers import SentenceTransformer

from legal_intel.config import get_settings
from legal_intel.rag.ollama_embed import ollama_embed_texts


@functools.lru_cache(maxsize=1)
def _st_model(model_id: str) -> SentenceTransformer:
    return SentenceTransformer(model_id)


class SentenceTransformerEmbeddingModel:
    """HuggingFace sentence-transformers (default in tests / offline CPU)."""

    def __init__(self, model_id: str | None = None) -> None:
        s = get_settings()
        self.model_id = model_id or s.embedding_model
        self._st = _st_model(self.model_id)

    @property
    def dimension(self) -> int:
        return int(self._st.get_embedding_dimension())

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._st.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]


class OllamaEmbeddingModel:
    """Ollama native embeddings (/api/embed) — aligns RAG with local LLM stack."""

    def __init__(self) -> None:
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self.encode(["__legal_intel_dimension_probe__"])
        assert self._dim is not None
        return self._dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        s = get_settings()
        vecs = ollama_embed_texts(
            texts,
            model=s.ollama_embedding_model,
            openai_base_url=s.ollama_base_url,
            timeout_seconds=max(30.0, s.ollama_probe_timeout_seconds * 15),
        )
        if self._dim is None and vecs:
            self._dim = len(vecs[0])
        return vecs


def make_embedding_model(model_id: str | None = None) -> Any:
    """Factory: sentence-transformers vs Ollama based on ``embedding_provider``."""
    s = get_settings()
    if s.embedding_provider == "ollama":
        return OllamaEmbeddingModel()
    return SentenceTransformerEmbeddingModel(model_id=model_id)


# Back-compat: ``EmbeddingModel()`` used historically for ST-only
def EmbeddingModel(model_id: str | None = None) -> Any:
    return make_embedding_model(model_id=model_id)
