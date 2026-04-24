from __future__ import annotations
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from legal_intel.config import get_settings

@lru_cache(maxsize=1)
def _model(model_id: str) -> SentenceTransformer:
    return SentenceTransformer(model_id)

class EmbeddingModel:
    def __init__(self, model_id: str | None = None) -> None:
        s = get_settings()
        self.model_id = model_id or s.embedding_model
        self._st = _model(self.model_id)

    @property
    def dimension(self) -> int:
        return int(self._st.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._st.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]
