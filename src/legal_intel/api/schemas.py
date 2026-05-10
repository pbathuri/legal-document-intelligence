from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1)
    domain: Literal["india_re", "mna"] = "india_re"
    doc_ids: list[str] = Field(default_factory=list)
    doc_labels: dict[str, str] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    doc_id: str | None = Field(
        default=None,
        description="Optional: scope retrieval to one indexed document id.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=128,
        description="Override configured retrieval_top_k for this request.",
    )


class LocalPathIngestRequest(BaseModel):
    """Ingest a PDF from an absolute path on the API host (allowlisted prefixes only)."""

    path: str = Field(..., min_length=1)
    use_ocr: bool = False


class DocumentPurgeBatchRequest(BaseModel):
    """Remove Qdrant vectors for multiple indexed ``doc_id`` values."""

    doc_ids: list[str] = Field(..., min_length=1, max_length=200)


class IngestResponse(BaseModel):
    doc_id: str
    doc_label: str
    chunks: int
    persisted_path: str | None = None
    page_count: int | None = None
    char_count: int | None = None
    text_empty: bool | None = None


class BatchIngestResponse(BaseModel):
    items: list[IngestResponse]
    errors: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    mock_llm: bool
    llm_provider: str
    embedding_provider: str = "sentence_transformers"
    ollama_embedding_model: str = ""
    qdrant_url: str
    diligence_domain_default: str
    models: dict[str, str]
    ollama_origin: str | None = None
    ollama_models: list[str] | None = None
    ollama_error: str | None = None
    persist_uploads: bool = False
    upload_storage_dir: str | None = None
    persist_runs: bool = False
    runs_db_path: str | None = None
    warnings: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    domain: Literal["india_re", "mna"]
    result: dict[str, Any]
    run_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


class RetrieveOnlyResponse(BaseModel):
    """RAG retrieval + formatted context block without calling an LLM."""

    sources: list[dict[str, Any]]
    formatted_context: str
    retrieval_top_k: int = Field(
        ...,
        description="Effective retrieval depth for this response (config default or request override).",
    )


class EmbeddingSimilarityRequest(BaseModel):
    """Pairwise semantic similarity via the configured embedding backend."""

    text_a: str = Field(..., min_length=1, max_length=100_000)
    text_b: str = Field(..., min_length=1, max_length=100_000)


class EmbeddingSimilarityResponse(BaseModel):
    cosine_similarity: float
    dimension: int


class EmbeddingBatchRequest(BaseModel):
    """Encode many strings with the active embedding backend (Ollama /api/embed or sentence-transformers)."""

    texts: list[str] = Field(..., min_length=1, max_length=48)


class EmbeddingBatchResponse(BaseModel):
    dimension: int
    count: int
    embedding_provider: str
    ollama_embedding_model: str = ""
    embedding_model: str = ""
    vectors: list[list[float]]


class OllamaGenerateRequest(BaseModel):
    """Forward to Ollama ``POST /api/generate`` (non-streaming). Uses ``OLLAMA_BASE_URL`` origin."""

    model: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    stream: bool = False
    system: str | None = None
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional JSON fields merged into the Ollama request (temperature, num_ctx, etc.).",
    )


class OllamaShowRequest(BaseModel):
    """Inspect an installed Ollama model via native ``POST /api/show``."""

    model: str = Field(..., min_length=1)


class ChatMessage(BaseModel):
    """Single turn in Ollama native chat format."""

    role: Literal["system", "user", "assistant"] = "user"
    content: str = Field(..., min_length=1, max_length=200_000)


class OllamaChatRequest(BaseModel):
    """Forward to Ollama ``POST /api/chat`` (non-streaming). Same origin as ``OLLAMA_BASE_URL``."""

    model: str = Field(..., min_length=1)
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=64)
    stream: bool = False
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Merged into the JSON body (temperature, num_ctx, etc.).",
    )


class NearDuplicateChunksRequest(BaseModel):
    """Find high-similarity chunk pairs within one indexed ``doc_id`` (debug / dedup QA)."""

    doc_id: str = Field(..., min_length=1)
    min_similarity: float = Field(default=0.92, ge=0.5, le=1.0)
    max_chunks: int = Field(default=48, ge=2, le=64)
    max_pairs: int = Field(default=40, ge=1, le=200)


class RunSummaryOut(BaseModel):
    id: str
    created_at: str
    domain: str
    query: str
    doc_ids: list[str]


class RuntimeOut(BaseModel):
    cwd: str
    python_version: str
    platform: str
    upload_dir: str
    runs_db: str
    ollama_base_url: str
    qdrant_url: str
    device: dict[str, Any] | None = None
