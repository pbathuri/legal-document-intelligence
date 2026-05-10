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


class QueryHydeRequest(BaseModel):
    """HyDE-style RAG: synthesize a hypothetical excerpt, then retrieve using question + excerpt (local Ollama / routed LLM)."""

    question: str = Field(..., min_length=1)
    doc_id: str | None = Field(
        default=None,
        description="Optional: scope retrieval to one indexed document id.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=128,
        description="Override configured retrieval_top_k for retrieval.",
    )
    hyde_temperature: float = Field(
        default=0.12,
        ge=0.0,
        le=1.5,
        description="Sampling temperature for the hypothetical-document step only.",
    )


class QueryHydeResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    hypothetical_document: str
    retrieval_top_k: int


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


class QueryBatchRequest(BaseModel):
    """Multiple grounded Q&A calls sharing scope (same ``doc_id`` / retrieval depth)."""

    questions: list[str] = Field(..., min_length=1, max_length=24)
    doc_id: str | None = None
    limit: int | None = Field(None, ge=1, le=128)


class QueryBatchItem(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]]


class QueryBatchResponse(BaseModel):
    items: list[QueryBatchItem]
    retrieval_top_k_per_item: int


class DocumentSummaryRequest(BaseModel):
    """Retrieve chunks for one ``doc_id`` then synthesize a grounded summary (local LLM / Ollama)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="parties obligations definitions material terms dates consideration indemnity title",
        min_length=1,
        max_length=4000,
    )
    instruction: str = Field(
        default="Produce a counsel-ready summary grounded only in the excerpts.",
        min_length=1,
        max_length=12000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class DocumentSummaryResponse(BaseModel):
    doc_id: str
    summary: str
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class OllamaEmbedProxyRequest(BaseModel):
    """Forward to Ollama native ``POST /api/embed``; returns full daemon JSON."""

    input: list[str] = Field(..., min_length=1, max_length=64)
    model: str | None = Field(default=None, description="Defaults to OLLAMA_EMBEDDING_MODEL")
    truncate: bool | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class RetrieveBatchItem(BaseModel):
    question: str
    sources: list[dict[str, Any]]
    formatted_context: str
    retrieval_top_k: int


class RetrieveBatchResponse(BaseModel):
    items: list[RetrieveBatchItem]
    retrieval_top_k_per_item: int


class CompareDocumentsRequest(BaseModel):
    """Retrieve from two ``doc_id`` values then specialist comparison (Ollama when configured)."""

    doc_id_a: str = Field(..., min_length=1)
    doc_id_b: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="obligations indemnity governing law consideration parties definitions title warranty",
        min_length=1,
        max_length=4000,
    )
    instruction: str = Field(
        default="Compare and contrast the two documents based ONLY on the excerpts.",
        min_length=1,
        max_length=12000,
    )
    limit_per_document: int | None = Field(
        default=None,
        ge=2,
        le=64,
        description="Chunks per side; default half of configured retrieval_top_k (minimum 2).",
    )


class CompareDocumentsResponse(BaseModel):
    doc_id_a: str
    doc_id_b: str
    comparison: str
    sources_a: list[dict[str, Any]]
    sources_b: list[dict[str, Any]]
    retrieval_top_k_per_side: int


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


class CrossDocumentSummaryRequest(BaseModel):
    """Retrieve chunks from multiple ``doc_id`` values then one synthesis memo."""

    doc_ids: list[str] = Field(..., min_length=2, max_length=12)
    retrieval_query: str = Field(
        default="parties obligations representations warranties indemnity consideration governing law",
        min_length=1,
        max_length=4000,
    )
    instruction: str = Field(
        default="Synthesize across documents for counsel review; cite only excerpt indices.",
        min_length=1,
        max_length=12000,
    )
    limit_per_document: int | None = Field(
        None,
        ge=1,
        le=64,
        description="Chunks per document; default splits configured retrieval depth across documents.",
    )


class CrossDocumentSummaryResponse(BaseModel):
    doc_ids: list[str]
    summary: str
    sources_by_doc_id: dict[str, list[dict[str, Any]]]
    retrieval_top_k_per_document: int


class QueryCitationsResponse(BaseModel):
    """Grounded answer with JSON citation objects + flattened markdown answer."""

    answer_markdown: str
    citations: list[dict[str, Any]]
    limitations: str | None = None
    structured: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class OllamaModelsInspectRequest(BaseModel):
    """Bounded sequential native ``/api/show`` calls for operator/agent introspection."""

    models: list[str] = Field(..., min_length=1, max_length=8)


class SqliteCheckpointRequest(BaseModel):
    """Optional WAL checkpoint aggressiveness for SQLite maintenance."""

    truncate_wal: bool = Field(
        default=False,
        description="When True, uses TRUNCATE checkpoint mode (more aggressive than PASSIVE).",
    )


class EmbedLocalTextFilesRequest(BaseModel):
    """Read UTF-8 text from absolute paths under ``LEGAL_INTEL_ALLOW_LOCAL_PATHS`` (device-local)."""

    paths: list[str] = Field(..., min_length=1, max_length=16)


class EmbedLocalTextItem(BaseModel):
    path: str
    resolved_path: str | None = None
    ok: bool
    error: str | None = None
    bytes_read: int | None = None
    vector: list[float] | None = None


class EmbedLocalTextFilesResponse(BaseModel):
    dimension: int
    count_ok: int
    embedding_provider: str
    ollama_embedding_model: str = ""
    embedding_model: str = ""
    items: list[EmbedLocalTextItem]


class StructuredExtractRequest(BaseModel):
    """Retrieval scoped to one ``doc_id`` + JSON extraction for requested category keys."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="parties dates consideration governing law termination indemnity confidentiality",
        min_length=1,
        max_length=4000,
    )
    categories: list[str] = Field(
        ...,
        min_length=1,
        max_length=24,
        description="JSON keys to populate (e.g. parties, consideration).",
    )
    limit: int | None = Field(None, ge=1, le=128)


class StructuredExtractResponse(BaseModel):
    doc_id: str
    extraction: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class OllamaGenerateBatchRequest(BaseModel):
    """Sequential native ``POST /api/generate`` calls — same model, many prompts (agent batching)."""

    model: str = Field(..., min_length=1)
    prompts: list[str] = Field(..., min_length=1, max_length=12)
    system: str | None = None
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Merged into each Ollama generate payload.",
    )


class OllamaGenerateBatchItem(BaseModel):
    index: int
    ok: bool
    detail: dict[str, Any] | None = None
    error: str | None = None


class OllamaGenerateBatchResponse(BaseModel):
    model: str
    count: int
    items: list[OllamaGenerateBatchItem]


class EmbeddingPairwiseMatrixRequest(BaseModel):
    """Full cosine similarity matrix over 2–24 texts (same embedding backend as RAG)."""

    texts: list[str] = Field(..., min_length=2, max_length=24)


class EmbeddingPairwiseMatrixResponse(BaseModel):
    count: int
    dimension: int
    matrix: list[list[float]]
    text_previews: list[str]
    embedding_provider: str
    ollama_embedding_model: str = ""
    embedding_model: str = ""


class TimelineExtractRequest(BaseModel):
    """Retrieval scoped to one ``doc_id`` + JSON timeline (dates/events + evidence refs)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="dates closing effective termination renewal milestone payment schedule notice",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class TimelineExtractResponse(BaseModel):
    doc_id: str
    timeline: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class RiskScanRequest(BaseModel):
    """Single-doc retrieval + JSON risk register (severity, evidence_refs ↔ [n])."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="indemnity liability cap warranty representations termination change of control confidentiality carve-out",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class RiskScanResponse(BaseModel):
    doc_id: str
    risk_register: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int
