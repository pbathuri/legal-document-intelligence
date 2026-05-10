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


class ContradictionsScanRequest(CrossDocumentSummaryRequest):
    """Same multi-doc retrieval as cross-summary; defaults tuned for structured tension mining."""

    retrieval_query: str = Field(
        default="representations warranties indemnity survival disclosure carve-out fundamental breach governing law consideration",
        min_length=1,
        max_length=4000,
    )
    instruction: str = Field(
        default="Identify structured cross-document tensions, inconsistencies, or materially ambiguous alignments; cite only [n] excerpt indices.",
        min_length=1,
        max_length=12000,
    )


class ContradictionsScanResponse(BaseModel):
    doc_ids: list[str]
    contradictions: dict[str, Any]
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


class GlossaryExtractRequest(BaseModel):
    """Single-doc retrieval + JSON glossary / defined terms (``glossary_extract_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="definitions construed capitalized terms exhibit schedule appendix interpretation meaning",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class GlossaryExtractResponse(BaseModel):
    doc_id: str
    glossary: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class DocumentOutlineRequest(BaseModel):
    """Single-doc retrieval + JSON outline / section map (``document_outline_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="article section exhibit schedule appendix heading title preamble definitions indemnification termination",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class DocumentOutlineResponse(BaseModel):
    doc_id: str
    outline: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class EmbeddingCentroidRequest(BaseModel):
    """Mean embedding vector + each text's cosine similarity to that centroid (cluster coherence QA)."""

    texts: list[str] = Field(..., min_length=2, max_length=48)


class EmbeddingCentroidResponse(BaseModel):
    count: int
    dimension: int
    centroid: list[float]
    cosine_to_centroid: list[float]
    text_previews: list[str]
    embedding_provider: str
    ollama_embedding_model: str = ""
    embedding_model: str = ""


class EmbeddingNearestQueryRequest(BaseModel):
    """Rank candidate strings by cosine similarity to one query embedding (same backend as RAG)."""

    query: str = Field(..., min_length=1, max_length=64_000)
    candidates: list[str] = Field(..., min_length=1, max_length=64)


class EmbeddingNearestRankItem(BaseModel):
    index: int
    cosine_similarity: float
    text_preview: str


class EmbeddingNearestQueryResponse(BaseModel):
    dimension: int
    query_preview: str
    ranked: list[EmbeddingNearestRankItem]
    embedding_provider: str
    ollama_embedding_model: str = ""
    embedding_model: str = ""


class DiligenceChecklistRequest(BaseModel):
    """Single-doc retrieval + JSON diligence checklist (``diligence_checklist_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="representations warranties indemnity tax employment intellectual property regulatory permits litigation environmental benefits title material contracts customers suppliers",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class DiligenceChecklistResponse(BaseModel):
    doc_id: str
    checklist: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class IssueSpotterRequest(BaseModel):
    """Single-doc retrieval + JSON issue register (``issue_spotter_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="risk anomaly gap inconsistency carve-out survival fundamental breach material adverse change consent assignment indemnity cap warranty disclosure",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class IssueSpotterResponse(BaseModel):
    doc_id: str
    issue_register: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class SuggestedQuestionsRequest(BaseModel):
    """Single-doc retrieval + JSON suggested diligence questions (``suggested_questions_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="obligations representations warranties indemnity consideration termination assignment consent lien litigation permits benefits employment intellectual property",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class SuggestedQuestionsResponse(BaseModel):
    doc_id: str
    suggestions: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class DealThesisRequest(BaseModel):
    """Single-doc retrieval + JSON bull/bear deal thesis (``deal_thesis_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="consideration earn-out escrow indemnity warranty covenant carve-out MAC termination assignment regulatory consent competitive restrictions employees customers suppliers ip litigation title lien debt facilities dividend",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class DealThesisResponse(BaseModel):
    doc_id: str
    thesis: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class BibliographyExportRequest(BaseModel):
    """Single-doc retrieval + markdown bibliography / excerpt digest (specialist synthesis)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="parties definitions obligations representations warranties indemnity consideration termination schedules exhibits",
        min_length=1,
        max_length=4000,
    )
    instruction: str = Field(
        default="Format excerpts as a citation-ready bibliography for counsel review.",
        min_length=1,
        max_length=12000,
    )
    citation_style: Literal["neutral", "deal_memo", "compact"] = "neutral"
    limit: int | None = Field(None, ge=1, le=128)


class BibliographyExportResponse(BaseModel):
    doc_id: str
    bibliography_markdown: str
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class EmbeddingFarthestPairRequest(BaseModel):
    """Find the pair of texts with **lowest** cosine similarity (most divergent) among 3–40 strings."""

    texts: list[str] = Field(..., min_length=3, max_length=40)


class EmbeddingFarthestPairResponse(BaseModel):
    index_a: int
    index_b: int
    cosine_similarity: float
    dimension: int
    text_preview_a: str
    text_preview_b: str
    embedding_provider: str
    ollama_embedding_model: str = ""
    embedding_model: str = ""


class CovenantMatrixRequest(BaseModel):
    """Single-doc retrieval + JSON covenant / obligation matrix (``covenant_matrix_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="affirmative covenant negative covenant shall not material breach cure notice survival non-compete non-solicit confidentiality reporting consent assignment carve-out basket cap escrow earn-out working capital",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class CovenantMatrixResponse(BaseModel):
    doc_id: str
    covenant_matrix: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class FinancialTermsLedgerRequest(BaseModel):
    """Single-doc retrieval + JSON quantitative term ledger (``financial_terms_ledger_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="purchase price consideration escrow holdback earn-out threshold basket cap indemnity liability fees expenses multiple EBITDA working capital adjustment currency USD percent",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class FinancialTermsLedgerResponse(BaseModel):
    doc_id: str
    ledger: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class RemediesPlaybookRequest(BaseModel):
    """Single-doc retrieval + JSON remedies / forum / enforcement map (``remedies_playbook_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="governing law jurisdiction venue arbitration AAA ICC judicial forum injunctive relief specific performance cure notice breach waiver attorneys fees costs prevailing party indemnification exclusive remedy",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class RemediesPlaybookResponse(BaseModel):
    doc_id: str
    playbook: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class ConditionsPrecedentRequest(BaseModel):
    """Single-doc retrieval + JSON CP / closing-condition inventory (``conditions_precedent_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="conditions precedent closing deliverables bring-down certificate regulatory approval HSR consent waiver satisfaction filings permits title lien payoff FIRPTA environmental employee benefit tax representations schedules disclosure schedules material contracts financing commitment letter knowledge definition MAC MAE",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class ConditionsPrecedentResponse(BaseModel):
    doc_id: str
    conditions_register: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class ExecutionFormalitiesRequest(BaseModel):
    """Single-doc retrieval + JSON execution / counterparts / e-sign signals (``execution_formalities_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="counterparts facsimile electronic signature DocuSign Adobe Sign PDF delivery counterpart originals acknowledgment joinder signature authority officer secretary attest seal notary notices Section notices address",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class ExecutionFormalitiesResponse(BaseModel):
    doc_id: str
    formalities: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class RetrievalExpandPlanRequest(BaseModel):
    """Single-doc retrieval + JSON follow-up retrieval queries for agents (``retrieval_expand_plan_v1``)."""

    doc_id: str = Field(..., min_length=1)
    agent_goal: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="What the calling agent is trying to accomplish (grounds suggested_queries).",
    )
    retrieval_query: str = Field(
        default="indemnity escrow representations warranties termination assignment intellectual property employment benefits litigation consent MAC MAC carve-out earn-out purchase price working capital adjustment schedules exhibits disclosure schedules",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class RetrievalExpandPlanResponse(BaseModel):
    doc_id: str
    expand_plan: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class DocumentCentroidSimilarityRequest(BaseModel):
    """Cosine similarity between mean embeddings of two indexed documents (same backend as RAG)."""

    doc_id_a: str = Field(..., min_length=1)
    doc_id_b: str = Field(..., min_length=1)
    max_chunks_per_document: int = Field(48, ge=4, le=128)


class DocumentCentroidSimilarityResponse(BaseModel):
    doc_id_a: str
    doc_id_b: str
    chunks_used_a: int
    chunks_used_b: int
    cosine_between_centroids: float
    dimension: int
    embedding_provider: str
    ollama_embedding_model: str = ""
    embedding_model: str = ""


class SurvivalScheduleRequest(BaseModel):
    """Single-doc retrieval + JSON survival-of-obligations schedule (``survival_schedule_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="survive survival eighteen months twelve months anniversary Closing representations warranties indemnity covenant confidentiality non-compete non-solicitation tax environmental fundamental baskets disclosed schedules disclosure schedules sunset expiration statute limitations fundamental breach exclusive remedy",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class SurvivalScheduleResponse(BaseModel):
    doc_id: str
    survival_schedule: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class AssignmentCoCRequest(BaseModel):
    """Single-doc retrieval + JSON assignment / change-of-control map (``assignment_coc_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="assignment assign successor affiliate permitted transfer consent prohibit delegation novation merger consolidation sale of assets stock sale change of control MAC MAE lender financing collateral participation prohibited transfers exceptions",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class AssignmentCoCResponse(BaseModel):
    doc_id: str
    assignment_map: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class IpAssetsSweepRequest(BaseModel):
    """Single-doc retrieval + JSON IP / software sweep (``ip_assets_sweep_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="intellectual property patent trademark copyright trade secret proprietary software source code object code SaaS license sublicense domain OSS open source GPL MIT escrow schedule intellectual property schedule proprietary materials derivative works moral rights work made for hire mask work database rights publicity privacy know-how",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class IpAssetsSweepResponse(BaseModel):
    doc_id: str
    ip_register: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class DocumentChunkStatsRequest(BaseModel):
    """Device-local Qdrant chunk statistics for one indexed document (no LLM)."""

    doc_id: str = Field(..., min_length=1)
    max_chunks_scanned: int = Field(512, ge=1, le=5000)


class DocumentChunkStatsResponse(BaseModel):
    doc_id: str
    chunk_count_scanned: int
    nonempty_chunk_count: int
    empty_chunk_count: int
    total_characters_nonempty: int
    mean_chars_nonempty: float | None = None
    min_chars_nonempty: int | None = None
    max_chars_nonempty: int | None = None
    doc_label: str | None = None
    truncated_scan: bool


class PostClosingCovenantsRequest(BaseModel):
    """Single-doc retrieval + JSON post-closing / transition obligations (``post_closing_covenants_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="transition services TSA integration payroll benefits separation IT systems carve-out cooperation earn-out cooperation post-closing covenant ongoing employment retention accounting assistance financial statements bridge bring-down assistance cooperation reasonable assistance",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class PostClosingCovenantsResponse(BaseModel):
    doc_id: str
    post_closing: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class EarnOutMechanicsRequest(BaseModel):
    """Single-doc retrieval + JSON earn-out / contingent consideration mechanics (``earn_out_mechanics_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="earn-out earn out contingent consideration milestone KPI EBITDA revenue gross profit measurement period true-up dispute accountant expert GAAP working capital adjustment offset purchase price adjustment calculation statement objection certificate",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class EarnOutMechanicsResponse(BaseModel):
    doc_id: str
    earn_out: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class RepresentationsBucketsRequest(BaseModel):
    """Single-doc retrieval + JSON R&W thematic buckets (``reps_buckets_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="representations warranties fundamental knowledge material adverse disclosed schedules disclosure schedules bring-down schedules capitalization taxes litigation environmental intellectual property employees benefits title assets material contracts compliance organization authority subsidiaries",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class RepresentationsBucketsResponse(BaseModel):
    doc_id: str
    reps_buckets: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class DocumentLexicalJaccardRequest(BaseModel):
    """Token Jaccard overlap across two indexed documents (no embeddings — lexical complement to centroid similarity)."""

    doc_id_a: str = Field(..., min_length=1)
    doc_id_b: str = Field(..., min_length=1)
    max_chunks_per_document: int = Field(64, ge=4, le=256)


class DocumentLexicalJaccardResponse(BaseModel):
    doc_id_a: str
    doc_id_b: str
    chunks_used_a: int
    chunks_used_b: int
    unique_tokens_a: int
    unique_tokens_b: int
    intersection_token_count: int
    union_token_count: int
    jaccard_similarity: float


class TaxWithholdingRequest(BaseModel):
    """Single-doc retrieval + JSON tax / withholding hooks (``tax_withholding_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="withholding FIRPTA treaty gross-up certificate Section 338 754 transfer tax stamp duty VAT GST purchase price allocation stepped-up basis installment sale backup withholding foreign tax credits passthrough",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class TaxWithholdingResponse(BaseModel):
    doc_id: str
    tax_register: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class InsuranceRequirementsRequest(BaseModel):
    """Single-doc retrieval + JSON insurance covenant signals (``insurance_requirements_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="insurance representation warranty R&W buy-side D&O tail occurrence claims-made certificate additional insured umbrella cyber workers compensation general liability policy endorsement notice carrier limits deductible retroactive date",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class InsuranceRequirementsResponse(BaseModel):
    doc_id: str
    insurance_register: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class SanctionsExportComplianceRequest(BaseModel):
    """Single-doc retrieval + JSON sanctions / export / ABC hooks (``sanctions_export_compliance_v1``)."""

    doc_id: str = Field(..., min_length=1)
    retrieval_query: str = Field(
        default="OFAC sanctions denied party export control ITAR EAR anti-bribery FCPA facilitation payment gifts entertainment compliance policy ABC modern slavery human trafficking export import embargo restricted party",
        min_length=1,
        max_length=4000,
    )
    limit: int | None = Field(None, ge=1, le=128)


class SanctionsExportComplianceResponse(BaseModel):
    doc_id: str
    compliance_register: dict[str, Any]
    sources: list[dict[str, Any]]
    retrieval_top_k: int


class DocumentTokenDifferenceRequest(BaseModel):
    """Sorted token-set difference samples for two indexed documents (lexical delta — no embeddings)."""

    doc_id_a: str = Field(..., min_length=1)
    doc_id_b: str = Field(..., min_length=1)
    max_chunks_per_document: int = Field(64, ge=4, le=256)
    max_tokens_per_side: int = Field(400, ge=10, le=5000)


class DocumentTokenDifferenceResponse(BaseModel):
    doc_id_a: str
    doc_id_b: str
    chunks_used_a: int
    chunks_used_b: int
    unique_tokens_a: int
    unique_tokens_b: int
    total_only_in_a: int
    total_only_in_b: int
    tokens_only_in_a: list[str]
    tokens_only_in_b: list[str]
    truncated_only_in_a: bool
    truncated_only_in_b: bool
