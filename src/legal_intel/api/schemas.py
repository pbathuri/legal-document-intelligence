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


class IngestResponse(BaseModel):
    doc_id: str
    doc_label: str
    chunks: int
    persisted_path: str | None = None


class HealthResponse(BaseModel):
    status: str
    mock_llm: bool
    llm_provider: str
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


class AnalyzeResponse(BaseModel):
    domain: Literal["india_re", "mna"]
    result: dict[str, Any]
    run_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


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
