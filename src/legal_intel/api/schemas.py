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


class HealthResponse(BaseModel):
    status: str
    mock_llm: bool
    llm_provider: str
    qdrant_url: str
    diligence_domain_default: str
    models: dict[str, str]


class AnalyzeResponse(BaseModel):
    domain: Literal["india_re", "mna"]
    result: dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
