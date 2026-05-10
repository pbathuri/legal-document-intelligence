from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from legal_intel.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from legal_intel.config import get_settings
from legal_intel.graph.build import run_diligence_for_domain
from legal_intel.llm.client import chat_complete, resolve_model_for_task
from legal_intel.pipeline import ingest_pdf
from legal_intel.prompts import QUERY_SYSTEM, format_context_block
from legal_intel.rag.store import LegalVectorStore


def _cors_origins() -> list[str]:
    raw = os.environ.get("LEGAL_INTEL_CORS_ORIGINS", "*")
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(
    title="Legal Document Intelligence API",
    description="Ingest PDFs, run agentic diligence (India RE / M&A), or ask grounded questions.",
    version="0.3.0",
)

_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False if "*" in _origins else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s = get_settings()
    return HealthResponse(
        status="ok",
        mock_llm=s.legal_intel_mock_llm,
        llm_provider=s.llm_provider,
        qdrant_url=s.qdrant_url,
        diligence_domain_default=s.diligence_domain,
        models={
            "default": s.llm_model,
            "extraction": resolve_model_for_task("extraction"),
            "specialist": resolve_model_for_task("specialist"),
            "synthesis": resolve_model_for_task("synthesis"),
        },
    )


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    use_ocr: bool = False,
) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc_id, n = ingest_pdf(tmp_path, doc_label=file.filename, use_ocr=use_ocr)
        return IngestResponse(doc_id=doc_id, doc_label=file.filename, chunks=n)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    if body.domain == "india_re" and not body.doc_ids:
        raise HTTPException(
            status_code=400,
            detail="India property mode requires at least one indexed doc_id (upload via /v1/ingest first).",
        )
    result = run_diligence_for_domain(
        body.query,
        domain=body.domain,
        doc_ids=body.doc_ids,
        doc_labels=body.doc_labels,
    )
    out: dict[str, Any] = dict(result)
    return AnalyzeResponse(domain=body.domain, result=out)


@app.post("/v1/query", response_model=QueryResponse)
def rag_query(body: QueryRequest) -> QueryResponse:
    s = get_settings()
    store = LegalVectorStore()
    hits = store.search(body.question, limit=s.retrieval_top_k, doc_id=body.doc_id)
    ctx = format_context_block(hits)
    user = f"QUESTION:\n{body.question}\n\nCONTEXT EXCERPTS:\n{ctx}"
    answer = chat_complete(QUERY_SYSTEM, user, temperature=0.05, task="specialist")

    sources: list[dict[str, Any]] = []
    for h in hits:
        sources.append(
            {
                "doc_id": h.get("doc_id"),
                "doc_label": h.get("doc_label"),
                "chunk_index": h.get("chunk_index"),
                "page_start": h.get("page_start"),
                "page_end": h.get("page_end"),
                "score": h.get("score"),
                "text_preview": (h.get("text") or "")[:1200],
            }
        )
    return QueryResponse(answer=answer, sources=sources)
