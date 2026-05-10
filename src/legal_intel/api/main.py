from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from legal_intel.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RunSummaryOut,
    RuntimeOut,
)
from legal_intel.config import get_settings
from legal_intel.graph.build import run_diligence_for_domain, stream_diligence_for_domain
from legal_intel.llm.client import chat_complete, chat_stream, resolve_model_for_task
from legal_intel.persistence.runs import get_run, insert_run, list_runs
from legal_intel.pipeline import doc_id_from_pdf_bytes, ingest_pdf
from legal_intel.prompts import QUERY_SYSTEM, format_context_block
from legal_intel.rag.store import LegalVectorStore
from legal_intel.runtime.ollama_probe import (
    fetch_ollama_model_names,
    ollama_origin_from_openai_base,
)
from legal_intel.runtime.uploads import persist_pdf_bytes


def _cors_origins() -> list[str]:
    raw = os.environ.get("LEGAL_INTEL_CORS_ORIGINS", "*")
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def _sse_payload(data: dict[str, Any]) -> str:
    return "data: " + json.dumps(data, default=str) + "\n\n"


app = FastAPI(
    title="Legal Document Intelligence API",
    description="Ingest PDFs, run agentic diligence (India RE / M&A), or ask grounded questions. "
    "Optimized for local Ollama agents + on-device storage.",
    version="0.4.0",
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
    o_models: list[str] | None = None
    o_err: str | None = None
    o_origin: str | None = None
    if s.llm_provider == "ollama":
        try:
            o_origin = ollama_origin_from_openai_base(s.ollama_base_url)
            o_models, o_err = fetch_ollama_model_names(
                s.ollama_base_url,
                timeout_seconds=s.ollama_probe_timeout_seconds,
            )
        except Exception as e:
            o_err = str(e)

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
        ollama_origin=o_origin,
        ollama_models=o_models,
        ollama_error=o_err,
        persist_uploads=s.persist_uploads,
        upload_storage_dir=s.upload_storage_dir,
        persist_runs=s.persist_runs,
        runs_db_path=s.runs_db_path,
    )


@app.get("/v1/runtime", response_model=RuntimeOut)
def runtime_info() -> RuntimeOut:
    s = get_settings()
    return RuntimeOut(
        cwd=os.getcwd(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        upload_dir=str(Path(s.upload_storage_dir).resolve()),
        runs_db=str(Path(s.runs_db_path).resolve()),
        ollama_base_url=s.ollama_base_url,
        qdrant_url=s.qdrant_url,
    )


@app.get("/v1/agents")
def agents_manifest() -> dict[str, Any]:
    s = get_settings()
    return {
        "llm_provider": s.llm_provider,
        "ollama_base_url": s.ollama_base_url,
        "openai_api_base": s.openai_api_base,
        "model_routing": {
            "extraction": resolve_model_for_task("extraction"),
            "specialist_subagents": resolve_model_for_task("specialist"),
            "synthesis": resolve_model_for_task("synthesis"),
        },
        "langgraph_pipelines": {
            "mna": {
                "nodes": [
                    "retrieve",
                    "obligations",
                    "risks",
                    "cross_ref",
                    "compliance",
                    "synthesize",
                ],
                "description": "Contract obligations, risks, cross-document, compliance, memo.",
            },
            "india_re": {
                "nodes": [
                    "retrieve",
                    "extract_facts",
                    "build_titlegraph",
                    "dispute_check",
                    "specialists",
                    "synthesize",
                ],
                "description": "Title graph, instrument extraction, dispute signals, memo.",
            },
        },
    }


@app.get("/v1/runs", response_model=list[RunSummaryOut])
def list_diligence_runs(limit: int = 40) -> list[RunSummaryOut]:
    s = get_settings()
    if not s.persist_runs:
        return []
    rows = list_runs(db_path=Path(s.runs_db_path), limit=min(limit, 200))
    return [
        RunSummaryOut(
            id=r.id,
            created_at=r.created_at,
            domain=r.domain,
            query=r.query,
            doc_ids=r.doc_ids,
        )
        for r in rows
    ]


@app.get("/v1/runs/{run_id}")
def get_diligence_run(run_id: str) -> dict[str, Any]:
    s = get_settings()
    if not s.persist_runs:
        raise HTTPException(status_code=404, detail="Run persistence disabled")
    row = get_run(db_path=Path(s.runs_db_path), run_id=run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return row


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

    s = get_settings()
    doc_id = doc_id_from_pdf_bytes(content, file.filename)
    persisted: str | None = None
    tmp_path: str | None = None

    try:
        if s.persist_uploads:
            path, _manifest = persist_pdf_bytes(
                content=content,
                doc_id=doc_id,
                original_filename=file.filename,
                storage_dir=s.upload_storage_dir,
            )
            persisted = str(path.resolve())
            n_doc, n_chunks = ingest_pdf(
                str(path),
                doc_label=file.filename,
                use_ocr=use_ocr,
                doc_id=doc_id,
            )
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            n_doc, n_chunks = ingest_pdf(
                tmp_path,
                doc_label=file.filename,
                use_ocr=use_ocr,
                doc_id=doc_id,
            )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(
        doc_id=n_doc,
        doc_label=file.filename,
        chunks=n_chunks,
        persisted_path=persisted,
    )


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
    rid: str | None = None
    s = get_settings()
    if s.persist_runs:
        rid = insert_run(
            db_path=Path(s.runs_db_path),
            domain=body.domain,
            query=body.query,
            doc_ids=body.doc_ids,
            result=out,
        )
    return AnalyzeResponse(domain=body.domain, result=out, run_id=rid)


@app.post("/v1/analyze/stream")
def analyze_stream(body: AnalyzeRequest):
    if body.domain == "india_re" and not body.doc_ids:
        raise HTTPException(
            status_code=400,
            detail="India property mode requires at least one indexed doc_id (upload via /v1/ingest first).",
        )
    s = get_settings()

    def event_gen():
        merged: dict[str, Any] = {}
        try:
            for step in stream_diligence_for_domain(
                body.query,
                domain=body.domain,
                doc_ids=body.doc_ids,
                doc_labels=body.doc_labels,
            ):
                for _node, patch in step.items():
                    if isinstance(patch, dict):
                        merged.update(patch)
                yield _sse_payload({"step": step})
            rid = None
            if s.persist_runs:
                rid = insert_run(
                    db_path=Path(s.runs_db_path),
                    domain=body.domain,
                    query=body.query,
                    doc_ids=body.doc_ids,
                    result=merged,
                )
            yield _sse_payload({"event": "done", "run_id": rid, "result": merged})
        except Exception as e:
            yield _sse_payload({"event": "error", "message": str(e)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


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


@app.post("/v1/query/stream")
def rag_query_stream(body: QueryRequest):
    s = get_settings()

    def event_gen():
        try:
            store = LegalVectorStore()
            hits = store.search(body.question, limit=s.retrieval_top_k, doc_id=body.doc_id)
            ctx = format_context_block(hits)
            src: list[dict[str, Any]] = []
            for h in hits:
                src.append(
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
            yield _sse_payload({"event": "sources", "sources": src})
            user = f"QUESTION:\n{body.question}\n\nCONTEXT EXCERPTS:\n{ctx}"
            for piece in chat_stream(
                QUERY_SYSTEM, user, temperature=0.05, task="specialist"
            ):
                yield _sse_payload({"event": "token", "text": piece})
            yield _sse_payload({"event": "done"})
        except Exception as e:
            yield _sse_payload({"event": "error", "message": str(e)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/v1/disk")
def disk_usage():
    """Free space on the volume holding upload storage (local ops)."""
    s = get_settings()
    path = Path(s.upload_storage_dir)
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return {
        "upload_storage_dir": str(path.resolve()),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }
