from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from starlette.requests import Request

from legal_intel.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchIngestResponse,
    CompareDocumentsRequest,
    CompareDocumentsResponse,
    DocumentPurgeBatchRequest,
    DocumentSummaryRequest,
    DocumentSummaryResponse,
    EmbeddingBatchRequest,
    EmbeddingBatchResponse,
    EmbeddingSimilarityRequest,
    EmbeddingSimilarityResponse,
    HealthResponse,
    IngestResponse,
    LocalPathIngestRequest,
    NearDuplicateChunksRequest,
    OllamaChatRequest,
    OllamaEmbedProxyRequest,
    OllamaGenerateRequest,
    OllamaShowRequest,
    QueryBatchItem,
    QueryBatchRequest,
    QueryBatchResponse,
    QueryRequest,
    QueryResponse,
    RetrieveBatchItem,
    RetrieveBatchResponse,
    RetrieveOnlyResponse,
    RunSummaryOut,
    RuntimeOut,
)
from legal_intel.config import get_settings
from legal_intel.graph.build import run_diligence_for_domain, stream_diligence_for_domain
from legal_intel.llm.client import chat_complete, chat_stream, resolve_model_for_task
from legal_intel.persistence.runs import (
    delete_run,
    export_runs_json_array,
    gather_run_statistics,
    get_run,
    insert_run,
    iter_runs_ndjson_lines,
    list_runs,
    optimize_sqlite_file,
    search_runs,
    sqlite_integrity_check,
    vacuum_sqlite_file,
)
from legal_intel.pipeline import doc_id_from_pdf_bytes, ingest_pdf_with_stats
from legal_intel.prompts import (
    COMPARE_DOCUMENTS_SYSTEM,
    QUERY_SYSTEM,
    SUMMARIZE_SYSTEM,
    format_context_block,
)
from legal_intel.rag.store import LegalVectorStore
from legal_intel.runtime.api_metrics import bucket_path as metrics_bucket_path
from legal_intel.runtime.api_metrics import incr_request as metrics_incr_request
from legal_intel.runtime.audit_log import append_audit_event
from legal_intel.runtime.build_info import gather_build_info
from legal_intel.runtime.device_profile import gather_device_profile
from legal_intel.runtime.chunk_near_duplicate import near_duplicate_chunk_pairs
from legal_intel.runtime.embedding_similarity import similarity_for_text_pair
from legal_intel.runtime.ollama_embed_raw_proxy import ollama_native_embed_raw
from legal_intel.runtime.process_info import gather_api_process_snapshot
from legal_intel.runtime.ollama_chat_proxy import ollama_native_chat
from legal_intel.runtime.ollama_generate_proxy import ollama_native_generate
from legal_intel.runtime.ollama_show_proxy import ollama_native_show
from legal_intel.runtime.ollama_ps_proxy import fetch_ollama_running_models
from legal_intel.runtime.ollama_version_proxy import ollama_native_version
from legal_intel.runtime.system_snapshot import gather_system_snapshot
from legal_intel.runtime.uploads_scan import list_upload_storage_files
from legal_intel.runtime.local_paths import is_path_under_allowlist, parse_allow_prefixes
from legal_intel.runtime.ollama_probe import (
    fetch_ollama_model_names,
    ollama_origin_from_openai_base,
)
from legal_intel.runtime.ollama_warnings import build_ollama_model_warnings
from legal_intel.runtime.ollama_deep import gather_ollama_host_snapshot
from legal_intel.runtime.ollama_agent_stack import gather_ollama_agent_stack
from legal_intel.runtime.preflight import gather_preflight
from legal_intel.runtime.storage_inventory import gather_storage_inventory
from legal_intel.runtime.uploads import persist_pdf_bytes
from legal_intel.runtime.uploads_manifest import tail_upload_manifest


def _cors_origins() -> list[str]:
    raw = os.environ.get("LEGAL_INTEL_CORS_ORIGINS", "*")
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def _sse_payload(data: dict[str, Any]) -> str:
    return "data: " + json.dumps(data, default=str) + "\n\n"


def _public_settings_dict() -> dict[str, Any]:
    s = get_settings()
    d = s.model_dump(mode="json")
    for key in (
        "openai_api_key",
        "langfuse_secret_key",
        "langfuse_public_key",
        "indian_kanoon_api_token",
        "legal_intel_audit_jsonl",
    ):
        if d.get(key):
            d[key] = "***"
    return d


def _audit_prefixes() -> tuple[str, ...]:
    return (
        "/v1/ingest",
        "/v1/analyze",
        "/v1/query",
        "/v1/documents",
        "/v1/runs",
        "/v1/embeddings",
        "/v1/rag",
        "/v1/llm",
        "/v1/ollama",
        "/v1/maintenance",
    )


def _effective_retrieval_limit(body: QueryRequest) -> int:
    s = get_settings()
    if body.limit is not None:
        return body.limit
    return s.retrieval_top_k


def _batch_retrieval_limit(limit: int | None) -> int:
    s = get_settings()
    return limit if limit is not None else s.retrieval_top_k


def _rag_sources_from_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    return sources


def _prepare_document_summary_parts(
    body: DocumentSummaryRequest,
) -> tuple[int, str, str, list[dict[str, Any]]]:
    store = LegalVectorStore()
    s = get_settings()
    lim = body.limit if body.limit is not None else s.retrieval_top_k
    did = body.doc_id.strip()
    rq = body.retrieval_query.strip()
    hits = store.search(rq, limit=lim, doc_id=did)
    ctx = format_context_block(hits)
    user = f"INSTRUCTION:\n{body.instruction.strip()}\n\nCONTEXT EXCERPTS:\n{ctx}"
    return lim, did, user, _rag_sources_from_hits(hits)


def _prepare_compare_documents_parts(
    body: CompareDocumentsRequest,
) -> tuple[str, str, str, list[dict[str, Any]], list[dict[str, Any]], int]:
    store = LegalVectorStore()
    s = get_settings()
    da = body.doc_id_a.strip()
    db = body.doc_id_b.strip()
    if da == db:
        raise HTTPException(status_code=400, detail="doc_id_a and doc_id_b must differ")
    rq = body.retrieval_query.strip()
    if body.limit_per_document is not None:
        per = min(max(body.limit_per_document, 2), 64)
    else:
        per = max(2, min(32, s.retrieval_top_k // 2))
    hits_a = store.search(rq, limit=per, doc_id=da)
    hits_b = store.search(rq, limit=per, doc_id=db)
    ctx_a = format_context_block(hits_a)
    ctx_b = format_context_block(hits_b)
    user = (
        f"INSTRUCTION:\n{body.instruction.strip()}\n\n"
        f"DOCUMENT A (doc_id={da}):\n{ctx_a}\n\n"
        f"DOCUMENT B (doc_id={db}):\n{ctx_b}"
    )
    return da, db, user, _rag_sources_from_hits(hits_a), _rag_sources_from_hits(hits_b), per


def _should_audit_path(method: str, path: str) -> bool:
    if method not in ("POST", "PUT", "PATCH", "DELETE"):
        return False
    return any(path == p or path.startswith(p + "/") for p in _audit_prefixes())


def _ingest_pdf_core_bytes(content: bytes, filename: str, use_ocr: bool) -> IngestResponse:
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    s = get_settings()
    stable_doc_id = doc_id_from_pdf_bytes(content, filename)
    persisted: str | None = None
    tmp_path: str | None = None

    try:
        if s.persist_uploads:
            path, _manifest = persist_pdf_bytes(
                content=content,
                doc_id=stable_doc_id,
                original_filename=filename,
                storage_dir=s.upload_storage_dir,
            )
            persisted = str(path.resolve())
            doc_id_out, n_chunks, stats = ingest_pdf_with_stats(
                str(path),
                doc_label=filename,
                use_ocr=use_ocr,
                doc_id=stable_doc_id,
            )
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            doc_id_out, n_chunks, stats = ingest_pdf_with_stats(
                tmp_path,
                doc_label=filename,
                use_ocr=use_ocr,
                doc_id=stable_doc_id,
            )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(
        doc_id=doc_id_out,
        doc_label=filename,
        chunks=n_chunks,
        persisted_path=persisted,
        page_count=stats["page_count"],
        char_count=stats["char_count"],
        text_empty=stats["text_empty"],
    )


app = FastAPI(
    title="Legal Document Intelligence API",
    description="Ingest PDFs, run agentic diligence (India RE / M&A), or ask grounded questions. "
    "Optimized for local Ollama agents + on-device storage.",
    version="0.14.0",
)

_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False if "*" in _origins else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter() - start:.4f}"
    return response


@app.middleware("http")
async def api_metrics_middleware(request: Request, call_next):
    metrics_incr_request(metrics_bucket_path(request.url.path))
    return await call_next(request)


@app.middleware("http")
async def audit_jsonl_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    s = get_settings()
    ap = (s.legal_intel_audit_jsonl or "").strip()
    path = request.url.path
    if ap and _should_audit_path(request.method, path):
        try:
            append_audit_event(
                ap,
                {
                    "ts_epoch": time.time(),
                    "method": request.method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 3),
                    "request_id": getattr(request.state, "request_id", None),
                    "client": request.client.host if request.client else None,
                },
            )
        except OSError:
            pass
    return response


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


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

    warns = build_ollama_model_warnings(s, o_models or [])

    return HealthResponse(
        status="ok",
        mock_llm=s.legal_intel_mock_llm,
        llm_provider=s.llm_provider,
        embedding_provider=s.embedding_provider,
        ollama_embedding_model=s.ollama_embedding_model,
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
        warnings=warns,
    )


@app.get("/health/live")
def health_live() -> dict[str, str]:
    """Liveness: process is accepting connections (no dependency checks)."""
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready() -> dict[str, Any]:
    """Readiness: aggregated gates from preflight (Qdrant, Ollama when configured, embed probe)."""
    pf = gather_preflight()
    return {"ready": pf["ready"], "degraded": pf.get("degraded", False)}


@app.get("/v1/preflight")
def preflight(deep: bool = False) -> dict[str, Any]:
    """Single payload for dashboards: Qdrant, Ollama tags/embed, disk, device. ``deep=1`` adds Ollama /version+/ps."""
    out = gather_preflight()
    if deep:
        s = get_settings()
        out["ollama_host"] = gather_ollama_host_snapshot(
            s.ollama_base_url,
            timeout_seconds=max(5.0, s.ollama_probe_timeout_seconds * 3),
        )
    return out


@app.get("/v1/metrics")
def metrics_json() -> dict[str, Any]:
    """In-process HTTP request counters (resets on process restart)."""
    from legal_intel.runtime.api_metrics import snapshot

    return snapshot()


@app.get("/v1/metrics/prometheus", response_class=PlainTextResponse)
def metrics_prometheus() -> PlainTextResponse:
    """Prometheus text exposition for ``legal_intel_http_requests_*`` counters."""
    from legal_intel.runtime.api_metrics import prometheus_text

    return PlainTextResponse(
        prometheus_text(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@app.get("/v1/ollama/host")
def ollama_host_probe() -> dict[str, Any]:
    """Native Ollama daemon state: ``/api/version`` + ``/api/ps`` (running models on device)."""
    s = get_settings()
    return gather_ollama_host_snapshot(
        s.ollama_base_url,
        timeout_seconds=max(5.0, s.ollama_probe_timeout_seconds * 3),
    )


@app.get("/v1/embeddings/info")
def embeddings_backend_info() -> dict[str, Any]:
    """Resolved embedding provider/model names + dimension + one probe encode timing (device-local)."""
    s = get_settings()
    from legal_intel.rag.embeddings import make_embedding_model

    m = make_embedding_model()
    t0 = time.perf_counter()
    _ = m.encode(["legal_intel.embedding.info.probe."])
    dt = time.perf_counter() - t0
    return {
        "embedding_provider": s.embedding_provider,
        "embedding_model": s.embedding_model,
        "ollama_embedding_model": s.ollama_embedding_model,
        "dimension": m.dimension,
        "probe_encode_seconds": round(dt, 4),
    }


@app.post("/v1/embeddings/warmup")
def embeddings_warmup() -> dict[str, Any]:
    """Force-load the configured embedding backend (ST download or Ollama /api/embed)."""
    from legal_intel.rag.embeddings import make_embedding_model

    m = make_embedding_model()
    t0 = time.perf_counter()
    _ = m.encode(["legal_intel embedding warmup probe."])
    dt = time.perf_counter() - t0
    return {"ok": True, "dimension": m.dimension, "elapsed_seconds": round(dt, 4)}


@app.post("/v1/embeddings/embed-texts", response_model=EmbeddingBatchResponse)
def embedding_embed_texts(body: EmbeddingBatchRequest) -> EmbeddingBatchResponse:
    """Batch encode via the configured embedding backend (device-local Ollama or sentence-transformers)."""
    max_chars = 64_000
    cleaned: list[str] = []
    for i, t in enumerate(body.texts):
        ts = (t or "").strip()
        if not ts:
            raise HTTPException(status_code=400, detail=f"texts[{i}] is empty")
        if len(ts) > max_chars:
            raise HTTPException(
                status_code=400,
                detail=f"texts[{i}] exceeds {max_chars} characters",
            )
        cleaned.append(ts)
    from legal_intel.rag.embeddings import make_embedding_model

    s = get_settings()
    m = make_embedding_model()
    raw = m.encode(cleaned)
    vectors = [list(map(float, row)) for row in raw]
    return EmbeddingBatchResponse(
        dimension=m.dimension,
        count=len(vectors),
        embedding_provider=s.embedding_provider,
        ollama_embedding_model=s.ollama_embedding_model,
        embedding_model=s.embedding_model,
        vectors=vectors,
    )


@app.post("/v1/embeddings/similarity", response_model=EmbeddingSimilarityResponse)
def embedding_cosine_similarity(body: EmbeddingSimilarityRequest) -> EmbeddingSimilarityResponse:
    """Cosine similarity between two texts using the active embedding backend (local)."""
    try:
        out = similarity_for_text_pair(body.text_a, body.text_b)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return EmbeddingSimilarityResponse(
        cosine_similarity=float(out["cosine_similarity"]),
        dimension=int(out["dimension"]),
    )


@app.get("/v1/build")
def build_metadata() -> dict[str, Any]:
    """Package / Python / optional git SHA (set ``LEGAL_INTEL_GIT_SHA`` in deploy)."""
    return gather_build_info(api_version=app.version)


@app.post("/v1/llm/probe")
def llm_route_probe() -> dict[str, Any]:
    """Single completion via configured stack (Ollama OpenAI /v1 when not mock)."""
    s = get_settings()
    if s.legal_intel_mock_llm:
        return {
            "skipped": True,
            "reason": "LEGAL_INTEL_MOCK_LLM",
            "note": "Disable mock to hit Ollama/vLLM.",
        }
    try:
        t0 = time.perf_counter()
        text = chat_complete(
            "Reply with exactly one word: pong",
            "ping",
            temperature=0.0,
            max_tokens=8,
            task="specialist",
        )
        dt = time.perf_counter() - t0
        return {
            "ok": True,
            "model": resolve_model_for_task("specialist"),
            "reply_preview": (text or "")[:300],
            "elapsed_seconds": round(dt, 4),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/v1/ollama/version")
def ollama_daemon_version() -> dict[str, Any]:
    """Native ``GET /api/version`` from the Ollama process (daemon build metadata)."""
    s = get_settings()
    try:
        return ollama_native_version(
            s.ollama_base_url,
            timeout_seconds=max(10.0, s.ollama_probe_timeout_seconds * 5),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/v1/ollama/ps")
def ollama_running_processes() -> dict[str, Any]:
    """Native ``/api/ps`` — models loaded in memory on the local Ollama daemon."""
    s = get_settings()
    try:
        return fetch_ollama_running_models(
            s.ollama_base_url,
            timeout_seconds=max(15.0, s.ollama_probe_timeout_seconds * 8),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/v1/ollama/agent-stack")
def ollama_agent_stack() -> dict[str, Any]:
    """
    One JSON blob for local agent orchestration: Ollama ``/api/version``, ``/api/tags``, ``/api/ps``,
    embed ping, per-task model routing, and configuration warnings (never fails the whole response).
    """
    return gather_ollama_agent_stack()


@app.post("/v1/ollama/generate")
def ollama_generate_native(body: OllamaGenerateRequest) -> dict[str, Any]:
    """
    Native Ollama ``POST /api/generate`` (non-streaming). Sends **prompt**/**model** to the daemon
    derived from ``OLLAMA_BASE_URL`` — complements OpenAI-compatible ``/v1/chat/completions``.
    """
    if body.stream:
        raise HTTPException(
            status_code=400, detail="Set stream=false; streaming not supported here."
        )
    s = get_settings()
    payload: dict[str, Any] = {
        "model": body.model,
        "prompt": body.prompt,
        "stream": False,
    }
    if body.system:
        payload["system"] = body.system
    payload.update(body.options)
    try:
        return ollama_native_generate(
            s.ollama_base_url,
            payload,
            timeout_seconds=max(60.0, s.ollama_probe_timeout_seconds * 30),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/v1/ollama/chat")
def ollama_chat_native(body: OllamaChatRequest) -> dict[str, Any]:
    """Native Ollama ``POST /api/chat`` — multi-turn messages (non-streaming)."""
    if body.stream:
        raise HTTPException(
            status_code=400, detail="Set stream=false; streaming not supported here."
        )
    s = get_settings()
    payload: dict[str, Any] = {
        "model": body.model.strip(),
        "messages": [m.model_dump() for m in body.messages],
        "stream": False,
    }
    payload.update(body.options)
    try:
        return ollama_native_chat(
            s.ollama_base_url,
            payload,
            timeout_seconds=max(120.0, s.ollama_probe_timeout_seconds * 60),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/v1/ollama/show")
def ollama_show_model(body: OllamaShowRequest) -> dict[str, Any]:
    """Native Ollama ``POST /api/show`` — model parameters, template, license (local daemon)."""
    s = get_settings()
    try:
        return ollama_native_show(
            s.ollama_base_url,
            body.model.strip(),
            timeout_seconds=max(60.0, s.ollama_probe_timeout_seconds * 30),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/v1/ollama/embed-proxy")
def ollama_embed_proxy(body: OllamaEmbedProxyRequest) -> dict[str, Any]:
    """Raw passthrough to Ollama ``POST /api/embed`` — full JSON response (daemon debugging)."""
    max_chars = 64_000
    cleaned: list[str] = []
    for i, t in enumerate(body.input):
        ts = (t or "").strip()
        if not ts:
            raise HTTPException(status_code=400, detail=f"input[{i}] is empty")
        if len(ts) > max_chars:
            raise HTTPException(
                status_code=400,
                detail=f"input[{i}] exceeds {max_chars} characters",
            )
        cleaned.append(ts)
    s = get_settings()
    model = (body.model or "").strip() or s.ollama_embedding_model
    payload: dict[str, Any] = {"model": model, "input": cleaned}
    if body.truncate is not None:
        payload["truncate"] = body.truncate
    payload.update(body.options)
    try:
        return ollama_native_embed_raw(
            s.ollama_base_url,
            payload,
            timeout_seconds=max(120.0, s.ollama_probe_timeout_seconds * 60),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/v1/system/snapshot")
def system_snapshot(top_n: int = 8) -> dict[str, Any]:
    """Load averages (Unix) + optional psutil top RSS processes."""
    return gather_system_snapshot(top_n=min(max(top_n, 1), 32))


@app.get("/v1/system/process")
def api_process_metrics() -> dict[str, Any]:
    """RSS / threads for this API worker process (optional psutil)."""
    return gather_api_process_snapshot()


@app.post("/v1/maintenance/optimize-sqlite")
def maintenance_optimize_runs_db() -> dict[str, Any]:
    """SQLite ``PRAGMA optimize`` on the runs database (requires ``persist_runs``)."""
    s = get_settings()
    if not s.persist_runs:
        raise HTTPException(status_code=400, detail="Run persistence disabled")
    try:
        return optimize_sqlite_file(Path(s.runs_db_path))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/maintenance/integrity-sqlite")
def maintenance_integrity_runs_db() -> dict[str, Any]:
    """SQLite ``PRAGMA integrity_check`` on the runs database (requires ``persist_runs``)."""
    s = get_settings()
    if not s.persist_runs:
        raise HTTPException(status_code=400, detail="Run persistence disabled")
    try:
        return sqlite_integrity_check(Path(s.runs_db_path))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/maintenance/vacuum-sqlite")
def maintenance_vacuum_runs_db() -> dict[str, Any]:
    """SQLite VACUUM on the diligence runs database (requires persist_runs)."""
    s = get_settings()
    if not s.persist_runs:
        raise HTTPException(status_code=400, detail="Run persistence disabled")
    try:
        return vacuum_sqlite_file(Path(s.runs_db_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        device=gather_device_profile(),
    )


@app.get("/v1/runtime/storage")
def runtime_storage_detail() -> dict[str, Any]:
    """Bounded walk of upload storage + manifest / runs DB sizes (device-local ops)."""
    s = get_settings()
    return gather_storage_inventory(
        upload_storage_dir=s.upload_storage_dir,
        runs_db_path=s.runs_db_path,
        persist_uploads=s.persist_uploads,
    )


@app.get("/v1/settings/effective")
def effective_settings() -> dict[str, Any]:
    """Resolved configuration with secrets redacted (safe for screenshots)."""
    return _public_settings_dict()


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


@app.get("/v1/runs/stats")
def diligence_runs_statistics() -> dict[str, Any]:
    """SQLite aggregates for the configured runs database path (works even when persistence is off)."""
    s = get_settings()
    return gather_run_statistics(db_path=Path(s.runs_db_path))


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


@app.get("/v1/runs/export/json")
def export_runs_json_blob(limit: int = 10_000) -> list[dict[str, Any]]:
    """Full diligence runs as a JSON array (bounded; prefer NDJSON for huge archives)."""
    s = get_settings()
    if not s.persist_runs:
        raise HTTPException(status_code=400, detail="Run persistence disabled")
    return export_runs_json_array(
        db_path=Path(s.runs_db_path),
        limit=min(limit, 50_000),
    )


@app.get("/v1/runs/export")
def export_runs_ndjson(limit: int = 50_000):
    s = get_settings()
    if not s.persist_runs:
        raise HTTPException(status_code=400, detail="Run persistence disabled")

    def gen():
        yield from iter_runs_ndjson_lines(
            db_path=Path(s.runs_db_path),
            limit=min(limit, 500_000),
        )

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/v1/runs/search", response_model=list[RunSummaryOut])
def search_diligence_runs(q: str = "", limit: int = 40) -> list[RunSummaryOut]:
    """Search persisted runs by substring (query text, domain, or id)."""
    s = get_settings()
    if not s.persist_runs:
        return []
    rows = search_runs(db_path=Path(s.runs_db_path), q=q, limit=min(limit, 200))
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


def _memo_markdown_from_run(row: dict[str, Any]) -> str:
    res = row.get("result") or {}
    body = res.get("final_report")
    if body is None:
        body = json.dumps(res, indent=2, default=str)
    elif not isinstance(body, str):
        body = str(body)
    header_lines = [
        "# Diligence memo",
        "",
        f"- **Run id**: `{row.get('id')}`",
        f"- **Created**: {row.get('created_at')}",
        f"- **Domain**: {row.get('domain')}",
        f"- **Query**: {row.get('query')}",
        "",
        "---",
        "",
    ]
    return "\n".join(header_lines) + body


@app.get("/v1/runs/{run_id}/memo.md", response_class=PlainTextResponse)
def export_run_memo(run_id: str) -> PlainTextResponse:
    """Export ``final_report`` as Markdown/plain text."""
    s = get_settings()
    if not s.persist_runs:
        raise HTTPException(status_code=404, detail="Run persistence disabled")
    row = get_run(db_path=Path(s.runs_db_path), run_id=run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    text = _memo_markdown_from_run(row)
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")


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
    content = await file.read()
    return _ingest_pdf_core_bytes(content, file.filename or "", use_ocr)


@app.post("/v1/ingest/local", response_model=IngestResponse)
def ingest_local_pdf(body: LocalPathIngestRequest) -> IngestResponse:
    """Index a PDF already on disk (requires ``LEGAL_INTEL_ALLOW_LOCAL_PATHS``)."""
    s = get_settings()
    raw_allow = (s.legal_intel_allow_local_paths or "").strip()
    if not raw_allow:
        raise HTTPException(
            status_code=403,
            detail="Local path ingest disabled. Set LEGAL_INTEL_ALLOW_LOCAL_PATHS to comma-separated absolute directory prefixes.",
        )
    prefixes = parse_allow_prefixes(raw_allow)
    if not prefixes:
        raise HTTPException(status_code=403, detail="No valid allowlist prefixes configured.")
    try:
        p = Path(body.path).expanduser().resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}") from e
    if not p.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    if p.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if not is_path_under_allowlist(p, prefixes):
        raise HTTPException(
            status_code=403,
            detail="Path is not under any configured allow prefix",
        )
    content = p.read_bytes()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    return _ingest_pdf_core_bytes(content, p.name, body.use_ocr)


@app.post("/v1/ingest/batch", response_model=BatchIngestResponse)
async def ingest_batch(
    files: list[UploadFile] = File(...),
    use_ocr: bool = False,
) -> BatchIngestResponse:
    """Upload many PDFs in one multipart request (partial success: ``items`` + ``errors``)."""
    items: list[IngestResponse] = []
    errors: list[dict[str, Any]] = []
    for f in files:
        try:
            raw = await f.read()
            items.append(_ingest_pdf_core_bytes(raw, f.filename or "unnamed.pdf", use_ocr))
        except HTTPException as he:
            detail = he.detail
            if isinstance(detail, list):
                detail = str(detail)
            errors.append({"filename": f.filename, "detail": detail})
        except Exception as e:
            errors.append({"filename": f.filename, "detail": str(e)})
    return BatchIngestResponse(items=items, errors=errors)


@app.delete("/v1/runs/{run_id}")
def delete_diligence_run(run_id: str) -> dict[str, Any]:
    s = get_settings()
    if not s.persist_runs:
        raise HTTPException(status_code=400, detail="Run persistence disabled")
    ok = delete_run(db_path=Path(s.runs_db_path), run_id=run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"deleted": True, "id": run_id}


@app.get("/v1/qdrant/info")
def qdrant_collection_info() -> dict[str, Any]:
    """Inspect the same vector store client used for RAG (includes shared :memory: singleton)."""
    s = get_settings()
    name = s.qdrant_collection
    try:
        store = LegalVectorStore()
        client = store._client
        exists = client.collection_exists(name)
        info = client.get_collection(name) if exists else None
        points = None
        if exists:
            try:
                cnt = client.count(collection_name=name, exact=True)
                points = getattr(cnt, "count", cnt)
            except Exception:
                points = None
        return {
            "qdrant_url_setting": s.qdrant_url,
            "collection": name,
            "exists": exists,
            "points_count": points,
            "status": getattr(info, "status", None) if info else None,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {e}") from e


@app.get("/v1/uploads/manifest")
def uploads_manifest(tail: int = 80) -> dict[str, Any]:
    """Tail of ``manifest.jsonl`` next to persisted uploads (ingest audit trail)."""
    s = get_settings()
    if not s.persist_uploads:
        raise HTTPException(status_code=400, detail="Upload persistence disabled")
    return tail_upload_manifest(s.upload_storage_dir, tail_lines=tail)


@app.get("/v1/uploads/files")
def uploads_directory_listing(limit: int = 100) -> dict[str, Any]:
    """Bounded listing of files under ``UPLOAD_STORAGE_DIR`` (mtime-descending)."""
    s = get_settings()
    if not s.persist_uploads:
        raise HTTPException(status_code=400, detail="Upload persistence disabled")
    lim = max(1, min(limit, 500))
    return list_upload_storage_files(s.upload_storage_dir, limit=lim)


@app.get("/v1/documents")
def list_indexed_documents(max_points: int = 4000) -> dict[str, Any]:
    """Distinct indexed documents by scanning chunk payloads (bounded work)."""
    store = LegalVectorStore()
    mp = max(100, min(max_points, 50_000))
    docs = store.aggregate_indexed_documents(max_points=mp)
    return {"documents": docs, "scan_budget_points": mp}


@app.get("/v1/documents/{doc_id}/chunks")
def get_document_chunks(
    doc_id: str,
    limit: int = 64,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Paginated chunk payloads for debugging retrieval (large ``text`` fields may truncate)."""
    store = LegalVectorStore()
    rows, next_c = store.scroll_chunks_for_document(doc_id, limit=limit, cursor=cursor)
    out: list[dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        t = item.get("text")
        if isinstance(t, str) and len(t) > 8000:
            item["text"] = t[:8000] + "\n… [truncated]"
        out.append(item)
    return {"doc_id": doc_id, "chunks": out, "next_cursor": next_c}


@app.delete("/v1/documents/{doc_id}")
def purge_document_vectors(doc_id: str) -> dict[str, Any]:
    """Remove all vectors for ``doc_id`` from Qdrant (does not delete SQLite runs)."""
    if not doc_id.strip():
        raise HTTPException(status_code=400, detail="Missing doc_id")
    store = LegalVectorStore()
    n = store.delete_document_vectors(doc_id)
    return {"doc_id": doc_id, "vectors_removed": n}


@app.post("/v1/documents/purge")
def purge_documents_batch(body: DocumentPurgeBatchRequest) -> dict[str, Any]:
    """Remove vectors for many ``doc_id`` values (destructive; SQLite runs unchanged)."""
    store = LegalVectorStore()
    counts = store.delete_document_vectors_batch(body.doc_ids)
    total = sum(counts.values())
    return {"vectors_removed_total": total, "by_doc_id": counts}


@app.post("/v1/rag/near-duplicate-chunks")
def rag_near_duplicate_chunks(body: NearDuplicateChunksRequest) -> dict[str, Any]:
    """
    Pairwise cosine similarity over chunk texts within one ``doc_id`` (bounded ``max_chunks``).
    Uses the same embedding backend as ingestion/RAG — ideal for duplicate-page / OCR overlap QA.
    """
    try:
        return near_duplicate_chunk_pairs(
            doc_id=body.doc_id.strip(),
            min_similarity=body.min_similarity,
            max_chunks=body.max_chunks,
            max_pairs=body.max_pairs,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/v1/rag/document-summary", response_model=DocumentSummaryResponse)
def rag_document_summary(body: DocumentSummaryRequest) -> DocumentSummaryResponse:
    """Scoped retrieval + synthesis summary — uses ``synthesis`` model routing (Ollama when configured)."""
    lim, did, user, sources = _prepare_document_summary_parts(body)
    summary = chat_complete(SUMMARIZE_SYSTEM, user, temperature=0.08, task="synthesis")
    return DocumentSummaryResponse(
        doc_id=did,
        summary=summary,
        sources=sources,
        retrieval_top_k=lim,
    )


@app.post("/v1/rag/document-summary/stream")
def rag_document_summary_stream(body: DocumentSummaryRequest):
    """SSE: sources event then token stream for synthesis-task summary (same retrieval as non-stream)."""

    def event_gen():
        try:
            lim, did, user, sources = _prepare_document_summary_parts(body)
            yield _sse_payload(
                {"event": "sources", "doc_id": did, "sources": sources, "retrieval_top_k": lim}
            )
            for piece in chat_stream(SUMMARIZE_SYSTEM, user, temperature=0.08, task="synthesis"):
                yield _sse_payload({"event": "token", "text": piece})
            yield _sse_payload({"event": "done"})
        except Exception as e:
            yield _sse_payload({"event": "error", "message": str(e)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/v1/rag/compare-documents", response_model=CompareDocumentsResponse)
def rag_compare_documents(body: CompareDocumentsRequest) -> CompareDocumentsResponse:
    """Side-by-side retrieval from two indexed docs + specialist comparison (local Ollama / routed LLM)."""
    da, db, user, src_a, src_b, per = _prepare_compare_documents_parts(body)
    comparison = chat_complete(
        COMPARE_DOCUMENTS_SYSTEM,
        user,
        temperature=0.06,
        task="specialist",
    )
    return CompareDocumentsResponse(
        doc_id_a=da,
        doc_id_b=db,
        comparison=comparison,
        sources_a=src_a,
        sources_b=src_b,
        retrieval_top_k_per_side=per,
    )


@app.post("/v1/rag/compare-documents/stream")
def rag_compare_documents_stream(body: CompareDocumentsRequest):
    """SSE: sources for both docs then specialist comparison tokens."""

    def event_gen():
        try:
            da, db, user, src_a, src_b, per = _prepare_compare_documents_parts(body)
            yield _sse_payload(
                {
                    "event": "sources",
                    "doc_id_a": da,
                    "doc_id_b": db,
                    "sources_a": src_a,
                    "sources_b": src_b,
                    "retrieval_top_k_per_side": per,
                }
            )
            for piece in chat_stream(
                COMPARE_DOCUMENTS_SYSTEM, user, temperature=0.06, task="specialist"
            ):
                yield _sse_payload({"event": "token", "text": piece})
            yield _sse_payload({"event": "done"})
        except Exception as e:
            yield _sse_payload({"event": "error", "message": str(e)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


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
    store = LegalVectorStore()
    lim = _effective_retrieval_limit(body)
    hits = store.search(body.question, limit=lim, doc_id=body.doc_id)
    ctx = format_context_block(hits)
    user = f"QUESTION:\n{body.question}\n\nCONTEXT EXCERPTS:\n{ctx}"
    answer = chat_complete(QUERY_SYSTEM, user, temperature=0.05, task="specialist")
    return QueryResponse(answer=answer, sources=_rag_sources_from_hits(hits))


@app.post("/v1/query/retrieve-only", response_model=RetrieveOnlyResponse)
def rag_retrieve_only(body: QueryRequest) -> RetrieveOnlyResponse:
    """Same retrieval path as grounded Q&A but **no LLM** — debug ranking and context assembly."""
    store = LegalVectorStore()
    lim = _effective_retrieval_limit(body)
    hits = store.search(body.question, limit=lim, doc_id=body.doc_id)
    ctx = format_context_block(hits)
    return RetrieveOnlyResponse(
        sources=_rag_sources_from_hits(hits),
        formatted_context=ctx,
        retrieval_top_k=lim,
    )


@app.post("/v1/query/retrieve-only/batch", response_model=RetrieveBatchResponse)
def rag_retrieve_only_batch(body: QueryBatchRequest) -> RetrieveBatchResponse:
    """Batch retrieval + formatted context blocks without LLM calls (debug / agent prefetch)."""
    lim = _batch_retrieval_limit(body.limit)
    store = LegalVectorStore()
    items: list[RetrieveBatchItem] = []
    for raw_q in body.questions:
        q = (raw_q or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="Each question must be non-empty")
        hits = store.search(q, limit=lim, doc_id=body.doc_id)
        ctx = format_context_block(hits)
        items.append(
            RetrieveBatchItem(
                question=q,
                sources=_rag_sources_from_hits(hits),
                formatted_context=ctx,
                retrieval_top_k=lim,
            )
        )
    return RetrieveBatchResponse(items=items, retrieval_top_k_per_item=lim)


@app.post("/v1/query/stream")
def rag_query_stream(body: QueryRequest):
    def event_gen():
        try:
            store = LegalVectorStore()
            lim = _effective_retrieval_limit(body)
            hits = store.search(body.question, limit=lim, doc_id=body.doc_id)
            ctx = format_context_block(hits)
            src = _rag_sources_from_hits(hits)
            yield _sse_payload({"event": "sources", "sources": src})
            user = f"QUESTION:\n{body.question}\n\nCONTEXT EXCERPTS:\n{ctx}"
            for piece in chat_stream(QUERY_SYSTEM, user, temperature=0.05, task="specialist"):
                yield _sse_payload({"event": "token", "text": piece})
            yield _sse_payload({"event": "done"})
        except Exception as e:
            yield _sse_payload({"event": "error", "message": str(e)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/v1/query/batch", response_model=QueryBatchResponse)
def rag_query_batch(body: QueryBatchRequest) -> QueryBatchResponse:
    """Grounded Q&A for multiple questions with shared ``doc_id`` / retrieval depth (sequential LLM calls)."""
    lim = _batch_retrieval_limit(body.limit)
    store = LegalVectorStore()
    items: list[QueryBatchItem] = []
    for raw_q in body.questions:
        q = (raw_q or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="Each question must be non-empty")
        hits = store.search(q, limit=lim, doc_id=body.doc_id)
        ctx = format_context_block(hits)
        user = f"QUESTION:\n{q}\n\nCONTEXT EXCERPTS:\n{ctx}"
        ans = chat_complete(QUERY_SYSTEM, user, temperature=0.05, task="specialist")
        items.append(QueryBatchItem(question=q, answer=ans, sources=_rag_sources_from_hits(hits)))
    return QueryBatchResponse(items=items, retrieval_top_k_per_item=lim)


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
