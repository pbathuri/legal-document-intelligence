"""FastAPI smoke tests (mock LLM + in-memory Qdrant)."""

from __future__ import annotations

import io
import os

import pytest


@pytest.fixture
def api_client():
    os.environ["LEGAL_INTEL_MOCK_LLM"] = "1"
    os.environ["QDRANT_URL"] = ":memory:"
    os.environ["DILIGENCE_DOMAIN"] = "mna"
    os.environ["PERSIST_UPLOADS"] = "0"
    os.environ["PERSIST_RUNS"] = "0"
    os.environ["LLM_PROVIDER"] = "openai_compatible"
    os.environ["EMBEDDING_PROVIDER"] = "sentence_transformers"
    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mock_llm"] is True
    assert body["embedding_provider"] == "sentence_transformers"
    assert "models" in body


def test_analyze_india_requires_doc_ids(api_client):
    r = api_client.post(
        "/v1/analyze",
        json={"query": "Title chain", "domain": "india_re", "doc_ids": []},
    )
    assert r.status_code == 400


def test_analyze_mna_roundtrip(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="c1",
        doc_label="contract.pdf",
        chunks=[
            ("Indemnification is capped at five million USD.", {"page_start": 1, "page_end": 1}),
        ],
    )
    r = api_client.post(
        "/v1/analyze",
        json={"query": "Summarize indemnity risk.", "domain": "mna"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["domain"] == "mna"
    assert "final_report" in data["result"]


def test_query_grounded(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="q1",
        doc_label="nda.pdf",
        chunks=[
            (
                "Confidential Information must not be disclosed for 3 years.",
                {"page_start": 1, "page_end": 1},
            ),
        ],
    )
    r = api_client.post(
        "/v1/query",
        json={"question": "How long is confidentiality?", "doc_id": "q1"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert len(body["sources"]) >= 1


def test_ingest_pdf_minimal(api_client):
    import fitz

    buf = io.BytesIO()
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "This Sale Deed is executed between Seller A and Buyer B.")
    doc.save(buf)
    doc.close()
    buf.seek(0)
    r = api_client.post(
        "/v1/ingest",
        files={"file": ("minimal.pdf", buf.getvalue(), "application/pdf")},
    )
    assert r.status_code == 200
    out = r.json()
    assert "doc_id" in out
    assert out["chunks"] >= 1


def test_agents_manifest(api_client):
    r = api_client.get("/v1/agents")
    assert r.status_code == 200
    data = r.json()
    assert "model_routing" in data
    assert "langgraph_pipelines" in data


def test_analyze_stream_mna(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="s1",
        doc_label="c.pdf",
        chunks=[("Term: 12 months notice for termination.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/analyze/stream",
        json={"query": "List notice periods.", "domain": "mna"},
    )
    assert r.status_code == 200
    assert "data:" in r.text
    assert "done" in r.text


def test_query_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="sq1",
        doc_label="x.pdf",
        chunks=[("Governing law: Delaware.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/query/stream",
        json={"question": "What is governing law?", "doc_id": "sq1"},
    )
    assert r.status_code == 200
    assert "token" in r.text or "MOCK" in r.text


def test_persist_run(tmp_path, monkeypatch):
    db = tmp_path / "r.db"
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("DILIGENCE_DOMAIN", "mna")
    monkeypatch.setenv("PERSIST_UPLOADS", "0")
    monkeypatch.setenv("PERSIST_RUNS", "1")
    monkeypatch.setenv("RUNS_DB_PATH", str(db))
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="p1",
        doc_label="m.pdf",
        chunks=[("Payment net 30.", {"page_start": 1, "page_end": 1})],
    )
    with TestClient(app) as client:
        r = client.post(
            "/v1/analyze",
            json={"query": "Payment terms?", "domain": "mna"},
        )
        assert r.status_code == 200
        assert r.json().get("run_id")
        lst = client.get("/v1/runs")
        assert lst.status_code == 200
        assert len(lst.json()) >= 1
    get_settings.cache_clear()


def test_runtime(api_client):
    r = api_client.get("/v1/runtime")
    assert r.status_code == 200
    body = r.json()
    assert "python_version" in body
    assert "device" in body and body["device"].get("hostname")


def test_disk(api_client):
    r = api_client.get("/v1/disk")
    assert r.status_code == 200
    assert "free_bytes" in r.json()


def test_preflight(api_client):
    r = api_client.get("/v1/preflight")
    assert r.status_code == 200
    body = r.json()
    assert "ready" in body
    assert "checks" in body
    assert body["checks"]["embedding_provider"] == "sentence_transformers"
    assert body["checks"]["qdrant"]["ok"] is True


def test_local_ingest_disabled(api_client):
    r = api_client.post("/v1/ingest/local", json={"path": "/tmp/x.pdf"})
    assert r.status_code == 403


def test_effective_settings_redacts(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value-do-not-leak")
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("PERSIST_UPLOADS", "0")
    monkeypatch.setenv("PERSIST_RUNS", "0")
    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.get("/v1/settings/effective")
        assert r.status_code == 200
        assert r.json().get("openai_api_key") == "***"
    get_settings.cache_clear()


def test_qdrant_info(api_client):
    r = api_client.get("/v1/qdrant/info")
    assert r.status_code == 200
    body = r.json()
    assert body.get("exists") is True


def test_batch_ingest_two_pdfs(api_client):
    import fitz

    buf = io.BytesIO()
    doc = fitz.open()
    doc.new_page().insert_text((40, 72), "Batch contract clause one.")
    doc.save(buf)
    doc.close()
    pdf_bytes = buf.getvalue()
    files = [
        ("files", ("one.pdf", pdf_bytes, "application/pdf")),
        ("files", ("two.pdf", pdf_bytes, "application/pdf")),
    ]
    r = api_client.post("/v1/ingest/batch", files=files)
    assert r.status_code == 200
    data = r.json()
    assert len(data["items"]) == 2
    assert data["errors"] == []
    assert data["items"][0]["page_count"] is not None


def test_export_delete_run(tmp_path, monkeypatch):
    db = tmp_path / "exp.db"
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("PERSIST_RUNS", "1")
    monkeypatch.setenv("RUNS_DB_PATH", str(db))
    monkeypatch.setenv("PERSIST_UPLOADS", "0")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="ex1",
        doc_label="z.pdf",
        chunks=[("Clause X.", {"page_start": 1, "page_end": 1})],
    )
    with TestClient(app) as client:
        ar = client.post(
            "/v1/analyze",
            json={"query": "Any clause?", "domain": "mna"},
        )
        rid = ar.json()["run_id"]
        exp = client.get("/v1/runs/export")
        assert exp.status_code == 200
        assert rid in exp.text
        memo = client.get(f"/v1/runs/{rid}/memo.md")
        assert memo.status_code == 200
        assert "Diligence memo" in memo.text
        assert memo.headers.get("content-type", "").startswith("text/markdown")
        dr = client.delete(f"/v1/runs/{rid}")
        assert dr.status_code == 200
        gr = client.get(f"/v1/runs/{rid}")
        assert gr.status_code == 404
    get_settings.cache_clear()


def test_metrics_and_request_id(api_client):
    r = api_client.get("/health", headers={"X-Request-ID": "trace-test-1"})
    assert r.status_code == 200
    assert r.headers.get("x-request-id") == "trace-test-1"
    assert "x-process-time" in {k.lower() for k in r.headers.keys()}
    m = api_client.get("/v1/metrics")
    assert m.status_code == 200
    body = m.json()
    assert body["requests_total"] >= 2
    assert "by_path_bucket" in body


def test_documents_crud(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="dc1",
        doc_label="a.pdf",
        chunks=[("hello world clause", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.get("/v1/documents")
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert any(d["doc_id"] == "dc1" for d in docs)
    ch = api_client.get("/v1/documents/dc1/chunks")
    assert ch.status_code == 200
    assert len(ch.json()["chunks"]) >= 1
    d = api_client.delete("/v1/documents/dc1")
    assert d.status_code == 200
    assert d.json()["vectors_removed"] >= 1


def test_embeddings_warmup(api_client):
    r = api_client.post("/v1/embeddings/warmup")
    assert r.status_code == 200
    out = r.json()
    assert out.get("ok") is True
    assert out.get("dimension", 0) > 0


def test_ollama_host(api_client):
    r = api_client.get("/v1/ollama/host")
    assert r.status_code == 200
    body = r.json()
    assert "origin" in body


def test_upload_manifest_requires_persist(api_client):
    r = api_client.get("/v1/uploads/manifest")
    assert r.status_code == 400


def test_upload_manifest_tail(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("PERSIST_UPLOADS", "1")
    monkeypatch.setenv("UPLOAD_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    (tmp_path / "manifest.jsonl").write_text(
        '{"doc_id":"x","path":"/tmp/a.pdf"}\n', encoding="utf-8"
    )
    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.get("/v1/uploads/manifest?tail=5")
        assert r.status_code == 200
        data = r.json()
        assert data["exists"] is True
        assert len(data["items"]) == 1
        assert data["items"][0].get("doc_id") == "x"
    get_settings.cache_clear()


def test_build_endpoint(api_client):
    r = api_client.get("/v1/build")
    assert r.status_code == 200
    b = r.json()
    assert b.get("package_name") == "legal-document-intelligence"
    assert isinstance(b.get("api_version"), str) and len(b["api_version"]) >= 3


def test_llm_probe_mock(api_client):
    r = api_client.post("/v1/llm/probe")
    assert r.status_code == 200
    assert r.json().get("skipped") is True


def test_metrics_prometheus(api_client):
    r = api_client.get("/v1/metrics/prometheus")
    assert r.status_code == 200
    assert "legal_intel_http_requests_total" in r.text


def test_preflight_deep(api_client):
    r = api_client.get("/v1/preflight?deep=1")
    assert r.status_code == 200
    assert "ollama_host" in r.json()


def test_export_runs_json(tmp_path, monkeypatch):
    db = tmp_path / "js.db"
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("PERSIST_RUNS", "1")
    monkeypatch.setenv("RUNS_DB_PATH", str(db))
    monkeypatch.setenv("PERSIST_UPLOADS", "0")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="j1",
        doc_label="j.pdf",
        chunks=[("x", {"page_start": 1, "page_end": 1})],
    )
    with TestClient(app) as client:
        client.post("/v1/analyze", json={"query": "Q export json", "domain": "mna"})
        r = client.get("/v1/runs/export/json?limit=50")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0].get("result")
    get_settings.cache_clear()


def test_documents_purge_batch(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="p1",
        doc_label="p.pdf",
        chunks=[("batch purge", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post("/v1/documents/purge", json={"doc_ids": ["p1", "missing"]})
    assert r.status_code == 200
    out = r.json()
    assert out["vectors_removed_total"] >= 1
    assert "p1" in out["by_doc_id"]


def test_audit_jsonl_append(tmp_path, monkeypatch):
    logf = tmp_path / "audit.jsonl"
    monkeypatch.setenv("LEGAL_INTEL_AUDIT_JSONL", str(logf))
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        client.post("/v1/embeddings/warmup")
    assert logf.is_file()
    line = logf.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert "/v1/embeddings/warmup" in line
    get_settings.cache_clear()


def test_runs_search(tmp_path, monkeypatch):
    db = tmp_path / "search.db"
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("PERSIST_RUNS", "1")
    monkeypatch.setenv("RUNS_DB_PATH", str(db))
    monkeypatch.setenv("PERSIST_UPLOADS", "0")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="s1",
        doc_label="z.pdf",
        chunks=[("Clause X.", {"page_start": 1, "page_end": 1})],
    )
    with TestClient(app) as client:
        client.post(
            "/v1/analyze",
            json={"query": "UniqueSearchTokenABC indemnity", "domain": "mna"},
        )
        r = client.get("/v1/runs/search", params={"q": "UniqueSearchTokenABC"})
        assert r.status_code == 200
        assert len(r.json()) >= 1
        r2 = client.get("/v1/runs/search", params={"q": ""})
        assert r2.status_code == 200
        assert r2.json() == []
    get_settings.cache_clear()


def test_health_live_ready(api_client):
    assert api_client.get("/health/live").json()["status"] == "alive"
    r = api_client.get("/health/ready")
    assert r.status_code == 200
    assert "ready" in r.json()


def test_embedding_similarity(api_client):
    r = api_client.post(
        "/v1/embeddings/similarity",
        json={
            "text_a": "the indemnity clause caps liability.",
            "text_b": "the indemnity clause caps liability.",
        },
    )
    assert r.status_code == 200
    assert r.json()["cosine_similarity"] > 0.99
    assert r.json()["dimension"] > 0


def test_retrieve_only(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="ro1",
        doc_label="x.pdf",
        chunks=[("alpha beta gamma indemnity clause text.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/query/retrieve-only",
        json={"question": "alpha beta indemnity", "doc_id": "ro1"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["sources"]) >= 1
    assert body["formatted_context"]


def test_system_snapshot(api_client):
    r = api_client.get("/v1/system/snapshot")
    assert r.status_code == 200
    assert "loadavg" in r.json()


def test_vacuum_sqlite(tmp_path, monkeypatch):
    db = tmp_path / "vac.db"
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("PERSIST_RUNS", "1")
    monkeypatch.setenv("RUNS_DB_PATH", str(db))
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.post("/v1/maintenance/vacuum-sqlite")
        assert r.status_code == 200
        assert "bytes_after" in r.json()
    get_settings.cache_clear()


def test_ollama_generate_stream_rejected(api_client):
    r = api_client.post(
        "/v1/ollama/generate",
        json={"model": "m", "prompt": "p", "stream": True},
    )
    assert r.status_code == 400


def test_ollama_generate_proxied(monkeypatch):
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from unittest.mock import patch

    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    with patch(
        "legal_intel.api.main.ollama_native_generate",
        return_value={"response": "generated-body"},
    ):
        with TestClient(app) as client:
            r = client.post("/v1/ollama/generate", json={"model": "any", "prompt": "hi"})
            assert r.status_code == 200
            assert r.json().get("response") == "generated-body"
    get_settings.cache_clear()
