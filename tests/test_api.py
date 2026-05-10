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

    r_v1 = api_client.get("/v1/health")
    assert r_v1.status_code == 200
    assert r_v1.json() == body


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


def test_runs_stats_endpoint(api_client):
    r = api_client.get("/v1/runs/stats")
    assert r.status_code == 200
    body = r.json()
    assert "row_count" in body
    assert "by_domain" in body
    assert "db_path" in body


def test_runs_stats_after_persist(tmp_path, monkeypatch):
    db = tmp_path / "stats.db"
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("DILIGENCE_DOMAIN", "mna")
    monkeypatch.setenv("PERSIST_UPLOADS", "0")
    monkeypatch.setenv("PERSIST_RUNS", "1")
    monkeypatch.setenv("RUNS_DB_PATH", str(db))
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="st1",
        doc_label="z.pdf",
        chunks=[("Consideration is fifty dollars.", {"page_start": 1, "page_end": 1})],
    )
    with TestClient(app) as client:
        client.post(
            "/v1/analyze",
            json={"query": "Payment?", "domain": "mna"},
        )
        st = client.get("/v1/runs/stats").json()
        assert st["row_count"] >= 1
        assert st["by_domain"].get("mna", 0) >= 1
        assert st["created_at_min"] and st["created_at_max"]
    get_settings.cache_clear()


def test_embeddings_embed_texts(api_client):
    r = api_client.post(
        "/v1/embeddings/embed-texts",
        json={"texts": ["first passage about leases.", "second passage about leases."]},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert len(data["vectors"]) == 2
    assert data["dimension"] > 0


def test_query_retrieve_respects_limit(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="lim1",
        doc_label="x.pdf",
        chunks=[
            ("alpha indemnity clause one.", {"page_start": 1, "page_end": 1}),
            ("beta indemnity clause two.", {"page_start": 2, "page_end": 2}),
        ],
    )
    r = api_client.post(
        "/v1/query/retrieve-only",
        json={"question": "indemnity clause", "doc_id": "lim1", "limit": 1},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["retrieval_top_k"] == 1
    assert len(body["sources"]) == 1


def test_ollama_show_proxied(monkeypatch):
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
        "legal_intel.api.main.ollama_native_show",
        return_value={"license": "MIT", "parameters": "9B"},
    ):
        with TestClient(app) as client:
            r = client.post("/v1/ollama/show", json={"model": "dummy"})
            assert r.status_code == 200
            assert r.json().get("license") == "MIT"
    get_settings.cache_clear()


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


def test_embeddings_info(api_client):
    r = api_client.get("/v1/embeddings/info")
    assert r.status_code == 200
    body = r.json()
    assert body["dimension"] > 0
    assert body["embedding_provider"] == "sentence_transformers"
    assert "probe_encode_seconds" in body


def test_ollama_version_mocked(monkeypatch):
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
        "legal_intel.api.main.ollama_native_version",
        return_value={"version": "0.0.0-test"},
    ):
        with TestClient(app) as client:
            r = client.get("/v1/ollama/version")
            assert r.status_code == 200
            assert r.json().get("version") == "0.0.0-test"
    get_settings.cache_clear()


def test_ollama_chat_stream_rejected(api_client):
    r = api_client.post(
        "/v1/ollama/chat",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        },
    )
    assert r.status_code == 400


def test_ollama_chat_proxied(monkeypatch):
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
        "legal_intel.api.main.ollama_native_chat",
        return_value={"message": {"role": "assistant", "content": "pong"}},
    ):
        with TestClient(app) as client:
            r = client.post(
                "/v1/ollama/chat",
                json={
                    "model": "any",
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
            assert r.status_code == 200
            assert r.json()["message"]["content"] == "pong"
    get_settings.cache_clear()


def test_near_duplicate_chunks(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="dupx",
        doc_label="x.pdf",
        chunks=[
            ("The indemnity clause caps liability at five million dollars.", {"chunk_index": 0}),
            ("The indemnity clause caps liability at five million dollars.", {"chunk_index": 1}),
        ],
    )
    r = api_client.post(
        "/v1/rag/near-duplicate-chunks",
        json={"doc_id": "dupx", "min_similarity": 0.85},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["chunks_scanned"] >= 2
    assert len(body["pairs"]) >= 1


def test_system_process_endpoint(api_client):
    r = api_client.get("/v1/system/process")
    assert r.status_code == 200
    body = r.json()
    assert "pid" in body


def test_query_batch(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="bq1",
        doc_label="c.pdf",
        chunks=[
            ("Payment net thirty days from invoice.", {"page_start": 1, "page_end": 1}),
            ("Governing law is Delaware.", {"page_start": 2, "page_end": 2}),
        ],
    )
    r = api_client.post(
        "/v1/query/batch",
        json={
            "questions": ["What is payment term?", "What is governing law?"],
            "doc_id": "bq1",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["items"]) == 2
    assert data["retrieval_top_k_per_item"] >= 1


def test_document_summary(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="sum1",
        doc_label="d.pdf",
        chunks=[
            ("The seller warrants good title to the property.", {"page_start": 1, "page_end": 1}),
        ],
    )
    r = api_client.post(
        "/v1/rag/document-summary",
        json={
            "doc_id": "sum1",
            "retrieval_query": "warranty title property",
            "instruction": "Bullet summary only.",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["doc_id"] == "sum1"
    assert body["summary"]
    assert len(body["sources"]) >= 1


def test_ollama_embed_proxy_mocked(monkeypatch):
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
        "legal_intel.api.main.ollama_native_embed_raw",
        return_value={"model": "nomic", "embeddings": [[0.1, 0.2]]},
    ):
        with TestClient(app) as client:
            r = client.post("/v1/ollama/embed-proxy", json={"input": ["hello"]})
            assert r.status_code == 200
            assert r.json().get("model") == "nomic"
    get_settings.cache_clear()


def test_optimize_sqlite(tmp_path, monkeypatch):
    db = tmp_path / "opt.db"
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("PERSIST_RUNS", "1")
    monkeypatch.setenv("RUNS_DB_PATH", str(db))
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from legal_intel.config import get_settings
    from legal_intel.persistence.runs import insert_run

    insert_run(
        db_path=db,
        domain="mna",
        query="q",
        doc_ids=[],
        result={"final_report": "x"},
    )
    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.post("/v1/maintenance/optimize-sqlite")
        assert r.status_code == 200
        assert r.json().get("pragma_optimize") is True
    get_settings.cache_clear()


def test_retrieve_only_batch(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="rob1",
        doc_label="c.pdf",
        chunks=[
            ("Payment net thirty days from invoice.", {"page_start": 1, "page_end": 1}),
            ("Governing law is Delaware.", {"page_start": 2, "page_end": 2}),
        ],
    )
    r = api_client.post(
        "/v1/query/retrieve-only/batch",
        json={
            "questions": ["What is payment term?", "What is governing law?"],
            "doc_id": "rob1",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["items"]) == 2
    assert data["retrieval_top_k_per_item"] >= 1
    for item in data["items"]:
        assert item["formatted_context"]
        assert len(item["sources"]) >= 1
        assert "answer" not in item


def test_compare_documents(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="cmp_a",
        doc_label="a.pdf",
        chunks=[
            ("Party A shall pay Party B within 14 days.", {"page_start": 1, "page_end": 1}),
        ],
    )
    store.upsert_document_chunks(
        doc_id="cmp_b",
        doc_label="b.pdf",
        chunks=[
            (
                "Payment is due within 30 calendar days of invoice.",
                {"page_start": 1, "page_end": 1},
            ),
        ],
    )
    r = api_client.post(
        "/v1/rag/compare-documents",
        json={
            "doc_id_a": "cmp_a",
            "doc_id_b": "cmp_b",
            "retrieval_query": "payment days invoice",
            "instruction": "List payment timing differences only.",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["doc_id_a"] == "cmp_a"
    assert body["doc_id_b"] == "cmp_b"
    assert body["comparison"]
    assert len(body["sources_a"]) >= 1
    assert len(body["sources_b"]) >= 1


def test_ollama_ps_mocked(monkeypatch):
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from unittest.mock import patch

    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    fake = {"models": [{"name": "llama3.2", "size": 1, "digest": "x"}]}
    with patch(
        "legal_intel.api.main.fetch_ollama_running_models",
        return_value=fake,
    ):
        with TestClient(app) as client:
            r = client.get("/v1/ollama/ps")
            assert r.status_code == 200
            assert r.json() == fake
    get_settings.cache_clear()


def test_integrity_sqlite(tmp_path, monkeypatch):
    db = tmp_path / "int.db"
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("PERSIST_RUNS", "1")
    monkeypatch.setenv("RUNS_DB_PATH", str(db))
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from legal_intel.config import get_settings
    from legal_intel.persistence.runs import insert_run

    insert_run(
        db_path=db,
        domain="mna",
        query="q",
        doc_ids=[],
        result={"final_report": "x"},
    )
    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.post("/v1/maintenance/integrity-sqlite")
        assert r.status_code == 200
        out = r.json()
        assert out.get("ok") is True
        assert out.get("integrity_check") == "ok"
    get_settings.cache_clear()


def test_uploads_files_list(tmp_path, monkeypatch):
    store_dir = tmp_path / "upload_store"
    store_dir.mkdir()
    (store_dir / "sample.bin").write_bytes(b"abc")
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("PERSIST_UPLOADS", "1")
    monkeypatch.setenv("UPLOAD_STORAGE_DIR", str(store_dir))
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.get("/v1/uploads/files")
        assert r.status_code == 200
        body = r.json()
        assert body["scanned_files"] >= 1
        names = {f["relative_path"] for f in body["files"]}
        assert "sample.bin" in names
    get_settings.cache_clear()


def test_runtime_storage_inventory(tmp_path, monkeypatch):
    up = tmp_path / "up"
    up.mkdir()
    (up / "a.pdf").write_bytes(b"%PDF-1 fake")
    db = tmp_path / "runs.sqlite"
    db.write_bytes(b"sqlite")
    (up / "manifest.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("PERSIST_UPLOADS", "1")
    monkeypatch.setenv("UPLOAD_STORAGE_DIR", str(up))
    monkeypatch.setenv("RUNS_DB_PATH", str(db))
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.get("/v1/runtime/storage")
        assert r.status_code == 200
        body = r.json()
        assert body["upload_storage_file_count"] >= 2
        assert body["runs_db_bytes"] == len(b"sqlite")
        assert body["manifest_line_count"] == 2
    get_settings.cache_clear()


def test_ollama_agent_stack_mocked(monkeypatch):
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from unittest.mock import patch

    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    with (
        patch(
            "legal_intel.runtime.ollama_agent_stack.fetch_ollama_model_names",
            return_value=(["m1"], None),
        ),
        patch(
            "legal_intel.runtime.ollama_agent_stack.ollama_native_version",
            return_value={"version": "x.y"},
        ),
        patch(
            "legal_intel.runtime.ollama_agent_stack.fetch_ollama_running_models",
            return_value={"models": []},
        ),
    ):
        with TestClient(app) as client:
            r = client.get("/v1/ollama/agent-stack")
            assert r.status_code == 200
            j = r.json()
            assert j["model_routing"]["specialist"]
            assert j["daemon_version"]["version"] == "x.y"
            assert j["api_tags"]["models"] == ["m1"]
    get_settings.cache_clear()


def test_document_summary_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="ss1",
        doc_label="z.pdf",
        chunks=[("Liquidated damages capped at two percent.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/document-summary/stream",
        json={
            "doc_id": "ss1",
            "retrieval_query": "damages cap",
            "instruction": "One sentence.",
        },
    )
    assert r.status_code == 200
    assert "sources" in r.text
    assert "done" in r.text


def test_compare_documents_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="csa",
        doc_label="a.pdf",
        chunks=[("Fee is two percent of enterprise value.", {"page_start": 1, "page_end": 1})],
    )
    store.upsert_document_chunks(
        doc_id="csb",
        doc_label="b.pdf",
        chunks=[("Success fee equals 2% of EV.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/compare-documents/stream",
        json={
            "doc_id_a": "csa",
            "doc_id_b": "csb",
            "retrieval_query": "fee percent",
            "instruction": "Compare fee language.",
        },
    )
    assert r.status_code == 200
    assert "sources_a" in r.text
    assert "token" in r.text or "MOCK" in r.text


def test_cross_document_summary(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="xd1",
        doc_label="a.pdf",
        chunks=[("Seller warrants title to the parcel.", {"page_start": 1, "page_end": 1})],
    )
    store.upsert_document_chunks(
        doc_id="xd2",
        doc_label="b.pdf",
        chunks=[("Buyer accepts title as-is with no warranty.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/cross-document-summary",
        json={
            "doc_ids": ["xd1", "xd2"],
            "retrieval_query": "warranty title",
            "instruction": "Note tensions only.",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["doc_ids"] == ["xd1", "xd2"]
    assert body["summary"]
    assert "xd1" in body["sources_by_doc_id"]
    assert "xd2" in body["sources_by_doc_id"]


def test_query_citations(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="qc1",
        doc_label="x.pdf",
        chunks=[("Interest rate is LIBOR plus two percent.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/query/citations",
        json={"question": "What is the interest rate?", "doc_id": "qc1"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer_markdown"]
    assert isinstance(body["citations"], list)
    assert body["structured"]
    assert len(body["sources"]) >= 1


def test_checkpoint_sqlite(tmp_path, monkeypatch):
    db = tmp_path / "walchk.db"
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("PERSIST_RUNS", "1")
    monkeypatch.setenv("RUNS_DB_PATH", str(db))
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from legal_intel.config import get_settings
    from legal_intel.persistence.runs import insert_run

    insert_run(
        db_path=db,
        domain="mna",
        query="q",
        doc_ids=[],
        result={"final_report": "x"},
    )
    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.post("/v1/maintenance/checkpoint-sqlite", json={"truncate_wal": False})
        assert r.status_code == 200
        assert "journal_mode" in r.json()
    get_settings.cache_clear()


def test_ollama_inspect_batch_mocked(monkeypatch):
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
        "legal_intel.api.main.ollama_native_show",
        return_value={"license": "MIT"},
    ):
        with TestClient(app) as client:
            r = client.post(
                "/v1/ollama/models/inspect-batch",
                json={"models": ["a", "b"]},
            )
            assert r.status_code == 200
            j = r.json()
            assert j["count"] == 2
            assert all(x["ok"] for x in j["items"])
    get_settings.cache_clear()


def test_ollama_tags_mocked(monkeypatch):
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from unittest.mock import patch

    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    fake = {"models": [{"name": "x", "digest": "abc"}]}
    with patch("legal_intel.api.main.fetch_ollama_tags_raw", return_value=fake):
        with TestClient(app) as client:
            r = client.get("/v1/ollama/tags")
            assert r.status_code == 200
            assert r.json() == fake
    get_settings.cache_clear()


def test_runtime_git_endpoint(api_client):
    r = api_client.get("/v1/runtime/git")
    assert r.status_code == 200
    body = r.json()
    assert "git_available" in body


def test_embed_local_text_files(tmp_path, monkeypatch):
    allow = tmp_path / "allowed"
    allow.mkdir()
    f1 = allow / "a.txt"
    f1.write_text("hello local embed", encoding="utf-8")
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("LEGAL_INTEL_ALLOW_LOCAL_PATHS", str(allow))
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.post(
            "/v1/embeddings/embed-local-text-files",
            json={"paths": [str(f1)]},
        )
        assert r.status_code == 200
        out = r.json()
        assert out["count_ok"] == 1
        assert len(out["items"]) == 1
        assert out["items"][0]["ok"] is True
        assert out["items"][0]["vector"]
    get_settings.cache_clear()


def test_cross_document_summary_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="cds1",
        doc_label="a.pdf",
        chunks=[("Alpha covenant applies.", {"page_start": 1, "page_end": 1})],
    )
    store.upsert_document_chunks(
        doc_id="cds2",
        doc_label="b.pdf",
        chunks=[("Beta covenant applies.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/cross-document-summary/stream",
        json={
            "doc_ids": ["cds1", "cds2"],
            "retrieval_query": "covenant",
            "instruction": "Summarize.",
        },
    )
    assert r.status_code == 200
    assert "sources_by_doc_id" in r.text
    assert "done" in r.text


def test_structured_extract(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="se1",
        doc_label="c.pdf",
        chunks=[("Closing on January 15, 2026.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/structured-extract",
        json={
            "doc_id": "se1",
            "retrieval_query": "closing date",
            "categories": ["closing_date", "parties"],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["doc_id"] == "se1"
    assert body["extraction"]
    assert len(body["sources"]) >= 1


def test_timeline_extract(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="tl1",
        doc_label="d.pdf",
        chunks=[("Effective date March 1, 2026; termination notice 90 days.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/timeline-extract",
        json={"doc_id": "tl1", "retrieval_query": "date termination"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["doc_id"] == "tl1"
    assert body["timeline"]
    assert len(body["sources"]) >= 1


def test_maintenance_stats_sqlite(api_client):
    r = api_client.get("/v1/maintenance/stats-sqlite")
    assert r.status_code == 200
    body = r.json()
    assert "path" in body
    assert "exists" in body


def test_runtime_host_metrics(api_client):
    r = api_client.get("/v1/runtime/host-metrics")
    assert r.status_code == 200
    assert "cpu_interval_seconds" in r.json()


def test_embeddings_pairwise_matrix(api_client):
    r = api_client.post(
        "/v1/embeddings/pairwise-matrix",
        json={
            "texts": [
                "first indemnity clause alpha.",
                "second indemnity clause beta.",
                "unrelated cookie recipe.",
            ]
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 3
    assert len(data["matrix"]) == 3
    assert len(data["matrix"][0]) == 3
    assert data["matrix"][0][0] >= 0.99
    assert data["embedding_provider"] == "sentence_transformers"


def test_ollama_generate_batch_proxied(monkeypatch):
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from unittest.mock import patch

    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    calls: list[int] = []

    def fake_gen(origin: str, payload: dict, *, timeout_seconds: float) -> dict:
        calls.append(1)
        return {"response": payload.get("prompt", "")[:20], "model": payload.get("model")}

    with patch(
        "legal_intel.api.main.ollama_native_generate",
        side_effect=fake_gen,
    ):
        with TestClient(app) as client:
            r = client.post(
                "/v1/ollama/generate/batch",
                json={"model": "m", "prompts": ["one", "two"]},
            )
            assert r.status_code == 200
            out = r.json()
            assert out["count"] == 2
            assert len(calls) == 2
            assert out["items"][0]["ok"] is True
            assert out["items"][1]["ok"] is True
    get_settings.cache_clear()


def test_query_hyde(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="hyde1",
        doc_label="e.pdf",
        chunks=[("The indemnity cap is five million USD for fundamental breaches.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/query/hyde",
        json={"question": "What is the indemnity cap?", "doc_id": "hyde1", "limit": 4},
    )
    assert r.status_code == 200
    body = r.json()
    assert "hypothetical_document" in body
    assert body["answer"]
    assert len(body["sources"]) >= 1


def test_risk_scan(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="rs1",
        doc_label="f.pdf",
        chunks=[("Seller makes no warranty as to tax liabilities after closing.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/risk-scan",
        json={"doc_id": "rs1", "retrieval_query": "warranty tax"},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["doc_id"] == "rs1"
    assert b["risk_register"]
    assert len(b["sources"]) >= 1


def test_timeline_extract_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="tls1",
        doc_label="g.pdf",
        chunks=[("Board approval required before week 2 of January 2026.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/timeline-extract/stream",
        json={"doc_id": "tls1", "retrieval_query": "January board"},
    )
    assert r.status_code == 200
    assert "sources" in r.text
    assert "done" in r.text


def test_runtime_network(api_client):
    r = api_client.get("/v1/runtime/network")
    assert r.status_code == 200
    j = r.json()
    assert "hostname" in j
    assert "psutil_available" in j


def test_runtime_local_path_allowlist(tmp_path, monkeypatch):
    allow = tmp_path / "zallowed"
    allow.mkdir()
    monkeypatch.setenv("LEGAL_INTEL_MOCK_LLM", "1")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("LEGAL_INTEL_ALLOW_LOCAL_PATHS", str(allow))
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    from legal_intel.config import get_settings

    get_settings.cache_clear()
    from legal_intel.api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.get("/v1/runtime/local-path-allowlist")
        assert r.status_code == 200
        data = r.json()
        assert data["prefix_count"] >= 1
        assert data["items"][0]["exists"] is True
    get_settings.cache_clear()


def test_glossary_extract(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="gl1",
        doc_label="defs.pdf",
        chunks=[
            (
                '"Material Adverse Effect" means any change that is materially adverse to the business.',
                {"page_start": 1, "page_end": 1},
            )
        ],
    )
    r = api_client.post(
        "/v1/rag/glossary-extract",
        json={"doc_id": "gl1", "retrieval_query": "Material Adverse"},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["doc_id"] == "gl1"
    assert b["glossary"]
    assert len(b["sources"]) >= 1


def test_risk_scan_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="rss1",
        doc_label="r.pdf",
        chunks=[("No survival of representations after Closing.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/risk-scan/stream",
        json={"doc_id": "rss1", "retrieval_query": "representations"},
    )
    assert r.status_code == 200
    assert "sources" in r.text and "done" in r.text


def test_query_hyde_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="hys1",
        doc_label="h.pdf",
        chunks=[("Purchase price shall be adjusted per working capital schedule.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/query/hyde/stream",
        json={"question": "How is purchase price adjusted?", "doc_id": "hys1"},
    )
    assert r.status_code == 200
    assert "hypothetical_document" in r.text
    assert "done" in r.text


def test_embeddings_centroid_similarities(api_client):
    r = api_client.post(
        "/v1/embeddings/centroid-similarities",
        json={
            "texts": [
                "The lease term is ten years with renewal option.",
                "Renewal may extend the lease for five additional years.",
                "Unrelated: shipping crates arrive Tuesday.",
            ]
        },
    )
    assert r.status_code == 200
    d = r.json()
    assert d["count"] == 3
    assert len(d["centroid"]) == d["dimension"]
    assert len(d["cosine_to_centroid"]) == 3


def test_cross_document_contradictions(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="cd1",
        doc_label="spa.pdf",
        chunks=[("Indemnity survives for twenty-four months.", {"page_start": 1, "page_end": 1})],
    )
    store.upsert_document_chunks(
        doc_id="cd2",
        doc_label="disclosure.pdf",
        chunks=[("Indemnity for breaches survives twelve months only.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/cross-document-contradictions",
        json={
            "doc_ids": ["cd1", "cd2"],
            "retrieval_query": "indemnity survival months",
        },
    )
    assert r.status_code == 200
    b = r.json()
    assert len(b["doc_ids"]) == 2
    assert b["contradictions"]
    assert "cd1" in b["sources_by_doc_id"]


def test_document_outline(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="out1",
        doc_label="agr.pdf",
        chunks=[
            ("Article 5 — Confidentiality. Recipient shall protect Discloser's information.", {"page_start": 1, "page_end": 1}),
            ("Article 6 — Term; termination for material breach after thirty days notice.", {"page_start": 2, "page_end": 2}),
        ],
    )
    r = api_client.post(
        "/v1/rag/document-outline",
        json={"doc_id": "out1", "retrieval_query": "Article confidentiality termination"},
    )
    assert r.status_code == 200
    o = r.json()
    assert o["doc_id"] == "out1"
    assert o["outline"]
    assert len(o["sources"]) >= 1


def test_glossary_extract_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="gls1",
        doc_label="t.pdf",
        chunks=[('"Seller" means Party A.', {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/glossary-extract/stream",
        json={"doc_id": "gls1", "retrieval_query": "Seller"},
    )
    assert r.status_code == 200
    assert "sources" in r.text and "done" in r.text


def test_runtime_rlimits(api_client):
    r = api_client.get("/v1/runtime/rlimits")
    assert r.status_code == 200
    body = r.json()
    assert "platform" in body
    assert "rlimits_available" in body or "note" in body


def test_runtime_sys_path(api_client):
    r = api_client.get("/v1/runtime/sys-path")
    assert r.status_code == 200
    j = r.json()
    assert "paths" in j
    assert j["shown_count"] >= 1


def test_embeddings_nearest_to_query(api_client):
    r = api_client.post(
        "/v1/embeddings/nearest-to-query",
        json={
            "query": "indemnity survival period after closing",
            "candidates": [
                "Indemnity obligations survive for twenty-four months.",
                "The confidentiality clause lasts five years.",
                "Parking spaces are allocated in Schedule B.",
            ],
        },
    )
    assert r.status_code == 200
    d = r.json()
    assert len(d["ranked"]) == 3
    assert d["ranked"][0]["index"] in (0, 1, 2)
    assert d["dimension"] > 0


def test_cross_document_contradictions_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="cx1",
        doc_label="a.pdf",
        chunks=[("Cap on liability is ten million.", {"page_start": 1, "page_end": 1})],
    )
    store.upsert_document_chunks(
        doc_id="cx2",
        doc_label="b.pdf",
        chunks=[("Liability is uncapped for fraud.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/cross-document-contradictions/stream",
        json={"doc_ids": ["cx1", "cx2"], "retrieval_query": "liability cap"},
    )
    assert r.status_code == 200
    assert "sources_by_doc_id" in r.text and "done" in r.text


def test_document_outline_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="dos1",
        doc_label="o.pdf",
        chunks=[("Section 4.2 — Non-compete for twelve months post-closing.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/document-outline/stream",
        json={"doc_id": "dos1", "retrieval_query": "non-compete"},
    )
    assert r.status_code == 200
    assert "sources" in r.text and "done" in r.text


def test_diligence_checklist(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="dd1",
        doc_label="dd.pdf",
        chunks=[("Environmental permits must be transferred before Closing.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/diligence-checklist",
        json={"doc_id": "dd1", "retrieval_query": "environmental permits closing"},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["checklist"]
    assert b["doc_id"] == "dd1"


def test_diligence_checklist_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="dds1",
        doc_label="x.pdf",
        chunks=[("Tax returns for three fiscal years shall be provided in the data room.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/diligence-checklist/stream",
        json={"doc_id": "dds1", "retrieval_query": "tax"},
    )
    assert r.status_code == 200
    assert "sources" in r.text and "done" in r.text


def test_runtime_path_entries(api_client):
    r = api_client.get("/v1/runtime/path-entries")
    assert r.status_code == 200
    j = r.json()
    assert "entries" in j
    assert "path_separator" in j


def test_issue_spotter(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="is1",
        doc_label="iss.pdf",
        chunks=[("Buyer may withhold up to ten percent of the purchase price in escrow.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/issue-spotter",
        json={"doc_id": "is1", "retrieval_query": "escrow withhold"},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["doc_id"] == "is1"
    assert b["issue_register"]


def test_issue_spotter_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="iss1",
        doc_label="y.pdf",
        chunks=[("Material contracts above five hundred thousand require Seller consent.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/issue-spotter/stream",
        json={"doc_id": "iss1", "retrieval_query": "consent contracts"},
    )
    assert r.status_code == 200
    assert "sources" in r.text and "done" in r.text


def test_suggested_questions(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="sq1",
        doc_label="sq.pdf",
        chunks=[("Intellectual property is licensed non-exclusively for the Territory.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/suggested-questions",
        json={"doc_id": "sq1", "retrieval_query": "intellectual property"},
    )
    assert r.status_code == 200
    assert r.json()["suggestions"]


def test_suggested_questions_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="sqs1",
        doc_label="z.pdf",
        chunks=[("Non-compete restrictions apply for eighteen months post-closing.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/suggested-questions/stream",
        json={"doc_id": "sqs1", "retrieval_query": "non-compete"},
    )
    assert r.status_code == 200
    assert "sources" in r.text and "done" in r.text


def test_deal_thesis(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="dt1",
        doc_label="spa.pdf",
        chunks=[("Purchase Price shall be paid fifty percent at Closing and fifty percent subject to earn-out metrics.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/deal-thesis",
        json={"doc_id": "dt1", "retrieval_query": "earn-out closing consideration"},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["doc_id"] == "dt1"
    assert b["thesis"]


def test_deal_thesis_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="dts1",
        doc_label="x.pdf",
        chunks=[("Seller indemnifies Buyer for breaches of representations for twelve months.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/deal-thesis/stream",
        json={"doc_id": "dts1", "retrieval_query": "indemnity"},
    )
    assert r.status_code == 200
    assert "sources" in r.text and "done" in r.text


def test_bibliography_export(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="bib1",
        doc_label="nda.pdf",
        chunks=[("Recipient shall return or destroy all Confidential Information upon written request.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/bibliography-export",
        json={"doc_id": "bib1", "retrieval_query": "confidential return destroy"},
    )
    assert r.status_code == 200
    assert r.json()["bibliography_markdown"]


def test_bibliography_export_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="bibs1",
        doc_label="y.pdf",
        chunks=[("Non-solicitation applies to employees for twenty-four months.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/bibliography-export/stream",
        json={"doc_id": "bibs1", "retrieval_query": "non-solicitation"},
    )
    assert r.status_code == 200
    assert "sources" in r.text and "done" in r.text


def test_embeddings_farthest_pair(api_client):
    r = api_client.post(
        "/v1/embeddings/farthest-pair",
        json={
            "texts": [
                "Indemnity survives closing for environmental liabilities.",
                "The moon is made of cheese according to folklore.",
                "Closing shall occur on the third business day after conditions precedent.",
            ]
        },
    )
    assert r.status_code == 200
    j = r.json()
    assert j["index_a"] != j["index_b"]
    assert "cosine_similarity" in j
    assert j["dimension"] >= 1


def test_runtime_platform_detail(api_client):
    r = api_client.get("/v1/runtime/platform-detail")
    assert r.status_code == 200
    j = r.json()
    assert j.get("system")
    assert "python_implementation" in j


def test_runtime_agent_bootstrap(api_client):
    r = api_client.get("/v1/runtime/agent-bootstrap")
    assert r.status_code == 200
    j = r.json()
    assert j.get("api_version")
    assert "model_routing" in j and "extraction" in j["model_routing"]
    assert "preflight" in j and "platform_detail" in j and "device" in j
    assert "route_hints" in j


def test_covenant_matrix(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="cov1",
        doc_label="cov.pdf",
        chunks=[("Seller shall not solicit employees of the Target for twelve months following Closing.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/covenant-matrix",
        json={"doc_id": "cov1", "retrieval_query": "non-solicit seller"},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["doc_id"] == "cov1"
    assert b["covenant_matrix"]


def test_financial_terms_ledger(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="fin1",
        doc_label="spa.pdf",
        chunks=[("The indemnification obligations shall not exceed ten million United States dollars.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/financial-terms-ledger",
        json={"doc_id": "fin1", "retrieval_query": "indemnification cap million"},
    )
    assert r.status_code == 200
    assert r.json()["ledger"]


def test_remedies_playbook(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="rem1",
        doc_label="nda.pdf",
        chunks=[("Any dispute shall be resolved by arbitration in New York under AAA rules.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/remedies-playbook",
        json={"doc_id": "rem1", "retrieval_query": "arbitration AAA dispute"},
    )
    assert r.status_code == 200
    assert r.json()["playbook"]


def test_covenant_matrix_stream(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="covs1",
        doc_label="x.pdf",
        chunks=[("Buyer must maintain insurance naming Seller as additional insured.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/covenant-matrix/stream",
        json={"doc_id": "covs1", "retrieval_query": "insurance"},
    )
    assert r.status_code == 200
    assert "sources" in r.text and "done" in r.text


def test_conditions_precedent(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="cp1",
        doc_label="spa.pdf",
        chunks=[("Closing is conditioned on receipt of all required regulatory approvals.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/conditions-precedent",
        json={"doc_id": "cp1", "retrieval_query": "regulatory approvals closing"},
    )
    assert r.status_code == 200
    assert r.json()["conditions_register"]


def test_execution_formalities(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="ex1",
        doc_label="nda.pdf",
        chunks=[("This Agreement may be executed in counterparts, each of which shall be deemed an original.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/execution-formalities",
        json={"doc_id": "ex1", "retrieval_query": "counterparts executed"},
    )
    assert r.status_code == 200
    assert r.json()["formalities"]


def test_retrieval_expand_plan(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="rep1",
        doc_label="spa.pdf",
        chunks=[("Indemnification obligations survive until the eighteen-month anniversary of Closing.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/retrieval-expand-plan",
        json={
            "doc_id": "rep1",
            "agent_goal": "Map indemnity and survival topics for second-pass retrieval.",
            "retrieval_query": "indemnification survival closing",
        },
    )
    assert r.status_code == 200
    assert r.json()["expand_plan"]


def test_document_centroid_similarity(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="dc_a",
        doc_label="a.pdf",
        chunks=[
            ("The Purchase Price shall be paid in cash at Closing.", {"page_start": 1, "page_end": 1}),
            ("Seller represents authority to enter this Agreement.", {"page_start": 2, "page_end": 2}),
        ],
    )
    store.upsert_document_chunks(
        doc_id="dc_b",
        doc_label="b.pdf",
        chunks=[
            ("Consideration includes a holdback of ten percent for indemnity claims.", {"page_start": 1, "page_end": 1}),
        ],
    )
    r = api_client.post(
        "/v1/embeddings/document-centroid-similarity",
        json={"doc_id_a": "dc_a", "doc_id_b": "dc_b", "max_chunks_per_document": 8},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["chunks_used_a"] >= 1 and j["chunks_used_b"] >= 1
    assert -1.0 <= j["cosine_between_centroids"] <= 1.0


def test_survival_schedule(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="sv1",
        doc_label="spa.pdf",
        chunks=[("The representations and warranties shall survive until the twelve-month anniversary of Closing.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/survival-schedule",
        json={"doc_id": "sv1", "retrieval_query": "representations warranties survive twelve"},
    )
    assert r.status_code == 200
    assert r.json()["survival_schedule"]


def test_assignment_coc(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="ac1",
        doc_label="spa.pdf",
        chunks=[("Neither party may assign this Agreement without the prior written consent of the other party.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/assignment-coc",
        json={"doc_id": "ac1", "retrieval_query": "assign consent"},
    )
    assert r.status_code == 200
    assert r.json()["assignment_map"]


def test_ip_assets_sweep(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="ip1",
        doc_label="license.pdf",
        chunks=[("Seller grants Buyer a non-exclusive license to use the Licensed Software solely for internal business purposes.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/ip-assets-sweep",
        json={"doc_id": "ip1", "retrieval_query": "software license exclusive"},
    )
    assert r.status_code == 200
    assert r.json()["ip_register"]


def test_document_chunk_stats(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="st1",
        doc_label="stats.pdf",
        chunks=[
            ("Short.", {"page_start": 1, "page_end": 1}),
            ("A somewhat longer chunk for statistics.", {"page_start": 2, "page_end": 2}),
        ],
    )
    r = api_client.post(
        "/v1/embeddings/document-chunk-stats",
        json={"doc_id": "st1", "max_chunks_scanned": 64},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["chunk_count_scanned"] == 2
    assert j["nonempty_chunk_count"] == 2
    assert j["total_characters_nonempty"] > 10
    assert j["truncated_scan"] is False


def test_runtime_optional_imports(api_client):
    r = api_client.get("/v1/runtime/optional-imports")
    assert r.status_code == 200
    j = r.json()
    assert "modules" in j
    assert "qdrant_client" in j["modules"]


def test_document_lexical_jaccard(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="lj_a",
        doc_label="a.pdf",
        chunks=[
            ("Indemnification covenant survives eighteen months post-closing.", {"page_start": 1, "page_end": 1}),
        ],
    )
    store.upsert_document_chunks(
        doc_id="lj_b",
        doc_label="b.pdf",
        chunks=[
            ("Indemnification obligations survive for eighteen months following the Closing.", {"page_start": 1, "page_end": 1}),
        ],
    )
    r = api_client.post(
        "/v1/embeddings/document-lexical-jaccard",
        json={"doc_id_a": "lj_a", "doc_id_b": "lj_b", "max_chunks_per_document": 16},
    )
    assert r.status_code == 200
    j = r.json()
    assert 0.0 <= j["jaccard_similarity"] <= 1.0
    assert j["union_token_count"] >= 1


def test_post_closing_covenants(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="pc1",
        doc_label="tsa.pdf",
        chunks=[("Seller shall provide reasonable transition assistance for sixty days following Closing.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/post-closing-covenants",
        json={"doc_id": "pc1", "retrieval_query": "transition assistance closing"},
    )
    assert r.status_code == 200
    assert r.json()["post_closing"]


def test_earn_out_mechanics(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="eo1",
        doc_label="spa.pdf",
        chunks=[("The Earn-Out Payment shall be calculated based on Adjusted EBITDA for the two fiscal years post-Closing.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/earn-out-mechanics",
        json={"doc_id": "eo1", "retrieval_query": "earn-out EBITDA fiscal"},
    )
    assert r.status_code == 200
    assert r.json()["earn_out"]


def test_representations_buckets(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="rb1",
        doc_label="spa.pdf",
        chunks=[("Seller represents that, to Seller's Knowledge, there is no pending litigation material to the Business.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/representations-buckets",
        json={"doc_id": "rb1", "retrieval_query": "knowledge litigation material"},
    )
    assert r.status_code == 200
    assert r.json()["reps_buckets"]


def test_runtime_process_memory(api_client):
    r = api_client.get("/v1/runtime/process-memory")
    assert r.status_code == 200
    j = r.json()
    assert "pid" in j


def test_document_token_difference(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="td_a",
        doc_label="a.pdf",
        chunks=[("Arbitration shall be seated in Delaware.", {"page_start": 1, "page_end": 1})],
    )
    store.upsert_document_chunks(
        doc_id="td_b",
        doc_label="b.pdf",
        chunks=[("Forum selection specifies California courts exclusively.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/embeddings/document-token-difference",
        json={
            "doc_id_a": "td_a",
            "doc_id_b": "td_b",
            "max_chunks_per_document": 16,
            "max_tokens_per_side": 200,
        },
    )
    assert r.status_code == 200
    j = r.json()
    assert j["total_only_in_a"] >= 1 or j["total_only_in_b"] >= 1


def test_tax_withholding(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="tw1",
        doc_label="spa.pdf",
        chunks=[("Buyer shall cooperate with reasonable efforts to obtain FIRPTA withholding certificates prior to Closing.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/tax-withholding",
        json={"doc_id": "tw1", "retrieval_query": "FIRPTA withholding"},
    )
    assert r.status_code == 200
    assert r.json()["tax_register"]


def test_insurance_requirements(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="ins1",
        doc_label="spa.pdf",
        chunks=[("Seller shall maintain directors and officers liability insurance through the tail period.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/insurance-requirements",
        json={"doc_id": "ins1", "retrieval_query": "D&O tail insurance"},
    )
    assert r.status_code == 200
    assert r.json()["insurance_register"]


def test_sanctions_export_compliance(api_client):
    from legal_intel.rag.store import LegalVectorStore

    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="se1",
        doc_label="spa.pdf",
        chunks=[("Each party represents compliance with applicable anti-corruption laws including the U.S. Foreign Corrupt Practices Act.", {"page_start": 1, "page_end": 1})],
    )
    r = api_client.post(
        "/v1/rag/sanctions-export-compliance",
        json={"doc_id": "se1", "retrieval_query": "anti-corruption FCPA"},
    )
    assert r.status_code == 200
    assert r.json()["compliance_register"]
