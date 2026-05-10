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
            ("Confidential Information must not be disclosed for 3 years.", {"page_start": 1, "page_end": 1}),
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
        dr = client.delete(f"/v1/runs/{rid}")
        assert dr.status_code == 200
        gr = client.get(f"/v1/runs/{rid}")
        assert gr.status_code == 404
    get_settings.cache_clear()
