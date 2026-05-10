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
