"""Test M&A graph end-to-end with mock LLM."""

import os

os.environ["LEGAL_INTEL_MOCK_LLM"] = "1"
os.environ["QDRANT_URL"] = ":memory:"
os.environ["DILIGENCE_DOMAIN"] = "mna"

from legal_intel.graph.build import run_diligence
from legal_intel.rag.store import LegalVectorStore


def test_mna_graph_runs():
    store = LegalVectorStore()
    store.upsert_document_chunks(
        doc_id="test_contract",
        doc_label="test_contract.pdf",
        chunks=[
            (
                "The seller agrees to transfer all assets by December 31.",
                {"page_start": 1, "page_end": 1},
            ),
            ("Indemnification cap is limited to $5M.", {"page_start": 2, "page_end": 2}),
        ],
    )
    result = run_diligence("Review obligations and risks in this contract.")
    assert "final_report" in result
    assert len(result["final_report"]) > 0
