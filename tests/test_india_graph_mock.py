"""Test India graph end-to-end with mock LLM."""
import os
os.environ["LEGAL_INTEL_MOCK_LLM"] = "1"
os.environ["QDRANT_URL"] = ":memory:"
os.environ["DILIGENCE_DOMAIN"] = "india_re"

from legal_intel.graph.build import run_diligence_india
from legal_intel.rag.store import LegalVectorStore

def test_india_graph_runs():
    store = LegalVectorStore()
    doc_id = "test_deed"
    store.upsert_document_chunks(
        doc_id=doc_id,
        doc_label="sale_deed_2020.pdf",
        chunks=[
            ("This sale deed executed by seller Ramesh Kumar in favour of buyer Suresh Rao for plot survey number 123/A in Kukatpally village.", {"page_start": 1, "page_end": 1}),
            ("Consideration amount Rs 50,00,000. Encumbrance Certificate shows no prior charge.", {"page_start": 2, "page_end": 2}),
        ],
    )
    result = run_diligence_india(
        "Analyze title chain and risks.",
        doc_ids=[doc_id],
        doc_labels={doc_id: "sale_deed_2020.pdf"},
    )
    assert "final_report" in result
    assert "instrument_facts_json" in result
    assert "title_graph_json" in result
