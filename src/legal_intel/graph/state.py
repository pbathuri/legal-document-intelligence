from __future__ import annotations
from typing import TypedDict


class DiligenceState(TypedDict, total=False):
    user_query: str
    retrieved_context: str
    obligation_section: str
    risk_section: str
    cross_ref_section: str
    compliance_section: str
    final_report: str


class IndiaDiligenceState(TypedDict, total=False):
    user_query: str
    doc_ids: list[str]
    doc_labels: dict[str, str]
    retrieved_context: str
    instrument_facts_json: str
    title_graph_json: str
    chain_section: str
    encumbrance_section: str
    records_section: str
    dispute_check_results: str  # NEW: agent tool results
    final_report: str
