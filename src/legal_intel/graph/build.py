from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from langgraph.graph import END, START, StateGraph

from legal_intel.config import DiligenceDomain, get_settings
from legal_intel.graph.state import DiligenceState, IndiaDiligenceState
from legal_intel.india.extraction import extract_instrument_fact
from legal_intel.india.prompts_india import (
    CHAIN_SYSTEM,
    ENCUMBRANCE_SYSTEM,
    RECORDS_SYSTEM,
    RETRIEVAL_QUERY_SUFFIX,
    SYNTHESIS_INDIA_SYSTEM,
)
from legal_intel.india.schemas import InstrumentFact
from legal_intel.india.title_graph import TitleGraph
from legal_intel.llm.client import chat_complete
from legal_intel.prompts import (
    COMPLIANCE_SYSTEM,
    CROSS_REF_SYSTEM,
    OBLIGATION_SYSTEM,
    RISK_SYSTEM,
    SYNTHESIS_SYSTEM,
    format_context_block,
)
from legal_intel.rag.store import LegalVectorStore


# ===== M&A graph nodes =====


def _retrieve(state: DiligenceState) -> DiligenceState:
    s = get_settings()
    store = LegalVectorStore()
    q = state.get("user_query") or ""
    combined_query = f"{q}\n\nExtract obligations, risks, cross-document inconsistencies, regulatory hooks from contracts."
    hits = store.search(combined_query, limit=s.retrieval_top_k)
    ctx = format_context_block(hits)
    return {"retrieved_context": ctx}


def _obligation(state: DiligenceState) -> DiligenceState:
    ctx = state.get("retrieved_context", "")
    uq = state.get("user_query", "")
    user = f"USER FOCUS:\n{uq}\n\nCONTEXT EXCERPTS:\n{ctx}"
    out = chat_complete(OBLIGATION_SYSTEM, user, temperature=0.05, task="specialist")
    return {"obligation_section": out}


def _risk(state: DiligenceState) -> DiligenceState:
    ctx = state.get("retrieved_context", "")
    uq = state.get("user_query", "")
    user = f"USER FOCUS:\n{uq}\n\nCONTEXT EXCERPTS:\n{ctx}"
    out = chat_complete(RISK_SYSTEM, user, temperature=0.05, task="specialist")
    return {"risk_section": out}


def _cross(state: DiligenceState) -> DiligenceState:
    ctx = state.get("retrieved_context", "")
    uq = state.get("user_query", "")
    user = f"USER FOCUS:\n{uq}\n\nCONTEXT EXCERPTS:\n{ctx}"
    out = chat_complete(CROSS_REF_SYSTEM, user, temperature=0.05, task="specialist")
    return {"cross_ref_section": out}


def _compliance(state: DiligenceState) -> DiligenceState:
    ctx = state.get("retrieved_context", "")
    uq = state.get("user_query", "")
    user = f"USER FOCUS:\n{uq}\n\nCONTEXT EXCERPTS:\n{ctx}"
    out = chat_complete(COMPLIANCE_SYSTEM, user, temperature=0.05, task="specialist")
    return {"compliance_section": out}


def _synthesize(state: DiligenceState) -> DiligenceState:
    bundle = (
        f"## Obligations\n{state.get('obligation_section', '')}\n\n"
        f"## Risks\n{state.get('risk_section', '')}\n\n"
        f"## Cross-document\n{state.get('cross_ref_section', '')}\n\n"
        f"## Compliance\n{state.get('compliance_section', '')}\n"
    )
    user = (
        f"ORIGINAL USER REQUEST:\n{state.get('user_query', '')}\n\nSPECIALIST SECTIONS:\n{bundle}"
    )
    out = chat_complete(SYNTHESIS_SYSTEM, user, temperature=0.1, task="synthesis")
    return {"final_report": out}


def build_graph():
    g = StateGraph(DiligenceState)
    g.add_node("retrieve", _retrieve)
    g.add_node("obligations", _obligation)
    g.add_node("risks", _risk)
    g.add_node("cross_ref", _cross)
    g.add_node("compliance", _compliance)
    g.add_node("synthesize", _synthesize)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "obligations")
    g.add_edge("obligations", "risks")
    g.add_edge("risks", "cross_ref")
    g.add_edge("cross_ref", "compliance")
    g.add_edge("compliance", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def run_diligence(user_query: str) -> DiligenceState:
    graph = build_graph()
    return graph.invoke({"user_query": user_query})


# ===== India property diligence graph =====


def _retrieve_india(state: IndiaDiligenceState) -> IndiaDiligenceState:
    s = get_settings()
    store = LegalVectorStore()
    q = (state.get("user_query") or "") + RETRIEVAL_QUERY_SUFFIX
    hits = store.search(q, limit=s.retrieval_top_k)
    ctx = format_context_block(hits)
    return {"retrieved_context": ctx}


def _extract_facts_india(state: IndiaDiligenceState) -> IndiaDiligenceState:
    s = get_settings()
    store = LegalVectorStore()
    doc_ids = state.get("doc_ids") or []
    labels = state.get("doc_labels") or {}
    uq = state.get("user_query") or ""
    combined_q = uq + RETRIEVAL_QUERY_SUFFIX
    facts: list[InstrumentFact] = []
    per_doc = max(8, s.retrieval_top_k)
    for did in doc_ids:
        label = labels.get(did, did)
        hits = store.search(combined_q, limit=per_doc, doc_id=did)
        text = format_context_block(hits)
        if "(No retrieved context" in text or not text.strip():
            hits = store.search(combined_q, limit=per_doc)
            filt = [h for h in hits if h.get("doc_id") == did]
            text = format_context_block(filt)
        fact = extract_instrument_fact(did, label, text)
        facts.append(fact)
    return {"instrument_facts_json": json.dumps([f.model_dump(mode="json") for f in facts])}


def _build_titlegraph_node(state: IndiaDiligenceState) -> IndiaDiligenceState:
    raw = state.get("instrument_facts_json") or "[]"
    rows = json.loads(raw)
    tg = TitleGraph()
    for row in rows:
        tg.add_fact(InstrumentFact.model_validate(row))
    tg.link_transfer_chain()
    return {"title_graph_json": json.dumps(tg.to_json())}


def _dispute_check_india(state: IndiaDiligenceState) -> IndiaDiligenceState:
    """Check disputes for extracted party names (timeout-bounded; never blocks indefinitely)."""
    s = get_settings()
    raw = state.get("instrument_facts_json") or "[]"
    rows = json.loads(raw)

    def _run_checks() -> IndiaDiligenceState:
        all_parties: set[str] = set()
        for row in rows:
            all_parties.update(row.get("buyer_names", []))
            all_parties.update(row.get("seller_names", []))
        results: list[dict] = []
        for party in list(all_parties)[:3]:
            if party and "mock" not in party.lower():
                try:
                    from legal_intel.agents.tools import check_disputes

                    result = check_disputes(party_name=party)
                    results.append({"party": party, "result": result})
                except Exception as e:
                    results.append({"party": party, "error": str(e)})
        return {"dispute_check_results": json.dumps(results)}

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run_checks)
        try:
            return fut.result(timeout=s.dispute_check_timeout_seconds)
        except FutureTimeout:
            return {"dispute_check_results": json.dumps([{"error": "dispute_check_timeout"}])}


def _chain_india(state: IndiaDiligenceState) -> IndiaDiligenceState:
    ctx = state.get("retrieved_context", "")
    tg = state.get("title_graph_json", "{}")
    uq = state.get("user_query", "")
    disputes = state.get("dispute_check_results", "[]")
    user = f"USER FOCUS:\n{uq}\n\nTITLE_GRAPH_JSON:\n{tg}\n\nDISPUTE_CHECK:\n{disputes}\n\nCONTEXT EXCERPTS:\n{ctx}"
    out = chat_complete(CHAIN_SYSTEM, user, temperature=0.05, task="specialist")
    return {"chain_section": out}


def _encumbrance_india(state: IndiaDiligenceState) -> IndiaDiligenceState:
    ctx = state.get("retrieved_context", "")
    uq = state.get("user_query", "")
    user = f"USER FOCUS:\n{uq}\n\nCONTEXT EXCERPTS:\n{ctx}"
    out = chat_complete(ENCUMBRANCE_SYSTEM, user, temperature=0.05, task="specialist")
    return {"encumbrance_section": out}


def _records_india(state: IndiaDiligenceState) -> IndiaDiligenceState:
    ctx = state.get("retrieved_context", "")
    uq = state.get("user_query", "")
    user = f"USER FOCUS:\n{uq}\n\nCONTEXT EXCERPTS:\n{ctx}"
    out = chat_complete(RECORDS_SYSTEM, user, temperature=0.05, task="specialist")
    return {"records_section": out}


def _india_specialists_parallel(state: IndiaDiligenceState) -> IndiaDiligenceState:
    """Run chain, encumbrance, and records LLM calls concurrently (I/O-bound)."""
    with ThreadPoolExecutor(max_workers=3) as ex:
        fc = ex.submit(_chain_india, state)
        fe = ex.submit(_encumbrance_india, state)
        fr = ex.submit(_records_india, state)
        merged: IndiaDiligenceState = {}
        merged.update(fc.result())
        merged.update(fe.result())
        merged.update(fr.result())
    return merged


def _synthesize_india(state: IndiaDiligenceState) -> IndiaDiligenceState:
    bundle = (
        f"## Chain and continuity\n{state.get('chain_section', '')}\n\n"
        f"## Encumbrance / dispute\n{state.get('encumbrance_section', '')}\n\n"
        f"## Records / compliance context\n{state.get('records_section', '')}\n"
    )
    user = (
        f"USER REQUEST:\n{state.get('user_query', '')}\n\n"
        f"TITLE_GRAPH_JSON:\n{state.get('title_graph_json', '{}')}\n\n"
        f"DISPUTE_CHECK_RESULTS:\n{state.get('dispute_check_results', '[]')}\n\n"
        f"SPECIALIST SECTIONS:\n{bundle}"
    )
    out = chat_complete(SYNTHESIS_INDIA_SYSTEM, user, temperature=0.1, task="synthesis")
    return {"final_report": out}


def build_graph_india():
    g = StateGraph(IndiaDiligenceState)
    g.add_node("retrieve", _retrieve_india)
    g.add_node("extract_facts", _extract_facts_india)
    g.add_node("build_titlegraph", _build_titlegraph_node)
    g.add_node("dispute_check", _dispute_check_india)
    g.add_node("specialists", _india_specialists_parallel)
    g.add_node("synthesize", _synthesize_india)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "extract_facts")
    g.add_edge("extract_facts", "build_titlegraph")
    g.add_edge("build_titlegraph", "dispute_check")
    g.add_edge("dispute_check", "specialists")
    g.add_edge("specialists", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def run_diligence_india(
    user_query: str, doc_ids: list[str], doc_labels: dict[str, str] | None = None
) -> IndiaDiligenceState:
    graph = build_graph_india()
    return graph.invoke(
        {"user_query": user_query, "doc_ids": doc_ids, "doc_labels": doc_labels or {}}
    )


def run_diligence_auto(
    user_query: str, doc_ids: list[str] | None = None, doc_labels: dict[str, str] | None = None
) -> DiligenceState | IndiaDiligenceState:
    s = get_settings()
    if s.diligence_domain == "india_re":
        return run_diligence_india(user_query, doc_ids or [], doc_labels)
    return run_diligence(user_query)


def run_diligence_for_domain(
    user_query: str,
    *,
    domain: DiligenceDomain,
    doc_ids: list[str] | None = None,
    doc_labels: dict[str, str] | None = None,
) -> DiligenceState | IndiaDiligenceState:
    """Run diligence without mutating global settings (HTTP API friendly)."""
    if domain == "india_re":
        return run_diligence_india(user_query, doc_ids or [], doc_labels or {})
    return run_diligence(user_query)


def stream_diligence_for_domain(
    user_query: str,
    *,
    domain: DiligenceDomain,
    doc_ids: list[str] | None = None,
    doc_labels: dict[str, str] | None = None,
):
    """
    Stream LangGraph node completions as mapping updates (sync iterator).
    Each yielded value is ``{node_name: partial_state_update}``.
    """
    if domain == "india_re":
        graph = build_graph_india()
        initial: dict = {
            "user_query": user_query,
            "doc_ids": doc_ids or [],
            "doc_labels": doc_labels or {},
        }
    else:
        graph = build_graph()
        initial = {"user_query": user_query}
    yield from graph.stream(initial, stream_mode="updates")
