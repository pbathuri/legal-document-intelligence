"""Property Diligence Copilot — Streamlit UI.

Dual domain: India property (default) or M&A contracts.
"""
from __future__ import annotations
from legal_intel.pipeline import ingest_pdf
from legal_intel.graph.build import run_diligence, run_diligence_india
from legal_intel.config import get_settings

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Property Diligence Copilot",
                   page_icon="📜", layout="wide")

# ── Environment defaults ──
os.environ.setdefault("LEGAL_INTEL_MOCK_LLM", "1")
os.environ.setdefault("QDRANT_URL", ":memory:")


def _init_state():
    for key in ("india_doc_ids", "india_doc_labels", "indexed_files", "run_results"):
        if key not in st.session_state:
            st.session_state[key] = [] if key != "india_doc_labels" else {}
    if "run_results" not in st.session_state:
        st.session_state["run_results"] = None


_init_state()
s = get_settings()

# ── Sidebar ──
with st.sidebar:
    st.title("⚙️ Configuration")
    domain = st.radio(
        "Diligence domain",
        ["india_re", "mna"],
        index=0 if s.diligence_domain == "india_re" else 1,
        format_func=lambda x: "🇮🇳 India Property" if x == "india_re" else "🏢 M&A Contracts",
    )

    st.divider()
    st.caption(f"LLM: `{s.llm_model}`")
    st.caption(f"Mock mode: `{s.legal_intel_mock_llm}`")
    st.caption(f"Qdrant: `{s.qdrant_url}`")

    if st.button("🗑️ Clear session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Main ──
st.title("📜 Property Diligence Copilot")

if domain == "india_re":
    st.error(
        "**⚠️ IMPORTANT DISCLAIMER** — India operates a presumptive land-titling environment. "
        "Registration records a transaction; it does NOT guarantee title. "
        "This tool provides **assistive due diligence analysis**, not legal advice. "
        "Outputs must be verified by qualified legal counsel before reliance.",
        icon="⚖️",
    )

# ── File upload ──
st.subheader("1️⃣ Upload documents")
uploaded = st.file_uploader(
    "Upload PDF files (deeds, EC, mutation, contracts…)",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded:
    for f in uploaded:
        if f.name not in st.session_state.get("indexed_files", []):
            tmp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = tmp.name
                with st.spinner(f"Indexing {f.name}…"):
                    doc_id, n_chunks = ingest_pdf(tmp_path, doc_label=f.name)
                st.session_state.setdefault("indexed_files", []).append(f.name)
                st.session_state.setdefault("india_doc_ids", []).append(doc_id)
                st.session_state.setdefault("india_doc_labels", {})[
                    doc_id] = f.name
                st.success(
                    f"Indexed **{f.name}** -> `{doc_id}` ({n_chunks} chunks)")
            finally:
                if tmp_path:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except OSError:
                        pass


indexed = st.session_state.get("indexed_files", [])
if indexed:
    st.info(f"**{len(indexed)} document(s) indexed**: {', '.join(indexed)}")

# ── Query ──
st.subheader("2️⃣ Run diligence")
query = st.text_area(
    "Diligence question",
    value="Analyze the title chain, flag encumbrance or dispute risks, and identify gaps." if domain == "india_re"
    else "Review obligations, risks, and cross-document consistency.",
    height=80,
)

if st.button("🚀 Run analysis", type="primary", disabled=not indexed):
    with st.spinner("Running diligence pipeline…"):
        if domain == "india_re":
            result = run_diligence_india(
                query,
                doc_ids=st.session_state.get("india_doc_ids", []),
                doc_labels=st.session_state.get("india_doc_labels", {}),
            )
        else:
            result = run_diligence(query)
    st.session_state["run_results"] = dict(result)

# ── Results ──
results = st.session_state.get("run_results")
if results:
    st.subheader("3️⃣ Results")

    # Final report
    report = results.get("final_report", "")
    if report:
        st.markdown("### 📋 Diligence Memo")
        st.markdown(report)

    # Expandable details
    if domain == "india_re":
        with st.expander("🔗 Title Graph (JSON)"):
            tg = results.get("title_graph_json", "{}")
            if isinstance(tg, str):
                try:
                    tg = json.loads(tg)
                except json.JSONDecodeError:
                    pass
            st.json(tg)

        with st.expander("📄 Instrument Facts (JSON)"):
            facts = results.get("instrument_facts_json", "[]")
            if isinstance(facts, str):
                try:
                    facts = json.loads(facts)
                except json.JSONDecodeError:
                    pass
            st.json(facts)

        with st.expander("🔍 Dispute Check Results"):
            dc = results.get("dispute_check_results", "[]")
            st.code(dc if isinstance(dc, str) else json.dumps(dc, indent=2))

        cols = st.columns(3)
        sections = [
            ("Chain / Continuity", "chain_section"),
            ("Encumbrance / Dispute", "encumbrance_section"),
            ("Records Context", "records_section"),
        ]
        for col, (label, key) in zip(cols, sections):
            with col:
                with st.expander(label):
                    st.markdown(results.get(key, "_No data_"))
    else:
        sections = [
            ("Obligations", "obligation_section"),
            ("Risks", "risk_section"),
            ("Cross-document", "cross_ref_section"),
            ("Compliance", "compliance_section"),
        ]
        for label, key in sections:
            with st.expander(label):
                st.markdown(results.get(key, "_No data_"))

    with st.expander("📦 Retrieved Context"):
        st.text(results.get("retrieved_context", "")[:5000])

    # Download full JSON
    st.download_button(
        "⬇️ Download full JSON",
        data=json.dumps(results, indent=2, default=str),
        file_name="diligence_output.json",
        mime="application/json",
    )
