from __future__ import annotations

import json
import re

from legal_intel.config import get_settings
from legal_intel.india.schemas import Evidence, InstrumentFact
from legal_intel.llm.client import chat_complete, chat_complete_json

EXTRACTION_SYSTEM = """You extract structured facts from Indian real-estate / registration documents.
Return a single JSON object ONLY with keys:
doc_id, doc_label, doc_type, execution_date, registration_date, seller_names (array of strings),
buyer_names (array), parcel_ids (array), locality, consideration_amount (string or null),
mentions_dispute (boolean), dispute_details, mentions_encumbrance (boolean), encumbrance_details,
evidence (array of {page: number|null, quote: string}).
Use null where unknown. seller_names and buyer_names may be empty if not found.
Include 1–3 short evidence quotes with page numbers when visible in the document text.
Do not invent government IDs or case numbers not shown in the document text."""


def _mock_instrument_fact(doc_id: str, doc_label: str, context_text: str) -> InstrumentFact:
    sellers: list[str] = []
    buyers: list[str] = []
    if re.search(r"seller|vendor|executant", context_text, re.I):
        sellers = ["Extracted Seller (mock)"]
    if re.search(r"buyer|purchaser|vendee", context_text, re.I):
        buyers = ["Extracted Buyer (mock)"]
    if not sellers and not buyers:
        buyers = ["Unknown party (mock)"]
    parcels = re.findall(
        r"(?:survey|plot|khata|patta|ULPIN|CTS)[\s:No.]*[\w\-/]+", context_text, re.I
    )
    ev: list[Evidence] = []
    if context_text.strip():
        ev.append(Evidence(page=1, quote=context_text[:200].strip()))
    return InstrumentFact(
        doc_id=doc_id,
        doc_label=doc_label,
        doc_type="sale_deed" if "sale" in context_text.lower() else "unknown",
        seller_names=sellers,
        buyer_names=buyers,
        parcel_ids=parcels[:5] or [],
        evidence=ev,
    )


def _parse_json_loose(raw: str) -> dict:
    raw = raw.strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        raw = m.group(0)
    return json.loads(raw)


def extract_instrument_fact(doc_id: str, doc_label: str, context_text: str) -> InstrumentFact:
    s = get_settings()
    max_chars = max(4000, s.extraction_max_pages * 3500)
    context_text = context_text[:max_chars]
    if s.legal_intel_mock_llm:
        return _mock_instrument_fact(doc_id, doc_label, context_text)
    user = f"doc_id: {doc_id}\ndoc_label: {doc_label}\n\nDOCUMENT TEXT / EXCERPTS:\n{context_text[:50000]}"

    if s.llm_json_mode_extraction:
        try:
            raw = chat_complete_json(
                EXTRACTION_SYSTEM,
                user,
                temperature=0.0,
                max_tokens=2048,
                task="extraction",
            )
            data = _parse_json_loose(raw)
            data["doc_id"] = doc_id
            data["doc_label"] = doc_label
            return InstrumentFact.model_validate(data)
        except Exception:
            pass

    raw = chat_complete(
        EXTRACTION_SYSTEM,
        user,
        temperature=0.0,
        max_tokens=2048,
        task="extraction",
    )
    try:
        data = _parse_json_loose(raw)
        data["doc_id"] = doc_id
        data["doc_label"] = doc_label
        return InstrumentFact.model_validate(data)
    except Exception:
        raw2 = chat_complete(
            EXTRACTION_SYSTEM
            + "\nYour previous output was invalid JSON. Reply with ONLY valid JSON.",
            user + f"\n\nINVALID_OUTPUT:\n{raw[:4000]}",
            temperature=0.0,
            max_tokens=2048,
            task="extraction",
        )
        data = _parse_json_loose(raw2)
        data["doc_id"] = doc_id
        data["doc_label"] = doc_label
        return InstrumentFact.model_validate(data)
