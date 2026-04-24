from __future__ import annotations
import difflib
import re
from dataclasses import dataclass, field
from typing import Any
from legal_intel.config import get_settings
from legal_intel.india.schemas import InstrumentFact

def _norm_name(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _fuzzy_match(a: str, b: str, threshold: float) -> bool:
    if not a or not b:
        return False
    na, nb = _norm_name(a), _norm_name(b)
    if na == nb:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= threshold

def _year_from_fact(f: InstrumentFact) -> int | None:
    for d in (f.execution_date, f.registration_date):
        if not d:
            continue
        m = re.search(r"(19|20)\d{2}", str(d))
        if m:
            try:
                return int(m.group(0))
            except ValueError:
                pass
    return None

@dataclass
class TitleGraph:
    nodes: dict[str, InstrumentFact] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)

    def add_fact(self, fact: InstrumentFact) -> None:
        self.nodes[fact.doc_id] = fact

    def link_transfer_chain(self) -> None:
        self.edges.clear()
        s = get_settings()
        thr = s.titlegraph_name_fuzzy_threshold
        docs = list(self.nodes.values())
        for i in range(len(docs)):
            for j in range(len(docs)):
                if i == j:
                    continue
                a, b = docs[i], docs[j]
                for bn in a.buyer_names:
                    for sn in b.seller_names:
                        if _fuzzy_match(bn, sn, thr):
                            self.edges.append((a.doc_id, b.doc_id, "possible_transfer_chain"))
                            break

    def detect_breaks(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        linked: set[str] = set()
        for e in self.edges:
            linked.add(e[0])
            linked.add(e[1])
        for did, fact in self.nodes.items():
            if did not in linked and len(self.nodes) > 1:
                issues.append({"kind": "orphan_or_unlinked", "doc_id": did, "detail": "No inferred chain edge; verify missing prior deed or name mismatch."})
        years: list[tuple[str, int]] = []
        for did, fact in self.nodes.items():
            y = _year_from_fact(fact)
            if y is not None:
                years.append((did, y))
        years.sort(key=lambda x: x[1])
        for k in range(1, len(years)):
            if years[k][1] - years[k - 1][1] > 50:
                issues.append({"kind": "large_year_gap", "doc_ids": [years[k - 1][0], years[k][0]], "detail": f"Years {years[k-1][1]} → {years[k][1]} — check for missing instruments."})
        parcels = {did: set(p.lower() for p in f.parcel_ids) for did, f in self.nodes.items()}
        for e in self.edges:
            p0, p1 = parcels.get(e[0], set()), parcels.get(e[1], set())
            if p0 and p1 and not (p0 & p1):
                issues.append({"kind": "parcel_id_drift", "doc_ids": [e[0], e[1]], "detail": "Linked chain but parcel identifiers do not overlap; verify survey/khata updates."})
        return issues

    def to_json(self) -> dict[str, Any]:
        return {
            "nodes": {k: v.model_dump(mode="json") for k, v in self.nodes.items()},
            "edges": [{"from": a, "to": b, "relation": r} for a, b, r in self.edges],
            "breaks": self.detect_breaks(),
        }
