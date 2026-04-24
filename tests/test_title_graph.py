import os
os.environ["LEGAL_INTEL_MOCK_LLM"] = "1"
os.environ["QDRANT_URL"] = ":memory:"

from legal_intel.india.schemas import InstrumentFact
from legal_intel.india.title_graph import TitleGraph

def test_link_and_break():
    tg = TitleGraph()
    f1 = InstrumentFact(
        doc_id="deed1", doc_type="sale_deed",
        seller_names=["Alice"], buyer_names=["Bob"],
        parcel_ids=["SY123"], execution_date="2020-01-01",
    )
    f2 = InstrumentFact(
        doc_id="deed2", doc_type="sale_deed",
        seller_names=["Bob"], buyer_names=["Carol"],
        parcel_ids=["SY123"], execution_date="2022-06-15",
    )
    f3 = InstrumentFact(
        doc_id="deed3", doc_type="gift_deed",
        seller_names=["Xavier"], buyer_names=["Yolanda"],
        parcel_ids=["SY456"], execution_date="2023-01-01",
    )
    tg.add_fact(f1)
    tg.add_fact(f2)
    tg.add_fact(f3)
    tg.link_transfer_chain()
    
    # deed1 -> deed2 should link (Bob)
    assert any(e[0] == "deed1" and e[1] == "deed2" for e in tg.edges)
    
    # deed3 should be orphaned
    breaks = tg.detect_breaks()
    orphans = [b for b in breaks if b["kind"] == "orphan_or_unlinked"]
    assert any(b["doc_id"] == "deed3" for b in orphans)

def test_parcel_drift():
    tg = TitleGraph()
    f1 = InstrumentFact(
        doc_id="d1", seller_names=["A"], buyer_names=["B"],
        parcel_ids=["SY100"], execution_date="2015",
    )
    f2 = InstrumentFact(
        doc_id="d2", seller_names=["B"], buyer_names=["C"],
        parcel_ids=["SY999"], execution_date="2018",
    )
    tg.add_fact(f1)
    tg.add_fact(f2)
    tg.link_transfer_chain()
    breaks = tg.detect_breaks()
    drift = [b for b in breaks if b["kind"] == "parcel_id_drift"]
    assert len(drift) > 0
