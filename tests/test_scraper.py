"""Test scraper generates synthetic data correctly."""

import json

import legal_intel.scraper.base as scraper_base
from legal_intel.config import Settings, get_settings
from legal_intel.scraper.igrs import IGRSScraper
from legal_intel.scraper.ecourts import ECourtsScraper
from legal_intel.scraper.kaveri import KaveriScraper
from legal_intel.scraper.kanoon import IndianKanoonScraper
from legal_intel.scraper.bulk_kanoon import _backfill_missing_full_text, load_queries_from_file
from legal_intel.scraper.normalize import dedupe_records, normalize_land_record


def test_igrs_synthetic():
    scraper = IGRSScraper(state="telangana")
    records = scraper._generate_sample_data("Hyderabad", None, 2024, "sale_deed", 5)
    assert len(records) == 5
    for r in records:
        assert "seller_names" in r
        assert "buyer_names" in r
        assert r["is_synthetic"] is True


def test_igrs_to_training():
    scraper = IGRSScraper()
    records = scraper._generate_sample_data("Hyderabad", None, 2024, "sale_deed", 3)
    for r in records:
        tr = scraper.to_training_record(r)
        assert tr is not None
        assert "instruction" in tr
        assert "input" in tr
        assert "output" in tr


def test_ecourts_synthetic():
    scraper = ECourtsScraper()
    records = scraper._generate_sample_disputes("Test Party", 3)
    assert len(records) == 3
    for r in records:
        assert "case_number" in r


def test_kaveri_synthetic():
    scraper = KaveriScraper()
    records = scraper._generate_sample_ec("Bangalore", "123/A", 2015, 2024, 5)
    assert len(records) == 5


def test_kanoon_synthetic_without_token(tmp_path, monkeypatch):
    fake = Settings(
        indian_kanoon_api_token="",
        scraper_data_dir=str(tmp_path / "raw"),
        scraper_rate_limit=0.0,
    )
    monkeypatch.setattr(scraper_base, "get_settings", lambda: fake)
    get_settings.cache_clear()
    scraper = IndianKanoonScraper()
    recs = scraper.scrape(max_results=3, query="easement")
    assert len(recs) == 3
    assert all(r.get("doc_type") == "court_judgment" for r in recs)
    assert all(r.get("is_synthetic") for r in recs)
    tr = scraper.to_training_record(recs[0])
    assert tr is not None
    assert tr["doc_type"] == "court_judgment"
    assert "property_related" in tr["output"]


def test_kanoon_mock_api(tmp_path, monkeypatch):
    class MockResponse:
        def __init__(self, data, status=200):
            self._data = data
            self.status_code = status

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError("http error")

        def json(self):
            return self._data

    class MockClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, **kwargs):
            if "/search/" in url:
                return MockResponse(
                    {
                        "docs": [
                            {
                                "tid": 42,
                                "title": "Partition suit",
                                "headline": "property boundary",
                                "docsource": "High Court",
                            }
                        ]
                    }
                )
            return MockResponse({})

        def get(self, url, **kwargs):
            if "/doc/42/" in url:
                return MockResponse(
                    {"doc": "<p>Mortgage and possession dispute</p>", "cite": "AIR 1"}
                )
            return MockResponse({})

    fake = Settings(
        indian_kanoon_api_token="secret",
        scraper_data_dir=str(tmp_path / "raw"),
        scraper_rate_limit=0.0,
        kanoon_max_full_documents=5,
    )
    monkeypatch.setattr(scraper_base, "get_settings", lambda: fake)
    monkeypatch.setattr("legal_intel.scraper.kanoon.httpx.Client", MockClient)
    get_settings.cache_clear()
    scraper = IndianKanoonScraper()
    recs = scraper.scrape(max_results=1, query="q", fetch_full_text=True)
    assert len(recs) == 1
    assert recs[0]["tid"] == 42
    assert recs[0]["is_synthetic"] is False
    assert "possession" in (recs[0].get("full_text_plain") or "").lower()
    assert recs[0].get("property_related") is True


def test_kanoon_backfill_full_text(tmp_path, monkeypatch):
    def fake_fetch(self, tid: int):
        if tid == 99:
            return {"doc": "<p>Full judgment about possession.</p>", "cite": "X"}
        return None

    fake = Settings(
        indian_kanoon_api_token="secret",
        scraper_data_dir=str(tmp_path / "raw"),
        scraper_rate_limit=0.0,
    )
    monkeypatch.setattr(scraper_base, "get_settings", lambda: fake)
    monkeypatch.setattr(IndianKanoonScraper, "_fetch_document_json", fake_fetch)
    get_settings.cache_clear()
    scraper = IndianKanoonScraper()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "tid": 99,
                "title": "T",
                "headline": "property dispute",
                "is_synthetic": False,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    n = _backfill_missing_full_text(scraper, corpus, ft_budget=5, log_every=0)
    assert n == 1
    row = json.loads(corpus.read_text(encoding="utf-8").strip())
    assert "possession" in (row.get("full_text_plain") or "").lower()
    assert row.get("cite") == "X"


def test_load_queries_from_file(tmp_path):
    p = tmp_path / "q.txt"
    p.write_text(
        "# intro\nproperty dispute\n\npossession suit\n",
        encoding="utf-8",
    )
    assert load_queries_from_file(p) == ["property dispute", "possession suit"]


def test_normalize_dedupe():
    rows = [
        {"tid": 1, "x": "a"},
        {"tid": 1, "x": "b"},
        {"case_number": "C1"},
        {"case_number": "C1"},
    ]
    d = dedupe_records([normalize_land_record(r, source="t") for r in rows])
    assert len(d) == 2
