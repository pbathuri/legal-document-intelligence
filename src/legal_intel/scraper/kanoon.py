"""Indian Kanoon API client — judgments and orders (property / dispute corpus).

Uses the official JSON API with a user token. Without a token, returns synthetic
records so the training pipeline stays testable.

API docs: https://api.indiankanoon.org/documentation/
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import httpx

from legal_intel.scraper.base import BaseScraper

logger = logging.getLogger(__name__)

API_BASE = "https://api.indiankanoon.org"


def _strip_html(html: str) -> str:
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_PROP_HINT = re.compile(
    r"property|possession|title|survey|mortgage|encumbrance|partition|deed|immovable",
    re.I,
)


def _annotate_judgment_record(rec: dict[str, Any]) -> dict[str, Any]:
    out = dict(rec)
    body = out.get("full_text_plain") or out.get("headline") or ""
    out["doc_type"] = "court_judgment"
    out["document_type"] = "court_judgment"
    out["property_related"] = bool(_PROP_HINT.search(body))
    return out


class IndianKanoonScraper(BaseScraper):
    """Search Indian Kanoon; optionally fetch full judgment text per result."""

    source_name = "kanoon"

    def scrape(
        self,
        *,
        query: str = "property dispute possession",
        pagenum: int = 0,
        doctypes: str | None = None,
        max_results: int | None = None,
        fetch_full_text: bool = True,
        persist_to_disk: bool = True,
    ) -> list[dict[str, Any]]:
        max_results = max_results or self.settings.scraper_max_pages
        max_results = min(max_results, 50)
        token = (self.settings.indian_kanoon_api_token or "").strip()

        cache_key = f"kanoon_{query}_{pagenum}_{doctypes}_{max_results}_{fetch_full_text}"
        cache_id = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        if persist_to_disk:
            cached = self._load_cached(cache_id)
            if cached and "records" in cached:
                return [_annotate_judgment_record(r) for r in cached["records"]]

        if not token:
            logger.warning(
                "INDIAN_KANOON_API_TOKEN not set — using synthetic Kanoon-like records. "
                "Get a token at https://api.indiankanoon.org/ — use `export INDIAN_KANOON_API_TOKEN=...` "
                "(shell assignment alone does not pass to subprocesses) or add it to project `.env`."
            )
            records = [
                _annotate_judgment_record(r)
                for r in self._synthetic_judgments(query, max_results)
            ]
            if persist_to_disk:
                self._save_result(
                    cache_id, {"query": cache_key, "records": records})
            return records

        records: list[dict[str, Any]] = []
        try:
            params: dict[str, str | int] = {
                "formInput": query, "pagenum": pagenum}
            if doctypes:
                params["doctypes"] = doctypes
            data = self._api_search(params)
            if not data:
                raise RuntimeError("empty search response")
            docs = data.get("docs") or []
            cap_full = min(
                max_results,
                len(docs),
                self.settings.kanoon_max_full_documents if fetch_full_text else 0,
            )
            for i, doc in enumerate(docs[:max_results]):
                tid = doc.get("tid")
                if tid is None:
                    continue
                rec: dict[str, Any] = {
                    "tid": int(tid),
                    "title": doc.get("title") or "",
                    "headline": doc.get("headline") or "",
                    "docsource": doc.get("docsource") or "",
                    "publishdate": doc.get("publishdate"),
                    "docsize": doc.get("docsize"),
                    "query": query,
                    "is_synthetic": False,
                }
                if fetch_full_text and i < cap_full:
                    full = self._fetch_document_json(int(tid))
                    if full:
                        raw_html = full.get("doc") or ""
                        rec["full_text_plain"] = _strip_html(
                            raw_html) if raw_html else ""
                        rec["cite"] = full.get("cite")
                records.append(_annotate_judgment_record(rec))
        except Exception as e:
            logger.warning("Indian Kanoon API failed: %s", e)
            records = [
                _annotate_judgment_record(r)
                for r in self._synthetic_judgments(query, max_results)
            ]

        if persist_to_disk:
            self._save_result(
                cache_id, {"query": cache_key, "records": records})
        return records

    def _api_search(self, params: dict[str, str | int]) -> dict[str, Any] | None:
        """POST form data to /search/ — the live API returns 405 on GET (see IK AJAX docs)."""
        token = self.settings.indian_kanoon_api_token.strip()
        self._rate_limit()
        url = f"{API_BASE}/search/"
        headers = {
            "Authorization": f"Token {token}",
            "Accept": "application/json",
        }
        timeout = float(self.settings.agent_tool_timeout)
        form = {k: str(v) for k, v in params.items()}
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, data=form, headers=headers)
            if r.status_code == 403:
                logger.error(
                    "Indian Kanoon returned 403 — check INDIAN_KANOON_API_TOKEN")
                return None
            r.raise_for_status()
            return r.json()

    def _fetch_document_json(self, tid: int) -> dict[str, Any] | None:
        self._rate_limit()
        token = self.settings.indian_kanoon_api_token.strip()
        url = f"{API_BASE}/doc/{tid}/"
        headers = {
            "Authorization": f"Token {token}",
            "Accept": "application/json",
        }
        timeout = float(self.settings.agent_tool_timeout)
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.get(url, headers=headers)
                if r.status_code != 200:
                    return None
                return r.json()
        except Exception as e:
            logger.debug("doc fetch failed for tid=%s: %s", tid, e)
            return None

    def _synthetic_judgments(self, query: str, count: int) -> list[dict[str, Any]]:
        import random

        titles = [
            "Suit for declaration of title and recovery of possession",
            "Appeal against decree in partition suit",
            "Writ challenging mutation entry",
            "Second appeal — boundary dispute",
        ]
        out = []
        for i in range(min(count, 10)):
            text = (
                f"The appellant claims title to the schedule property based on "
                f"registered sale deed. The respondent disputes possession. "
                f"Survey reference: {random.randint(1, 500)}/{random.choice(['A', 'B'])}. "
                f"Court considered encumbrance and prior litigation."
            )
            out.append(
                {
                    "tid": 900000 + i,
                    "title": random.choice(titles),
                    "headline": text[:200] + "…",
                    "docsource": random.choice(
                        ["Supreme Court of India", "High Court of Karnataka",
                            "High Court of Telangana"]
                    ),
                    "publishdate": f"{random.randint(2018, 2025)}-{random.randint(1, 12):02d}-01",
                    "full_text_plain": text,
                    "query": query,
                    "is_synthetic": True,
                }
            )
        return out

    def to_training_record(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        body = raw.get("full_text_plain") or raw.get("headline") or ""
        if not body.strip():
            return None
        prop_hit = bool(
            re.search(
                r"property|possession|title|survey|mortgage|encumbrance|partition|deed",
                body,
                re.I,
            )
        )
        return {
            "instruction": (
                "Classify whether this Indian court judgment excerpt primarily concerns "
                "immovable property / title / possession / encumbrance. Reply as JSON with "
                "keys: property_related (boolean), risk_summary (one sentence)."
            ),
            "input": f"Title: {raw.get('title', '')}\nSource: {raw.get('docsource', '')}\n\n{body[:8000]}",
            "output": {
                "property_related": prop_hit,
                "risk_summary": (
                    "Discusses property, title, or possession — verify facts against records."
                    if prop_hit
                    else "Not clearly property-centric from excerpt; review full judgment."
                ),
            },
            "source": "indiankanoon_api" if not raw.get("is_synthetic") else "indiankanoon_synthetic",
            "language": "en",
            "doc_type": "court_judgment",
            "is_synthetic": raw.get("is_synthetic", False),
        }
