"""Scraper for Karnataka Kaveri Online (kavfrionline.karnataka.gov.in).

Kaveri is Karnataka's registration and stamps portal providing:
- Encumbrance Certificate (EC) search
- Document search by SRO/year/document number
- Property valuation guidance
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

from legal_intel.scraper.base import BaseScraper

logger = logging.getLogger(__name__)


class KaveriScraper(BaseScraper):
    """Scrape Karnataka Kaveri portal for EC and registration data."""

    source_name = "kaveri"
    BASE_URL = "https://kavfrionline.karnataka.gov.in"

    def scrape(
        self,
        *,
        property_id: str | None = None,
        district: str | None = None,
        taluk: str | None = None,
        village: str | None = None,
        survey_number: str | None = None,
        ec_year_from: int | None = None,
        ec_year_to: int | None = None,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        max_results = max_results or self.settings.scraper_max_pages
        
        cache_key = f"kaveri_{district}_{survey_number}_{ec_year_from}_{ec_year_to}"
        cache_id = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        cached = self._load_cached(cache_id)
        if cached and "records" in cached:
            return cached["records"]

        results: list[dict[str, Any]] = []
        fetcher = self._get_fetcher()
        
        if fetcher is None:
            results = self._generate_sample_ec(district, survey_number, ec_year_from, ec_year_to, max_results)
            self._save_result(cache_id, {"query": cache_key, "records": results})
            return results
        
        try:
            self._rate_limit()
            response = fetcher.get(self.BASE_URL, timeout=30)
            if response.status == 200:
                # Kaveri typically requires login for EC search
                # Public data extraction from available tables
                for table in response.css("table"):
                    for row in table.css("tr")[1:max_results]:
                        cells = row.css("td")
                        if cells:
                            results.append({
                                "raw": [c.text.strip() for c in cells],
                                "district": district,
                                "survey_number": survey_number,
                            })
        except Exception as e:
            logger.warning("Kaveri scrape failed (expected in demo): %s", e)
            results = self._generate_sample_ec(district, survey_number, ec_year_from, ec_year_to, max_results)
        
        self._save_result(cache_id, {"query": cache_key, "records": results})
        return results

    def _generate_sample_ec(
        self, district: str | None, survey_no: str | None,
        yr_from: int | None, yr_to: int | None, count: int,
    ) -> list[dict[str, Any]]:
        """Generate synthetic EC (Encumbrance Certificate) entries."""
        import random
        
        encumbrance_types = [
            "Sale Deed", "Gift Deed", "Mortgage Deed", "Release Deed",
            "Partition Deed", "Agreement of Sale", "Power of Attorney",
        ]
        
        records = []
        for i in range(min(count, 10)):
            yr = random.randint(yr_from or 2010, yr_to or 2025)
            has_mortgage = random.random() < 0.3
            records.append({
                "ec_entry_number": i + 1,
                "document_type": random.choice(encumbrance_types),
                "document_number": f"{random.randint(1000, 9999)}/{yr}",
                "registration_date": f"{yr}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                "sro": f"SRO {district or 'Bangalore North'}",
                "executant": f"Party_{random.randint(1, 20)}",
                "claimant": f"Party_{random.randint(1, 20)}",
                "survey_number": survey_no or f"{random.randint(1,500)}/A",
                "extent": f"{random.randint(500, 5000)} sq.ft",
                "consideration": str(random.randint(10, 200) * 100000),
                "has_mortgage": has_mortgage,
                "mortgage_to": "State Bank of India" if has_mortgage else None,
                "is_synthetic": True,
            })
        return records

    def to_training_record(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        if not raw.get("document_type"):
            return None
        ec_text = (
            f"EC Entry #{raw.get('ec_entry_number', 'N/A')}\n"
            f"Document: {raw['document_type']} No. {raw.get('document_number', 'N/A')}\n"
            f"Date: {raw.get('registration_date', 'N/A')}\n"
            f"SRO: {raw.get('sro', 'N/A')}\n"
            f"Executant: {raw.get('executant', 'N/A')}\n"
            f"Claimant: {raw.get('claimant', 'N/A')}\n"
            f"Survey No: {raw.get('survey_number', 'N/A')}\n"
            f"Consideration: Rs. {raw.get('consideration', 'N/A')}"
        )
        if raw.get("has_mortgage"):
            ec_text += f"\nMortgage to: {raw.get('mortgage_to', 'N/A')}"
        
        return {
            "instruction": (
                "Analyze this Encumbrance Certificate entry. Extract: document type, "
                "parties, survey number, whether there is an active mortgage or encumbrance, "
                "and risk level for a buyer."
            ),
            "input": ec_text,
            "output": {
                "document_type": raw["document_type"],
                "has_encumbrance": raw.get("has_mortgage", False),
                "encumbrance_type": "mortgage" if raw.get("has_mortgage") else None,
                "risk_level": "HIGH" if raw.get("has_mortgage") else "LOW",
                "survey_number": raw.get("survey_number"),
            },
            "source": "kaveri_synthetic",
        }
