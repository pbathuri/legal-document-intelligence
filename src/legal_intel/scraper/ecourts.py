"""Scraper for e-Courts (ecourts.gov.in) — public case status data.

e-Courts Services is India's e-governance initiative providing
case information, cause lists, and case status through the
National Judicial Data Grid. This scraper accesses ONLY the
publicly available case search API/pages.

Use case: detect pending litigation / lis pendens on a property
by searching party names or case numbers from the property packet.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from legal_intel.scraper.base import BaseScraper

logger = logging.getLogger(__name__)


class ECourtsScraper(BaseScraper):
    """Search e-Courts for property-related disputes by party name or case number.

    Public access only — no authentication bypass.
    """

    source_name = "ecourts"
    BASE_URL = "https://services.ecourts.gov.in/ecourtindia_v6"

    def scrape(
        self,
        *,
        party_name: str | None = None,
        case_number: str | None = None,
        state_code: str = "3",  # Telangana=3, Karnataka=4, etc.
        district_code: str = "1",
        case_type: str = "civil",
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        max_results = max_results or self.settings.scraper_max_pages

        cache_key = f"ecourts_{party_name}_{case_number}_{state_code}"
        cache_id = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        cached = self._load_cached(cache_id)
        if cached and "records" in cached:
            return cached["records"]

        results: list[dict[str, Any]] = []
        fetcher = self._get_fetcher()

        if fetcher is None:
            results = self._generate_sample_disputes(party_name, max_results)
            self._save_result(cache_id, {"query": cache_key, "records": results})
            return results

        try:
            self._rate_limit()
            # e-Courts uses POST forms with session tokens
            # Real implementation requires CAPTCHA handling
            response = fetcher.get(self.BASE_URL, timeout=30)

            if response.status == 200:
                # Parse case listing tables
                tables = response.css("table.table")
                for table in tables:
                    for row in table.css("tr")[1:max_results]:
                        cells = row.css("td")
                        if len(cells) >= 3:
                            results.append(
                                {
                                    "case_number": cells[0].text.strip() if cells else "",
                                    "parties": cells[1].text.strip() if len(cells) > 1 else "",
                                    "status": cells[2].text.strip() if len(cells) > 2 else "",
                                    "state_code": state_code,
                                }
                            )
        except Exception as e:
            logger.warning("e-Courts scrape failed (expected in demo): %s", e)
            # Generate synthetic dispute data
            results = self._generate_sample_disputes(party_name, max_results)

        self._save_result(cache_id, {"query": cache_key, "records": results})
        return results

    def _generate_sample_disputes(self, party_name: str | None, count: int) -> list[dict[str, Any]]:
        """Synthetic litigation records for pipeline testing."""
        import random

        case_types = [
            "OS (Original Suit)",
            "SA (Second Appeal)",
            "CRP (Civil Revision Petition)",
            "WP (Writ Petition)",
        ]
        statuses = ["Pending", "Disposed", "Pending - Next hearing scheduled", "Transferred"]

        records = []
        for i in range(min(count, 5)):
            yr = random.randint(2018, 2025)
            records.append(
                {
                    "case_number": f"{random.choice(case_types)} No. {random.randint(100, 9999)}/{yr}",
                    "parties": f"{party_name or 'Petitioner'} vs State of Telangana & Others",
                    "status": random.choice(statuses),
                    "court": f"District Court, {'Hyderabad' if i % 2 == 0 else 'Rangareddy'}",
                    "filing_date": f"{yr}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                    "property_related": random.choice([True, True, True, False]),
                    "subject": random.choice(
                        [
                            "Suit for declaration of title and possession",
                            "Suit for permanent injunction",
                            "Partition suit",
                            "Suit for specific performance of agreement of sale",
                        ]
                    ),
                    "is_synthetic": True,
                }
            )
        return records

    def to_training_record(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        if not raw.get("case_number"):
            return None
        return {
            "instruction": (
                "Analyze this court case record and determine if it indicates "
                "a property dispute that should be flagged during due diligence."
            ),
            "input": (
                f"Case: {raw['case_number']}\n"
                f"Parties: {raw.get('parties', 'N/A')}\n"
                f"Status: {raw.get('status', 'N/A')}\n"
                f"Court: {raw.get('court', 'N/A')}\n"
                f"Subject: {raw.get('subject', 'N/A')}"
            ),
            "output": {
                "is_property_dispute": raw.get("property_related", False),
                "risk_level": "HIGH" if raw.get("status") == "Pending" else "MEDIUM",
                "recommendation": (
                    "Active litigation on property. Recommend legal opinion before proceeding."
                    if raw.get("status") == "Pending"
                    else "Disposed case. Verify final order outcome."
                ),
            },
            "source": "ecourts_synthetic",
        }
