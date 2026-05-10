"""Scraper for IGRS (Inspector General of Registration and Stamps) portals.

Targets publicly available registration data from state IGRS portals.
Different states have different portals:
- Telangana: registration.telangana.gov.in
- AP: registration.ap.gov.in
- Maharashtra: igrmaharashtra.gov.in
- Karnataka: kavfrionline.karnataka.gov.in (separate scraper)
- Tamil Nadu: tnreginet.gov.in

This scraper focuses on publicly indexed document metadata (NOT private records).
It collects: document types, registration dates, SRO names, party counts,
property descriptions, and consideration amounts from public search results.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from legal_intel.scraper.base import BaseScraper

logger = logging.getLogger(__name__)


class IGRSScraper(BaseScraper):
    """Scrape publicly available IGRS registration metadata.

    NOTE: This scraper only accesses publicly searchable data.
    It does NOT bypass authentication, CAPTCHAs, or access restricted records.
    For hackathon purposes, it demonstrates the pipeline; production use requires
    proper API access agreements with state registrars.
    """

    source_name = "igrs"

    # Public search endpoints (example: Telangana — replace per state)
    ENDPOINTS = {
        "telangana": "https://registration.telangana.gov.in",
        "maharashtra": "https://igrmaharashtra.gov.in",
        "tamilnadu": "https://tnreginet.gov.in",
    }

    def __init__(self, state: str = "telangana") -> None:
        super().__init__()
        self.state = state
        self.base_url = self.ENDPOINTS.get(state, self.ENDPOINTS["telangana"])

    def scrape(
        self,
        *,
        district: str | None = None,
        sro: str | None = None,
        year: int | None = None,
        doc_type: str = "sale_deed",
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Scrape public registration search results.

        This is a DEMONSTRATION scraper. Real IGRS portals require:
        - CAPTCHA solving (human-in-loop or authorized API)
        - Session management
        - State-specific form field names

        For the hackathon, we provide:
        1. The scraper architecture that plugs into training pipeline
        2. Sample data generation for testing
        3. The actual HTTP request structure (commented) for real deployment
        """
        max_results = max_results or self.settings.scraper_max_pages
        results: list[dict[str, Any]] = []

        # Check cache first
        cache_key = f"{self.state}_{district}_{sro}_{year}_{doc_type}"
        cache_id = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        cached = self._load_cached(cache_id)
        if cached and "records" in cached:
            logger.info("Using cached IGRS data for %s", cache_key)
            return cached["records"]

        fetcher = self._get_fetcher()

        if fetcher is None:
            # Scrapling not available — use synthetic data
            results = self._generate_sample_data(district, sro, year, doc_type, max_results)
            self._save_result(cache_id, {"query": cache_key, "records": results})
            return results

        # Attempt to fetch the public search page
        try:
            self._rate_limit()
            response = fetcher.get(self.base_url, timeout=self.settings.agent_tool_timeout)

            if response.status == 200:
                # Parse available district/SRO options from the page
                page = response

                # Look for registration data tables
                tables = page.css("table")
                for table in tables:
                    rows = table.css("tr")
                    for row in rows[1:max_results]:  # skip header
                        cells = row.css("td")
                        if len(cells) >= 4:
                            record = {
                                "state": self.state,
                                "district": district or "unknown",
                                "sro": sro or "unknown",
                                "doc_type": doc_type,
                                "raw_cells": [c.text.strip() for c in cells],
                            }
                            results.append(record)

                logger.info("Scraped %d records from %s IGRS", len(results), self.state)
            else:
                logger.warning("IGRS returned status %d", response.status)

        except Exception as e:
            logger.warning("IGRS scrape failed (expected in demo): %s", e)
            # Generate synthetic sample data for pipeline testing
            results = self._generate_sample_data(district, sro, year, doc_type, max_results)

        # Save results
        self._save_result(cache_id, {"query": cache_key, "records": results})
        return results

    def _generate_sample_data(
        self,
        district: str | None,
        sro: str | None,
        year: int | None,
        doc_type: str,
        count: int,
    ) -> list[dict[str, Any]]:
        """Generate realistic synthetic IGRS records for pipeline testing."""
        import random

        districts = ["Hyderabad", "Rangareddy", "Medchal-Malkajgiri", "Sangareddy"]
        sros = ["SRO Kukatpally", "SRO Secunderabad", "SRO Musheerabad", "SRO Rajendranagar"]
        doc_types = ["sale_deed", "gift_deed", "mortgage", "partition", "release", "GPA"]

        first_names = ["Rajesh", "Srinivas", "Lakshmi", "Priya", "Venkat", "Suresh", "Deepa"]
        last_names = ["Reddy", "Sharma", "Kumar", "Rao", "Naidu", "Gupta", "Singh"]

        records = []
        for i in range(min(count, 20)):
            yr = year or random.randint(2015, 2025)
            rec = {
                "state": self.state,
                "district": district or random.choice(districts),
                "sro": sro or random.choice(sros),
                "doc_type": doc_type if doc_type != "sale_deed" else random.choice(doc_types),
                "registration_date": f"{yr}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "document_number": f"{yr}/{random.randint(1000, 9999)}",
                "seller_names": [f"{random.choice(first_names)} {random.choice(last_names)}"],
                "buyer_names": [f"{random.choice(first_names)} {random.choice(last_names)}"],
                "consideration_amount": str(random.randint(10, 500) * 100000),
                "property_description": (
                    f"Plot No. {random.randint(1, 500)}, "
                    f"Survey No. {random.randint(1, 999)}/{random.choice(['A', 'B', 'C', ''])}, "
                    f"Extent: {random.randint(100, 5000)} sq.yards"
                ),
                "survey_numbers": [f"{random.randint(1, 999)}/{random.choice(['A', 'B', 'C'])}"],
                "is_synthetic": True,
            }
            records.append(rec)
        return records

    def to_training_record(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        """Convert IGRS record to instruction-tuning format for legal extraction."""
        if not raw.get("seller_names") and not raw.get("buyer_names"):
            return None

        # Build a realistic "document text" from the structured data
        doc_text = self._record_to_document_text(raw)

        # Build the expected JSON output
        expected = {
            "doc_type": raw.get("doc_type", "unknown"),
            "registration_date": raw.get("registration_date"),
            "seller_names": raw.get("seller_names", []),
            "buyer_names": raw.get("buyer_names", []),
            "parcel_ids": raw.get("survey_numbers", []),
            "locality": raw.get("district"),
            "consideration_amount": raw.get("consideration_amount"),
            "mentions_dispute": False,
            "mentions_encumbrance": False,
        }

        return {
            "instruction": (
                "Extract structured facts from this Indian property registration document. "
                "Return JSON with: doc_type, registration_date, seller_names, buyer_names, "
                "parcel_ids, locality, consideration_amount, mentions_dispute, mentions_encumbrance."
            ),
            "input": doc_text,
            "output": expected,
            "source": "igrs_synthetic",
        }

    def _record_to_document_text(self, rec: dict[str, Any]) -> str:
        """Convert structured record to realistic document-like text."""
        sellers = ", ".join(rec.get("seller_names", ["Unknown"]))
        buyers = ", ".join(rec.get("buyer_names", ["Unknown"]))
        return (
            f"SALE DEED\n"
            f"Document No: {rec.get('document_number', 'N/A')}\n"
            f"Sub-Registrar Office: {rec.get('sro', 'N/A')}\n"
            f"Date of Registration: {rec.get('registration_date', 'N/A')}\n\n"
            f"THIS DEED OF SALE is executed by {sellers} (hereinafter called the VENDOR/SELLER) "
            f"in favour of {buyers} (hereinafter called the VENDEE/PURCHASER).\n\n"
            f"SCHEDULE OF PROPERTY:\n{rec.get('property_description', 'N/A')}\n\n"
            f"The total sale consideration agreed upon is Rs. {rec.get('consideration_amount', 'N/A')}/-\n"
        )
