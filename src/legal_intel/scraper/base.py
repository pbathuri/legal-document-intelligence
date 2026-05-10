"""Base scraper class with rate limiting and data persistence using Scrapling."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from legal_intel.config import get_settings

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Base class for all land-record scrapers.

    Implements:
    - Rate limiting between requests
    - Result caching to disk (data/raw/<source>/)
    - Structured output format for downstream training/RAG
    - Error logging with retry metadata
    """

    source_name: str = "base"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._last_request_time = 0.0
        self._output_dir = Path(self.settings.scraper_data_dir) / self.source_name
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        wait = self.settings.scraper_rate_limit - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()

    def _get_fetcher(self):
        """Get a Scrapling Fetcher (StealthyFetcher for JS-heavy sites,
        Fetcher for simple ones)."""
        try:
            from scrapling import Fetcher

            return Fetcher(auto_match=True)
        except ImportError:
            logger.warning("scrapling not installed. Using synthetic data fallback.")
            return None

    def _get_stealth_fetcher(self):
        """Get a stealth fetcher for sites with bot detection."""
        try:
            from scrapling import StealthyFetcher

            return StealthyFetcher(auto_match=True)
        except ImportError:
            logger.warning("scrapling not installed. Using synthetic data fallback.")
            return None

    def _save_result(self, record_id: str, data: dict[str, Any]) -> Path:
        """Persist a scraped record to JSON on disk."""
        data["_meta"] = {
            "source": self.source_name,
            "record_id": record_id,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }
        fname = f"{record_id}.json"
        path = self._output_dir / fname
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved %s → %s", record_id, path)
        return path

    def _load_cached(self, record_id: str) -> dict[str, Any] | None:
        """Return cached result if exists."""
        path = self._output_dir / f"{record_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    @abstractmethod
    def scrape(self, **kwargs) -> list[dict[str, Any]]:
        """Run the scraper. Returns list of structured records."""
        ...

    @abstractmethod
    def to_training_record(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a raw scraped record to a training-ready format.
        Returns None if the record is unusable."""
        ...
