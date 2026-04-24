"""Web scraping for Indian land record portals using Scrapling."""
from legal_intel.scraper.base import BaseScraper
from legal_intel.scraper.igrs import IGRSScraper
from legal_intel.scraper.ecourts import ECourtsScraper
from legal_intel.scraper.kaveri import KaveriScraper

__all__ = ["BaseScraper", "IGRSScraper", "ECourtsScraper", "KaveriScraper"]
