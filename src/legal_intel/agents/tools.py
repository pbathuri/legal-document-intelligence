"""Agent tool definitions for the diligence workflow.

These tools can be called by the LangGraph agents to gather
additional context during analysis — e.g., checking for court cases,
fetching EC data, or running targeted scrapes.
"""
from __future__ import annotations

import json
import logging
from typing import Callable


logger = logging.getLogger(__name__)

# Tool registry: name -> (function, description)
TOOL_REGISTRY: dict[str, tuple[Callable[..., str], str]] = {}


def register_tool(name: str, description: str):
    """Decorator to register a callable as an agent tool."""
    def decorator(func: Callable[..., str]):
        TOOL_REGISTRY[name] = (func, description)
        return func
    return decorator


@register_tool(
    "check_disputes",
    "Search e-Courts for pending litigation involving a party name. "
    "Returns list of case records with status and risk assessment."
)
def check_disputes(party_name: str, state_code: str = "3") -> str:
    """Search e-Courts for disputes involving a party."""
    try:
        from legal_intel.scraper.ecourts import ECourtsScraper
        scraper = ECourtsScraper()
        results = scraper.scrape(party_name=party_name, state_code=state_code, max_results=5)
        return json.dumps({"disputes_found": len(results), "records": results}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "disputes_found": 0})


@register_tool(
    "fetch_ec_data",
    "Fetch Encumbrance Certificate entries for a survey number in Karnataka. "
    "Returns list of EC entries with encumbrance status."
)
def fetch_ec_data(
    survey_number: str,
    district: str = "Bangalore Urban",
    year_from: int = 2010,
    year_to: int = 2025,
) -> str:
    """Fetch EC data from Kaveri portal."""
    try:
        from legal_intel.scraper.kaveri import KaveriScraper
        scraper = KaveriScraper()
        results = scraper.scrape(
            survey_number=survey_number, district=district,
            ec_year_from=year_from, ec_year_to=year_to, max_results=10,
        )
        return json.dumps({"ec_entries": len(results), "records": results}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "ec_entries": 0})


@register_tool(
    "search_registrations",
    "Search IGRS portal for registration records by district and year. "
    "Returns matching registration metadata."
)
def search_registrations(
    district: str = "Hyderabad",
    year: int = 2024,
    doc_type: str = "sale_deed",
) -> str:
    """Search IGRS for registration records."""
    try:
        from legal_intel.scraper.igrs import IGRSScraper
        scraper = IGRSScraper()
        results = scraper.scrape(district=district, year=year, doc_type=doc_type, max_results=10)
        return json.dumps({"registrations_found": len(results), "records": results}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "registrations_found": 0})


@register_tool(
    "redact_pii",
    "Apply PII redaction (Aadhaar, PAN, phone) to text. Best-effort, not compliance-grade."
)
def redact_pii(text: str) -> str:
    """Redact PII from text."""
    from legal_intel.privacy import redact_all
    return redact_all(text)


def run_tool(name: str, **kwargs) -> str:
    """Execute a registered tool by name."""
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {name}", "available": list(TOOL_REGISTRY.keys())})
    func, _ = TOOL_REGISTRY[name]
    try:
        return func(**kwargs)
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return json.dumps({"error": str(e)})


def get_tool_descriptions() -> list[dict[str, str]]:
    """Return tool descriptions for LLM function-calling prompts."""
    return [
        {"name": name, "description": desc}
        for name, (_, desc) in TOOL_REGISTRY.items()
    ]
