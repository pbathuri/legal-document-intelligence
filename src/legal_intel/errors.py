"""Structured errors for legal_intel (prefer over bare except Exception)."""


class LegalIntelError(Exception):
    """Base error for recoverable pipeline failures."""


class DisputeCheckError(LegalIntelError):
    """Dispute lookup or scraper tool failed."""


class ExtractionError(LegalIntelError):
    """Structured extraction could not produce valid facts."""


class RerankerError(LegalIntelError):
    """Cross-encoder re-ranking failed (ANN results may still be used)."""
