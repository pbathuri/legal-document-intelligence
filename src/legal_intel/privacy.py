"""Best-effort PII redaction for demos (DPDP-aware; not a compliance guarantee)."""

from __future__ import annotations
import re

_AADHAAR_LIKE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
_PAN_LIKE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_PHONE_LIKE = re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b")


def redact_aadhaar_like(text: str) -> str:
    return _AADHAAR_LIKE.sub("[REDACTED_AADHAAR]", text)


def redact_pan_like(text: str) -> str:
    return _PAN_LIKE.sub("[REDACTED_PAN]", text)


def redact_phone(text: str) -> str:
    return _PHONE_LIKE.sub("[REDACTED_PHONE]", text)


def redact_all(text: str) -> str:
    """Apply all redaction patterns. Best-effort, NOT a compliance guarantee."""
    text = redact_aadhaar_like(text)
    text = redact_pan_like(text)
    text = redact_phone(text)
    return text
