"""Lightweight in-process request counters (no external deps)."""

from __future__ import annotations

import re
import threading
from typing import Any

_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def bucket_path(path: str) -> str:
    """Normalize paths so metrics cardinality stays bounded (UUID run ids, etc.)."""
    if not path:
        return "/"
    p = path.split("?", 1)[0]
    return _UUID.sub("{id}", p) or "/"


_lock = threading.Lock()
_total: int = 0
_by_bucket: dict[str, int] = {}


def incr_request(bucket: str) -> None:
    global _total
    with _lock:
        _total += 1
        _by_bucket[bucket] = _by_bucket.get(bucket, 0) + 1


def snapshot() -> dict[str, Any]:
    with _lock:
        return {
            "requests_total": _total,
            "by_path_bucket": dict(sorted(_by_bucket.items())),
        }


def prometheus_text() -> str:
    """Minimal Prometheus exposition for scraping (path buckets as labels)."""
    lines: list[str] = []
    with _lock:
        lines.append("# HELP legal_intel_http_requests_total Process HTTP requests since startup.")
        lines.append("# TYPE legal_intel_http_requests_total counter")
        lines.append(f"legal_intel_http_requests_total {_total}")
        lines.append("# HELP legal_intel_http_requests_by_path_bucket Labeled request counts.")
        lines.append("# TYPE legal_intel_http_requests_by_path_bucket counter")
        for path, n in sorted(_by_bucket.items()):
            esc = path.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'legal_intel_http_requests_by_path_bucket{{path="{esc}"}} {n}')
    return "\n".join(lines) + "\n"
