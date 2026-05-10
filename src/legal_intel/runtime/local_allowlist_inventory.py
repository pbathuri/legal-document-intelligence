"""Inspect configured ``LEGAL_INTEL_ALLOW_LOCAL_PATHS`` prefixes on the API host."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from legal_intel.runtime.local_paths import parse_allow_prefixes


def gather_local_allowlist_inventory(*, raw_allow: str) -> dict[str, Any]:
    prefixes = parse_allow_prefixes(raw_allow or "")
    items: list[dict[str, Any]] = []
    for p in prefixes:
        path = Path(p)
        row: dict[str, Any] = {
            "prefix": str(p),
            "exists": path.exists(),
            "is_dir": path.is_dir() if path.exists() else False,
        }
        if path.exists():
            try:
                resolved = path.resolve()
                row["resolved"] = str(resolved)
                du_root = resolved if resolved.is_dir() else resolved.parent
                du = shutil.disk_usage(du_root)
                row["disk_free_bytes"] = int(du.free)
                row["disk_total_bytes"] = int(du.total)
            except OSError as e:
                row["error"] = str(e)
        items.append(row)
    return {
        "allowlist_configured": bool(prefixes),
        "prefix_count": len(prefixes),
        "items": items,
    }
