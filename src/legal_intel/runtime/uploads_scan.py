"""Bounded filesystem listing under persisted upload storage (device-local)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def list_upload_storage_files(
    storage_dir: str,
    *,
    limit: int = 100,
    max_scan: int = 4000,
) -> dict[str, Any]:
    """Recent-first file listing; caps directory walk for safety."""
    root = Path(storage_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lim = max(1, min(limit, 500))
    cap = max(50, min(max_scan, 20_000))
    rows: list[dict[str, Any]] = []
    scanned = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            scanned += 1
            fp = Path(dirpath) / fn
            try:
                st = fp.stat()
            except OSError:
                continue
            try:
                rel = str(fp.relative_to(root))
            except ValueError:
                rel = str(fp)
            rows.append(
                {
                    "relative_path": rel,
                    "bytes": st.st_size,
                    "mtime_epoch": round(st.st_mtime, 3),
                }
            )
            if scanned >= cap:
                break
        if scanned >= cap:
            break
    rows.sort(key=lambda x: x["mtime_epoch"], reverse=True)
    truncated_scan = scanned >= cap
    return {
        "upload_storage_dir": str(root),
        "files": rows[:lim],
        "scanned_files": scanned,
        "truncated_scan": truncated_scan,
    }
