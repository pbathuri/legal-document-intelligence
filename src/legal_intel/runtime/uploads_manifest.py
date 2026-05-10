"""Read tail of upload manifest.jsonl (device-local audit trail)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def tail_upload_manifest(
    storage_dir: str | Path,
    *,
    tail_lines: int = 80,
) -> dict[str, Any]:
    root = Path(storage_dir)
    path = root / "manifest.jsonl"
    if not path.is_file():
        return {"manifest_path": str(path.resolve()), "exists": False, "items": []}
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    n = max(1, min(tail_lines, 50_000))
    chunk = raw_lines[-n:]
    items: list[Any] = []
    for line in chunk:
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            items.append({"_parse_error": True, "raw": line[:500]})
    return {
        "manifest_path": str(path.resolve()),
        "exists": True,
        "total_lines_in_file": len(raw_lines),
        "returned": len(items),
        "items": items,
    }
