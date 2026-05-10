"""Persist uploaded PDFs on the local filesystem with an append-only manifest."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


def _safe_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^\w.\-]", "_", base)
    return base[:180] if len(base) > 180 else base


def persist_pdf_bytes(
    *,
    content: bytes,
    doc_id: str,
    original_filename: str,
    storage_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Write bytes to ``{storage_dir}/{doc_id}_{safe_name}`` and append a JSON line to manifest.jsonl.
    Returns (saved_path, manifest_path).
    """
    root = Path(storage_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe = _safe_filename(original_filename)
    if not safe.lower().endswith(".pdf"):
        safe = f"{safe}.pdf"
    dest = root / f"{doc_id}_{safe}"
    dest.write_bytes(content)

    manifest = root / "manifest.jsonl"
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "doc_id": doc_id,
        "doc_label": original_filename,
        "path": str(dest.resolve()),
        "bytes": len(content),
    }
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return dest, manifest
