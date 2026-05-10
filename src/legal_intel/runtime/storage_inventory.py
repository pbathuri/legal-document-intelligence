"""Device-local filesystem footprint for upload storage, manifests, and SQLite runs DB."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _walk_bounded(root: Path, *, max_files: int = 8000) -> tuple[int, int, int]:
    """Returns (total_bytes, file_count, scanned_paths_including_dirs_stopped_early)."""
    total = 0
    n_files = 0
    scanned = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            scanned += 1
            fp = Path(dirpath) / fn
            try:
                total += fp.stat().st_size
            except OSError:
                continue
            n_files += 1
            if n_files >= max_files:
                return total, n_files, scanned
        if n_files >= max_files:
            break
    return total, n_files, scanned


def _count_manifest_lines(path: Path, *, max_lines: int = 50_000) -> tuple[int | None, bool]:
    if not path.is_file():
        return None, False
    n = 0
    try:
        with path.open("rb") as f:
            for _ in f:
                n += 1
                if n >= max_lines:
                    return n, True
    except OSError:
        return None, False
    return n, False


def gather_storage_inventory(
    *,
    upload_storage_dir: str,
    runs_db_path: str,
    persist_uploads: bool,
) -> dict[str, Any]:
    """Sizes and counts under configured paths (bounded walks)."""
    up = Path(upload_storage_dir).resolve()
    up.mkdir(parents=True, exist_ok=True)

    total_b, n_files, scanned = _walk_bounded(up)

    manifest_p = up / "manifest.jsonl"
    man_size: int | None = None
    man_lines: int | None = None
    man_truncated = False
    if manifest_p.is_file():
        try:
            man_size = manifest_p.stat().st_size
        except OSError:
            man_size = None
        man_lines, man_truncated = _count_manifest_lines(manifest_p)

    db_p = Path(runs_db_path).resolve()
    runs_bytes: int | None = None
    if db_p.is_file():
        try:
            runs_bytes = db_p.stat().st_size
        except OSError:
            runs_bytes = None

    return {
        "persist_uploads_configured": persist_uploads,
        "upload_storage_dir": str(up),
        "upload_storage_bytes": total_b,
        "upload_storage_file_count": n_files,
        "upload_storage_walk_entries": scanned,
        "upload_walk_truncated_file_cap": n_files >= 8000,
        "manifest_path": str(manifest_p),
        "manifest_bytes": man_size,
        "manifest_line_count": man_lines,
        "manifest_line_count_truncated": man_truncated,
        "runs_db_path": str(db_p),
        "runs_db_bytes": runs_bytes,
    }
