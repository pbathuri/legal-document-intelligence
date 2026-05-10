"""Validate filesystem paths for allowlisted local PDF ingestion."""

from __future__ import annotations

from pathlib import Path


def parse_allow_prefixes(raw: str) -> list[Path]:
    """Comma-separated absolute directory prefixes (resolved)."""
    out: list[Path] = []
    for part in (raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(Path(p).expanduser().resolve())
    return out


def is_path_under_allowlist(resolved_file: Path, prefixes: list[Path]) -> bool:
    if not prefixes:
        return False
    rf = resolved_file.resolve()
    for pre in prefixes:
        try:
            if rf.is_relative_to(pre.resolve()):
                return True
        except ValueError:
            continue
    return False
