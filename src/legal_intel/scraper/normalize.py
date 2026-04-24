"""Normalize and deduplicate scraped records for the data lake + training prep."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable


def stable_record_id(rec: dict[str, Any], *, source: str) -> str:
    """Deterministic id for deduplication across runs."""
    keys: list[Any] = [source]
    for k in ("tid", "case_number", "document_number", "registration_date", "survey_number"):
        if rec.get(k) is not None:
            keys.append(rec.get(k))
    payload = json.dumps(keys, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def dedupe_records(
    records: Iterable[dict[str, Any]],
    key: Callable[[dict[str, Any]], tuple[Any, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Keep first occurrence per key(rec). Default key uses tid, case_number, or doc hash."""
    if key is None:
        def default_key(r: dict[str, Any]) -> tuple[Any, ...]:
            if r.get("tid") is not None:
                return ("tid", r["tid"])
            if r.get("case_number"):
                return ("case", str(r["case_number"]))
            if r.get("document_number"):
                return ("doc", str(r["document_number"]), str(r.get("registration_date", "")))
            return ("hash", hashlib.md5(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest())

        key = default_key

    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for r in records:
        k = key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def normalize_land_record(rec: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Attach canonical metadata for Parquet / manifests without losing raw fields."""
    n = dict(rec)
    n["_canonical_source"] = source
    n["_record_id"] = stable_record_id(rec, source=source)
    if "seller_names" in n and isinstance(n["seller_names"], str):
        n["seller_names"] = [n["seller_names"]]
    if "buyer_names" in n and isinstance(n["buyer_names"], str):
        n["buyer_names"] = [n["buyer_names"]]
    return n


def write_records_json(path: Path, records: list[dict[str, Any]]) -> None:
    """Write {records: [...]} for compatibility with training.prepare.load_scraped_records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records},
                    indent=2, ensure_ascii=False), encoding="utf-8")
