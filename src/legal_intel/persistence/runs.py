"""SQLite-backed storage for completed diligence runs (full JSON blob per row)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS diligence_runs (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            domain TEXT NOT NULL,
            query TEXT NOT NULL,
            doc_ids_json TEXT NOT NULL,
            result_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_diligence_runs_created ON diligence_runs(created_at DESC)"
    )
    conn.commit()


def insert_run(
    *,
    db_path: Path,
    domain: str,
    query: str,
    doc_ids: list[str],
    result: dict[str, Any],
) -> str:
    rid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        init_schema(conn)
        conn.execute(
            "INSERT INTO diligence_runs (id, created_at, domain, query, doc_ids_json, result_json) VALUES (?,?,?,?,?,?)",
            (rid, ts, domain, query, json.dumps(doc_ids), json.dumps(result, default=str)),
        )
        conn.commit()
    finally:
        conn.close()
    return rid


@dataclass
class RunSummary:
    id: str
    created_at: str
    domain: str
    query: str
    doc_ids: list[str]


def list_runs(*, db_path: Path, limit: int = 50) -> list[RunSummary]:
    conn = _connect(db_path)
    try:
        init_schema(conn)
        cur = conn.execute(
            "SELECT id, created_at, domain, query, doc_ids_json FROM diligence_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    out: list[RunSummary] = []
    for r in rows:
        doc_ids = json.loads(r["doc_ids_json"])
        out.append(
            RunSummary(
                id=r["id"],
                created_at=r["created_at"],
                domain=r["domain"],
                query=r["query"],
                doc_ids=doc_ids if isinstance(doc_ids, list) else [],
            )
        )
    return out


def get_run(*, db_path: Path, run_id: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        init_schema(conn)
        cur = conn.execute(
            "SELECT id, created_at, domain, query, doc_ids_json, result_json FROM diligence_runs WHERE id = ?",
            (run_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "domain": row["domain"],
        "query": row["query"],
        "doc_ids": json.loads(row["doc_ids_json"]),
        "result": json.loads(row["result_json"]),
    }
