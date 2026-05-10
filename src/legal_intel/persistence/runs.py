"""SQLite-backed storage for completed diligence runs (full JSON blob per row)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterator
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


def search_runs(*, db_path: Path, q: str, limit: int = 40) -> list[RunSummary]:
    """Substring match on query text, domain, or run id (case-insensitive)."""
    ln = (q or "").strip().lower()
    if not ln:
        return []
    conn = _connect(db_path)
    try:
        init_schema(conn)
        lim = min(limit, 200)
        cur = conn.execute(
            """
            SELECT id, created_at, domain, query, doc_ids_json FROM diligence_runs
            WHERE instr(lower(query), ?) > 0
               OR instr(lower(domain), ?) > 0
               OR instr(lower(id), ?) > 0
            ORDER BY created_at DESC LIMIT ?
            """,
            (ln, ln, ln, lim),
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


def optimize_sqlite_file(db_path: Path) -> dict[str, Any]:
    """Run ``PRAGMA optimize`` on an SQLite database (query planner stats; local maintenance)."""
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(str(db_path.resolve()))
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA optimize")
        conn.commit()
    finally:
        conn.close()
    return {"path": str(db_path.resolve()), "pragma_optimize": True}


def vacuum_sqlite_file(db_path: Path) -> dict[str, Any]:
    """Run SQLite VACUUM on the runs database (reclaim space / optimize pages)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    before = db_path.stat().st_size if db_path.is_file() else 0
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()
    after = db_path.stat().st_size if db_path.is_file() else 0
    return {
        "path": str(db_path.resolve()),
        "bytes_before": before,
        "bytes_after": after,
    }


def delete_run(*, db_path: Path, run_id: str) -> bool:
    conn = _connect(db_path)
    try:
        init_schema(conn)
        cur = conn.execute("DELETE FROM diligence_runs WHERE id = ?", (run_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def export_runs_json_array(*, db_path: Path, limit: int = 10_000) -> list[dict[str, Any]]:
    """Full run rows as JSON-serializable dicts (bounded list for browser/API export)."""
    conn = _connect(db_path)
    try:
        init_schema(conn)
        cur = conn.execute(
            """
            SELECT id, created_at, domain, query, doc_ids_json, result_json
            FROM diligence_runs
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "domain": row["domain"],
                "query": row["query"],
                "doc_ids": json.loads(row["doc_ids_json"]),
                "result": json.loads(row["result_json"]),
            }
        )
    return out


def iter_runs_ndjson_lines(*, db_path: Path, limit: int = 50_000) -> Iterator[str]:
    """Export rows as newline-delimited JSON (oldest first for stable archival)."""
    conn = _connect(db_path)
    try:
        init_schema(conn)
        cur = conn.execute(
            """
            SELECT id, created_at, domain, query, doc_ids_json, result_json
            FROM diligence_runs
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (limit,),
        )
        for row in cur:
            obj = {
                "id": row["id"],
                "created_at": row["created_at"],
                "domain": row["domain"],
                "query": row["query"],
                "doc_ids": json.loads(row["doc_ids_json"]),
                "result": json.loads(row["result_json"]),
            }
            yield json.dumps(obj, default=str) + "\n"
    finally:
        conn.close()


def gather_run_statistics(*, db_path: Path) -> dict[str, Any]:
    """Aggregate counts and time range for the diligence runs SQLite file (creates empty schema if missing)."""
    db_path = Path(db_path)
    resolved = str(db_path.resolve())
    if not db_path.is_file():
        return {
            "db_path": resolved,
            "file_bytes": 0,
            "row_count": 0,
            "by_domain": {},
            "created_at_min": None,
            "created_at_max": None,
        }
    file_bytes = db_path.stat().st_size
    conn = _connect(db_path)
    try:
        init_schema(conn)
        row = conn.execute("SELECT COUNT(*) AS n FROM diligence_runs").fetchone()
        row_count = int(row["n"]) if row else 0
        cur = conn.execute(
            "SELECT domain, COUNT(*) AS c FROM diligence_runs GROUP BY domain ORDER BY domain"
        )
        by_domain: dict[str, int] = {}
        for r in cur.fetchall():
            by_domain[str(r["domain"])] = int(r["c"])
        mm = conn.execute(
            "SELECT MIN(created_at) AS lo, MAX(created_at) AS hi FROM diligence_runs"
        ).fetchone()
    finally:
        conn.close()
    return {
        "db_path": resolved,
        "file_bytes": file_bytes,
        "row_count": row_count,
        "by_domain": by_domain,
        "created_at_min": mm["lo"] if mm else None,
        "created_at_max": mm["hi"] if mm else None,
    }


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
