"""Large-scale Indian Kanoon harvest: paginated search, dedupe, checkpoint, JSONL corpus.

Requires a real API token (no synthetic fallback). Respects Settings.scraper_rate_limit
between requests. Use `legal-scrape-bulk kanoon` from the CLI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from legal_intel.scraper.kanoon import IndianKanoonScraper, _annotate_judgment_record, _strip_html

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, OSError | TimeoutError):
        return True
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (408, 425, 429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TransportError | httpx.TimeoutException)


def _search_with_retry(
    scraper: IndianKanoonScraper, params: dict[str, str | int]
) -> dict[str, Any] | None:
    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=2, max=90),
        retry=retry_if_exception(_is_retryable),
    )
    def _call() -> dict[str, Any] | None:
        return scraper._api_search(params)

    return _call()


def _rebuild_seen_tids(corpus_path: Path) -> set[int]:
    seen: set[int] = set()
    if not corpus_path.exists():
        return seen
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = row.get("tid")
            if tid is not None:
                seen.add(int(tid))
    logger.info("Resuming: %d unique tids already in %s", len(seen), corpus_path)
    return seen


def _default_checkpoint(out_dir: Path) -> Path:
    return out_dir / "bulk_kanoon_checkpoint.json"


def _backfill_missing_full_text(
    scraper: IndianKanoonScraper,
    corpus_path: Path,
    *,
    ft_budget: int,
    log_every: int = 100,
) -> int:
    """Rewrite JSONL in place: fetch full judgment text for rows with empty ``full_text_plain``.

    Returns the number of document fetches performed (bounded by ``ft_budget``).
    """
    if ft_budget <= 0 or not corpus_path.exists():
        return 0

    lines = corpus_path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    fetched = 0
    changed = False
    for i, rec in enumerate(records):
        if ft_budget <= 0:
            break
        tid = rec.get("tid")
        if tid is None:
            continue
        plain = (rec.get("full_text_plain") or "").strip()
        if plain:
            continue
        try:
            full = scraper._fetch_document_json(int(tid))
        except Exception as e:
            logger.debug("backfill doc fetch tid=%s: %s", tid, e)
            full = None
        if not full:
            continue
        raw_html = full.get("doc") or ""
        rec["full_text_plain"] = _strip_html(raw_html) if raw_html else ""
        cite = full.get("cite")
        if cite is not None:
            rec["cite"] = cite
        records[i] = _annotate_judgment_record(rec)
        fetched += 1
        ft_budget -= 1
        changed = True
        if log_every > 0 and fetched % log_every == 0:
            logger.info("Full-text backfill: fetched %d documents so far", fetched)

    if changed:
        tmp = corpus_path.with_suffix(corpus_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(corpus_path)
        logger.info(
            "Full-text backfill: updated %s (%d new full texts)",
            corpus_path,
            fetched,
        )
    elif fetched == 0:
        logger.info(
            "Full-text backfill: no rows needed full text in %s",
            corpus_path,
        )
    return fetched


def run_bulk_kanoon(
    scraper: IndianKanoonScraper,
    queries: list[str],
    out_dir: Path,
    *,
    checkpoint_path: Path | None = None,
    corpus_filename: str = "kanoon_corpus.jsonl",
    max_search_requests_per_query: int = 500,
    maxpages_per_request: int = 1,
    doctypes: str | None = None,
    fetch_full_text: bool = False,
    max_full_text_total: int = 0,
    flush_every: int = 250,
) -> dict[str, Any]:
    """Paginate search for each query; append newline-delimited JSON records.

    Checkpoint stores resume position. Corpus is append-only JSONL for training prep.
    """
    token = (scraper.settings.indian_kanoon_api_token or "").strip()
    if not token:
        raise RuntimeError(
            "INDIAN_KANOON_API_TOKEN is required for bulk harvest "
            "(export it or add to .env in the project root)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / corpus_filename
    ck_path = checkpoint_path or _default_checkpoint(out_dir)

    seen_tids = _rebuild_seen_tids(corpus_path)
    ck: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "corpus_path": str(corpus_path),
        "queries_completed": [],
        "current_query": None,
        "next_pagenum": 0,
        "search_requests_this_query": 0,
        "records_appended_this_run": 0,
        "full_text_fetched_this_run": 0,
    }
    if ck_path.exists():
        try:
            loaded = json.loads(ck_path.read_text(encoding="utf-8"))
            if loaded.get("version") == CHECKPOINT_VERSION:
                ck.update({k: loaded[k] for k in ck if k in loaded})
                if loaded.get("current_query") in queries:
                    qidx = queries.index(str(loaded["current_query"]))
                    for q in queries[:qidx]:
                        if q not in ck["queries_completed"]:
                            ck["queries_completed"].append(q)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Ignoring bad checkpoint %s: %s", ck_path, e)

    buffer: list[dict[str, Any]] = []
    stats = {
        "queries_total": len(queries),
        "queries_finished": 0,
        "search_requests": 0,
        "records_new": 0,
        "records_skipped_dup": 0,
        "full_text_fetched": 0,
    }

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        with corpus_path.open("a", encoding="utf-8") as f:
            for rec in buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        buffer = []

    def save_checkpoint() -> None:
        ck_path.write_text(json.dumps(ck, indent=2, ensure_ascii=False), encoding="utf-8")

    ft_budget = max_full_text_total if fetch_full_text else 0

    for query in queries:
        if query in ck["queries_completed"]:
            continue

        if ck.get("current_query") == query:
            pagenum = int(ck.get("next_pagenum", 0))
            req_count = int(ck.get("search_requests_this_query", 0))
        else:
            pagenum = 0
            req_count = 0

        ck["current_query"] = query
        while req_count < max_search_requests_per_query:
            params: dict[str, str | int] = {
                "formInput": query,
                "pagenum": pagenum,
                "maxpages": max(1, min(maxpages_per_request, 50)),
            }
            if doctypes:
                params["doctypes"] = doctypes

            try:
                data = _search_with_retry(scraper, params)
            except Exception as e:
                logger.error("Search failed for query=%r pagenum=%s: %s", query, pagenum, e)
                save_checkpoint()
                raise

            stats["search_requests"] += 1
            req_count += 1
            ck["search_requests_this_query"] = req_count

            if not data:
                logger.warning("Empty search response; stopping this query.")
                break

            docs = data.get("docs") or []
            if not docs:
                logger.info("No more results for query=%r at pagenum=%s", query, pagenum)
                break

            for doc in docs:
                tid = doc.get("tid")
                if tid is None:
                    continue
                tid_i = int(tid)
                if tid_i in seen_tids:
                    stats["records_skipped_dup"] += 1
                    continue
                seen_tids.add(tid_i)

                rec: dict[str, Any] = {
                    "tid": tid_i,
                    "title": doc.get("title") or "",
                    "headline": doc.get("headline") or "",
                    "docsource": doc.get("docsource") or "",
                    "publishdate": doc.get("publishdate"),
                    "docsize": doc.get("docsize"),
                    "query": query,
                    "is_synthetic": False,
                    "bulk_harvest": True,
                }
                if fetch_full_text and ft_budget > 0:
                    try:
                        full = scraper._fetch_document_json(tid_i)
                    except Exception as e:
                        logger.debug("doc fetch tid=%s: %s", tid_i, e)
                        full = None
                    if full:
                        raw_html = full.get("doc") or ""
                        rec["full_text_plain"] = _strip_html(raw_html) if raw_html else ""
                        rec["cite"] = full.get("cite")
                        ft_budget -= 1
                        stats["full_text_fetched"] += 1
                        ck["full_text_fetched_this_run"] = stats["full_text_fetched"]

                rec = _annotate_judgment_record(rec)
                buffer.append(rec)
                stats["records_new"] += 1
                ck["records_appended_this_run"] = stats["records_new"]

                if len(buffer) >= flush_every:
                    flush_buffer()
                    ck["next_pagenum"] = pagenum + maxpages_per_request
                    save_checkpoint()

            pagenum += maxpages_per_request
            ck["next_pagenum"] = pagenum

        flush_buffer()
        ck["queries_completed"].append(query)
        ck["current_query"] = None
        ck["next_pagenum"] = 0
        ck["search_requests_this_query"] = 0
        stats["queries_finished"] += 1
        save_checkpoint()

    flush_buffer()

    if stats["queries_finished"] == 0 and queries:
        done = set(ck.get("queries_completed") or [])
        if done.issuperset(queries):
            stats["queries_finished"] = len(queries)

    if fetch_full_text and ft_budget > 0:
        bf = _backfill_missing_full_text(
            scraper,
            corpus_path,
            ft_budget=ft_budget,
            log_every=max(1, flush_every),
        )
        stats["full_text_fetched"] += bf
        ck["full_text_fetched_this_run"] = stats["full_text_fetched"]

    save_checkpoint()
    logger.info("Bulk Kanoon done: %s", stats)
    return stats


def load_queries_from_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    return out
