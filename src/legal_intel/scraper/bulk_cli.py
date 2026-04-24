"""CLI for large-scale scraper harvests (Indian Kanoon corpus, checkpoints, JSONL)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from legal_intel.scraper.bulk_kanoon import load_queries_from_file, run_bulk_kanoon
from legal_intel.scraper.kanoon import IndianKanoonScraper


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(
        description="Bulk harvest for legal data lakes (checkpoint + JSONL)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    kn = sub.add_parser(
        "kanoon",
        help="Paginate Indian Kanoon search across many queries; append kanoon_corpus.jsonl",
    )
    kn.add_argument(
        "--queries-file",
        type=Path,
        required=True,
        help="Text file: one search query per line (# comments allowed)",
    )
    kn.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/kanoon_bulk"),
        help="Output directory (default: data/raw/kanoon_bulk)",
    )
    kn.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint JSON path (default: <out-dir>/bulk_kanoon_checkpoint.json)",
    )
    kn.add_argument(
        "--corpus-name",
        default="kanoon_corpus.jsonl",
        help="JSONL filename inside out-dir",
    )
    kn.add_argument(
        "--max-requests-per-query",
        type=int,
        default=500,
        help="Max search API calls per query (safety cap)",
    )
    kn.add_argument(
        "--maxpages",
        type=int,
        default=1,
        help="Indian Kanoon maxpages per request (1–50; higher = fewer round-trips)",
    )
    kn.add_argument("--doctypes", default=None,
                    help="Optional doctypes filter (IK docs)")
    kn.add_argument(
        "--fetch-full-text",
        action="store_true",
        help=(
            "Fetch full judgment HTML for new hits; also backfills existing JSONL rows "
            "that lack full_text_plain (slow; uses more API quota)"
        ),
    )
    kn.add_argument(
        "--max-full-text",
        type=int,
        default=0,
        help="Max full-text document fetches in this run (0 = unlimited if --fetch-full-text)",
    )
    kn.add_argument(
        "--flush-every",
        type=int,
        default=250,
        help="Append JSONL batch size / checkpoint cadence",
    )

    args = ap.parse_args()
    if args.cmd == "kanoon":
        queries = load_queries_from_file(args.queries_file)
        if not queries:
            print("No queries in file.", file=sys.stderr)
            sys.exit(2)
        scraper = IndianKanoonScraper()
        max_ft = 0
        if args.fetch_full_text:
            max_ft = args.max_full_text if args.max_full_text > 0 else 999_999_999
        try:
            stats = run_bulk_kanoon(
                scraper,
                queries,
                args.out_dir,
                checkpoint_path=args.checkpoint,
                corpus_filename=args.corpus_name,
                max_search_requests_per_query=args.max_requests_per_query,
                maxpages_per_request=args.maxpages,
                doctypes=args.doctypes,
                fetch_full_text=args.fetch_full_text,
                max_full_text_total=max_ft,
                flush_every=args.flush_every,
            )
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
