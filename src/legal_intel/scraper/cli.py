"""CLI for running scrapers and building training datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from legal_intel.scraper.igrs import IGRSScraper
from legal_intel.scraper.ecourts import ECourtsScraper
from legal_intel.scraper.kaveri import KaveriScraper
from legal_intel.scraper.kanoon import IndianKanoonScraper
from legal_intel.scraper.normalize import dedupe_records, normalize_land_record, write_records_json


SCRAPERS = {
    "igrs": IGRSScraper,
    "ecourts": ECourtsScraper,
    "kaveri": KaveriScraper,
    "kanoon": IndianKanoonScraper,
}


def main() -> None:
    # Populate os.environ from .env so subprocess-visible vars match pydantic (cwd = project root).
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    p = argparse.ArgumentParser(description="Run land-record scrapers")
    p.add_argument("source", choices=list(SCRAPERS.keys()), help="Scraper source")
    p.add_argument("--district", default=None)
    p.add_argument("--party-name", default=None)
    p.add_argument("--survey-number", default=None)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--max-results", type=int, default=20)
    p.add_argument("--output", default=None, help="Output JSON path")
    p.add_argument("--training", action="store_true", help="Also output training records")
    p.add_argument(
        "--query",
        default="property dispute possession",
        help="Indian Kanoon search query (source=kanoon)",
    )
    p.add_argument("--pagenum", type=int, default=0, help="Kanoon results page (0-based)")
    p.add_argument("--doctypes", default=None, help="Kanoon doctypes filter (optional)")
    p.add_argument(
        "--no-full-text",
        action="store_true",
        help="Kanoon: skip per-document full-text fetch",
    )
    p.add_argument(
        "--write-raw",
        action="store_true",
        help="Write normalized {records: [...]} under data/raw/<source>/ for legal-train-prep",
    )
    args = p.parse_args()

    if args.source == "igrs":
        scraper = IGRSScraper()
        results = scraper.scrape(
            district=args.district, year=args.year, max_results=args.max_results
        )
    elif args.source == "ecourts":
        scraper = ECourtsScraper()
        results = scraper.scrape(party_name=args.party_name, max_results=args.max_results)
    elif args.source == "kaveri":
        scraper = KaveriScraper()
        results = scraper.scrape(
            district=args.district,
            survey_number=args.survey_number,
            max_results=args.max_results,
        )
    elif args.source == "kanoon":
        scraper = IndianKanoonScraper()
        results = scraper.scrape(
            query=args.query,
            pagenum=args.pagenum,
            doctypes=args.doctypes,
            max_results=args.max_results,
            fetch_full_text=not args.no_full_text,
        )
    else:
        print(f"Unknown source: {args.source}", file=sys.stderr)
        sys.exit(1)

    print(f"Scraped {len(results)} records from {args.source}", file=sys.stderr)

    if args.write_raw:
        if args.source == "kanoon":
            slug = hashlib.md5(args.query.encode()).hexdigest()[:10]
        else:
            slug = hashlib.md5(
                json.dumps(
                    {
                        "source": args.source,
                        "district": args.district,
                        "party_name": args.party_name,
                        "survey_number": args.survey_number,
                        "year": args.year,
                        "max_results": args.max_results,
                    },
                    sort_keys=True,
                ).encode(),
            ).hexdigest()[:10]
        raw_dir = Path(scraper.settings.scraper_data_dir) / args.source
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"batch_{slug}.json"
        normalized = dedupe_records([normalize_land_record(r, source=args.source) for r in results])
        write_records_json(raw_path, normalized)
        print(
            f"Wrote {len(normalized)} normalized records → {raw_path}",
            file=sys.stderr,
        )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        print(json.dumps(results, indent=2))

    if args.training:
        training_records = [scraper.to_training_record(r) for r in results]
        training_records = [r for r in training_records if r is not None]
        base = Path(args.output) if args.output else Path(f"{args.source}_training.jsonl")
        out_path = base.with_suffix(".training.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in training_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(training_records)} training records → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
