"""Batch due diligence from the command line."""

from __future__ import annotations
import argparse
import json
import sys
from legal_intel.config import get_settings
from legal_intel.graph.build import run_diligence, run_diligence_india
from legal_intel.pipeline import ingest_pdf


def main() -> None:
    p = argparse.ArgumentParser(description="Run diligence graph on indexed PDFs.")
    p.add_argument("pdfs", nargs="+", help="Paths to PDF files")
    p.add_argument("-q", "--query", required=True, help="Due diligence question")
    p.add_argument("--domain", choices=("mna", "india_re"), default=None)
    p.add_argument("--ocr", action="store_true", help="Enable OCR for scanned PDFs")
    p.add_argument("--json", action="store_true", help="Print full state as JSON")
    args = p.parse_args()

    doc_ids: list[str] = []
    labels: dict[str, str] = {}
    for path in args.pdfs:
        doc_id, n = ingest_pdf(path, use_ocr=args.ocr)
        doc_ids.append(doc_id)
        labels[doc_id] = path
        print(f"Indexed {path} -> {doc_id} ({n} chunks)", file=sys.stderr)

    domain = args.domain or get_settings().diligence_domain
    if domain == "india_re":
        out = run_diligence_india(args.query, doc_ids=doc_ids, doc_labels=labels)
    else:
        out = run_diligence(args.query)

    if args.json:
        print(json.dumps(dict(out), indent=2, default=str))
    else:
        print(out.get("final_report", ""))


if __name__ == "__main__":
    main()
