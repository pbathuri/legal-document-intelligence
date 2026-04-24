#!/usr/bin/env python3
"""Offline evaluation helpers for India property diligence (synthetic fixtures).

Metrics: field extraction overlap (proxy F1), chain-break flag accuracy, groundedness proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _token_set_f1(pred: list[str], gold: list[str]) -> float:
    ps, gs = set(pred), set(gold)
    if not ps and not gs:
        return 1.0
    if not ps or not gs:
        return 0.0
    tp = len(ps & gs)
    prec = tp / len(ps) if ps else 0.0
    rec = tp / len(gs) if gs else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate India diligence outputs vs expected.json")
    ap.add_argument("expected", type=Path, help="Path to expected.json")
    ap.add_argument("actual", type=Path,
                    help="Path to model JSON output (same shape as run --json)")
    args = ap.parse_args()
    exp = _load_json(args.expected)
    act = _load_json(args.actual)

    scores: dict[str, float] = {}
    gold_facts = exp.get("instrument_facts", [])
    pred_facts = act.get("instrument_facts_json")
    if isinstance(pred_facts, str):
        pred_facts = json.loads(pred_facts)
    if gold_facts and pred_facts:
        f1s = []
        for g, p in zip(gold_facts, pred_facts, strict=False):
            for key in ("buyer_names", "seller_names", "parcel_ids"):
                f1s.append(
                    _token_set_f1(
                        list(g.get(key) or []),
                        list(p.get(key) or []),
                    )
                )
        scores["field_f1_macro"] = sum(f1s) / len(f1s) if f1s else 0.0

    gold_breaks = exp.get("expect_chain_break", False)
    tg = act.get("title_graph_json")
    if isinstance(tg, str):
        tg = json.loads(tg)
    br = tg.get("breaks", []) if isinstance(tg, dict) else []
    pred_break = len(br) > 0
    scores["chain_break_accuracy"] = 1.0 if pred_break == gold_breaks else 0.0

    memo = act.get("final_report", "")
    grounded = 1.0 if ("p." in memo.lower() or "[page" in memo.lower(
    ) or "excerpt" in memo.lower()) else 0.0
    scores["groundedness_proxy"] = grounded

    print(json.dumps({"scores": scores}, indent=2))


if __name__ == "__main__":
    main()
