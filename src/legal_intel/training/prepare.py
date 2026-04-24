"""Prepare training datasets for fine-tuning legal extraction models on AMD GPUs.

Pipeline:
1. Collect raw records from scrapers (data/raw/)
2. Convert to instruction-tuning format (instruction, input, output)
3. Add synthetic augmentation (paraphrases, noise)
4. Split into train/val
5. Export as JSONL for vLLM/HuggingFace fine-tuning

Target: fine-tune Llama 3.1 8B or 70B on MI300X using:
- LoRA (Parameter-Efficient Fine-Tuning)
- ROCm + PyTorch + HuggingFace PEFT/TRL
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Literal


OutputFormat = Literal["jsonl", "parquet", "both"]

logger = logging.getLogger(__name__)


# === Instruction templates for different extraction tasks ===

EXTRACTION_TEMPLATE = """You are an AI assistant specialized in Indian property document analysis.
Given the following document text, extract structured information as JSON.

### Document:
{input}

### Required Fields:
- doc_type: Type of document (sale_deed, gift_deed, mortgage, partition, EC_entry, court_order, etc.)
- registration_date: Date of registration (YYYY-MM-DD or null)
- seller_names: List of seller/vendor/executant names
- buyer_names: List of buyer/purchaser/claimant names  
- parcel_ids: List of survey numbers, plot numbers, khata numbers
- locality: District/taluk/village
- consideration_amount: Transaction amount as string (or null)
- mentions_dispute: Boolean - does the document mention any dispute?
- mentions_encumbrance: Boolean - does the document mention mortgage/lien/charge?

### Output (JSON only):"""

RISK_ASSESSMENT_TEMPLATE = """You are a property due diligence specialist for Indian real estate.
Analyze the following document/record and assess risk for a potential buyer.

### Record:
{input}

### Assess:
1. Risk level (HIGH/MEDIUM/LOW)
2. Key risk factors
3. Recommended verification steps
4. Whether this blocks transaction

### Assessment (JSON only):"""

CHAIN_ANALYSIS_TEMPLATE = """You are a title chain analyst for Indian property transactions.
Given the following set of instruments, reconstruct the ownership chain and identify gaps.

### Instruments:
{input}

### Analyze:
1. Chronological chain of transfers
2. Any breaks in the chain (missing links)
3. Name mismatches between consecutive transfers
4. Survey/plot number consistency
5. Overall chain integrity score (0-1)

### Chain Analysis (JSON only):"""


def load_curated_manifest(manifest_path: str) -> list[dict[str, Any]]:
    """Load curated-upload manifest (JSONL): one JSON object per line with optional fields:
    path, doc_id, language, doc_type, is_scanned, ocr_engine, human_verified, source, etc.
    """
    path = Path(manifest_path)
    if not path.exists():
        logger.warning("Manifest %s does not exist", manifest_path)
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning("Skipping manifest line: %s", e)
    logger.info("Loaded %d manifest rows from %s", len(rows), manifest_path)
    return rows


def manifest_row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize manifest row + optional sidecar JSON into record_to_sft_examples input shape."""
    rec = dict(row)
    p = rec.get("path") or rec.get("file")
    if p and Path(p).exists() and p.lower().endswith(".json"):
        try:
            extra = json.loads(Path(p).read_text(encoding="utf-8"))
            if isinstance(extra, dict):
                rec = {**extra, **rec}
        except (json.JSONDecodeError, OSError):
            pass
    return rec


def load_scraped_records(data_dir: str) -> list[dict[str, Any]]:
    """Load all scraped records from data/raw/ subdirectories."""
    records = []
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning("Data dir %s does not exist", data_dir)
        return records

    for json_file in data_path.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict):
                        rec["_source_file"] = str(json_file)
                        records.append(rec)
            elif isinstance(data, dict):
                if "records" in data:
                    for rec in data["records"]:
                        if isinstance(rec, dict):
                            rec["_source_file"] = str(json_file)
                            records.append(rec)
                else:
                    data["_source_file"] = str(json_file)
                    records.append(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Skipping %s: %s", json_file, e)

    for jl_path in data_path.rglob("*.jsonl"):
        if "checkpoint" in jl_path.name.lower():
            continue
        try:
            for line in jl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    rec["_source_file"] = str(jl_path)
                    records.append(rec)
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Skipping %s: %s", jl_path, e)

    logger.info("Loaded %d raw records from %s", len(records), data_dir)
    return records


def record_to_sft_examples(record: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a single record to one or more SFT (supervised fine-tuning) examples.

    Each example is a dict with 'instruction', 'input', 'output' keys
    compatible with Alpaca-style training.
    """
    examples = []

    # Task 1: Extraction
    doc_text = _build_document_text(record)
    if doc_text:
        expected_output = _build_extraction_output(record)
        examples.append({
            "instruction": EXTRACTION_TEMPLATE.split("### Document:")[0].strip(),
            "input": doc_text,
            "output": json.dumps(expected_output, ensure_ascii=False),
        })

    # Task 2: Risk assessment (if record has risk indicators)
    if record.get("has_mortgage") or record.get("property_related"):
        risk_input = _build_risk_input(record)
        risk_output = _build_risk_output(record)
        examples.append({
            "instruction": RISK_ASSESSMENT_TEMPLATE.split("### Record:")[0].strip(),
            "input": risk_input,
            "output": json.dumps(risk_output, ensure_ascii=False),
        })

    return examples


def _build_document_text(record: dict[str, Any]) -> str:
    """Build a realistic document text from a structured record."""
    if record.get("doc_type") == "court_judgment" or (
        record.get("tid") is not None
        and (record.get("headline") or record.get("full_text_plain"))
    ):
        lines: list[str] = []
        if record.get("title"):
            lines.append(f"Title: {record['title']}")
        if record.get("docsource"):
            lines.append(f"Court: {record['docsource']}")
        if record.get("publishdate"):
            lines.append(f"Date: {record['publishdate']}")
        body = record.get("full_text_plain") or record.get("headline") or ""
        if body:
            lines.append(f"Judgment excerpt:\n{body}")
        return "\n".join(lines)

    parts = []
    if record.get("document_type") or record.get("doc_type"):
        dt = record.get("document_type") or record.get("doc_type", "Unknown")
        parts.append(f"Document Type: {dt}")
    if record.get("document_number"):
        parts.append(f"Document Number: {record['document_number']}")
    if record.get("registration_date"):
        parts.append(f"Registration Date: {record['registration_date']}")
    if record.get("sro"):
        parts.append(f"Sub-Registrar Office: {record['sro']}")

    sellers = record.get("seller_names") or record.get("executant")
    if sellers:
        if isinstance(sellers, list):
            sellers = ", ".join(sellers)
        parts.append(f"Seller/Executant: {sellers}")

    buyers = record.get("buyer_names") or record.get("claimant")
    if buyers:
        if isinstance(buyers, list):
            buyers = ", ".join(buyers)
        parts.append(f"Buyer/Claimant: {buyers}")

    if record.get("survey_number") or record.get("survey_numbers"):
        sn = record.get("survey_number") or ", ".join(
            record.get("survey_numbers", []))
        parts.append(f"Survey Number: {sn}")
    if record.get("property_description"):
        parts.append(f"Property: {record['property_description']}")
    if record.get("consideration") or record.get("consideration_amount"):
        amt = record.get("consideration") or record.get("consideration_amount")
        parts.append(f"Consideration: Rs. {amt}")

    return "\n".join(parts) if parts else ""


def _build_extraction_output(record: dict[str, Any]) -> dict[str, Any]:
    doc_t = record.get("doc_type") or record.get("document_type", "unknown")
    mentions_dispute = record.get("mentions_dispute")
    if mentions_dispute is None:
        mentions_dispute = doc_t == "court_judgment" or bool(
            record.get("property_related", False)
        )
    return {
        "doc_type": doc_t,
        "registration_date": record.get("registration_date") or record.get("publishdate"),
        "seller_names": record.get("seller_names", []),
        "buyer_names": record.get("buyer_names", []),
        "parcel_ids": record.get("survey_numbers", []) or ([record["survey_number"]] if record.get("survey_number") else []),
        "locality": record.get("district") or record.get("locality") or record.get("docsource"),
        "consideration_amount": record.get("consideration") or record.get("consideration_amount"),
        "mentions_dispute": mentions_dispute,
        "mentions_encumbrance": record.get("has_mortgage", False),
    }


def _build_risk_input(record: dict[str, Any]) -> str:
    parts = [_build_document_text(record)]
    if record.get("status"):
        parts.append(f"Case Status: {record['status']}")
    if record.get("subject"):
        parts.append(f"Case Subject: {record['subject']}")
    if record.get("has_mortgage"):
        parts.append(
            f"Mortgage to: {record.get('mortgage_to', 'Unknown lender')}")
    return "\n".join(parts)


def _build_risk_output(record: dict[str, Any]) -> dict[str, Any]:
    risk = "LOW"
    factors = []
    recommendations = []
    blocks = False

    if record.get("has_mortgage"):
        risk = "HIGH"
        factors.append("Active mortgage/charge on property")
        recommendations.append(
            "Obtain NOC/release deed from mortgagee before purchase")
        blocks = True
    if record.get("status") == "Pending":
        risk = "HIGH"
        factors.append("Pending litigation involving property")
        recommendations.append(
            "Obtain legal opinion on case merits and likely timeline")
        blocks = True
    if record.get("property_related") and record.get("status") == "Disposed":
        risk = "MEDIUM"
        factors.append("Past litigation involving property (now disposed)")
        recommendations.append(
            "Verify final court order and ensure compliance")

    return {
        "risk_level": risk,
        "risk_factors": factors,
        "recommendations": recommendations,
        "blocks_transaction": blocks,
    }


def _write_jsonl(path: Path, data: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_parquet(path: Path, data: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "Install pyarrow (pip install pyarrow) for Parquet export") from e
    if not data:
        return
    table = pa.Table.from_pylist(data)
    pq.write_table(table, path, compression="zstd")


def prepare_dataset(
    data_dir: str,
    output_dir: str,
    val_split: float = 0.1,
    max_seq_len: int = 8192,
    *,
    manifest_path: str | None = None,
    output_format: OutputFormat = "jsonl",
) -> dict[str, int]:
    """Full pipeline: load → convert → split → write JSONL and/or Parquet."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = load_scraped_records(data_dir)
    if manifest_path:
        for row in load_curated_manifest(manifest_path):
            records.append(manifest_row_to_record(row))

    all_examples: list[dict[str, Any]] = []
    for rec in records:
        examples = record_to_sft_examples(rec)
        for ex in examples:
            # Filter by approximate token count (chars/4 approx tokens)
            total_len = len(ex.get("instruction", "")) + \
                len(ex.get("input", "")) + len(ex.get("output", ""))
            if total_len // 4 <= max_seq_len:
                row: dict[str, Any] = dict(ex)
                for meta_key in (
                    "source",
                    "language",
                    "doc_type",
                    "is_scanned",
                    "ocr_engine",
                    "human_verified",
                    "is_synthetic",
                    "difficulty",
                ):
                    if rec.get(meta_key) is not None:
                        row[meta_key] = rec[meta_key]
                all_examples.append(row)

    random.shuffle(all_examples)

    split_idx = max(1, int(len(all_examples) * (1 - val_split)))
    train = all_examples[:split_idx]
    val = all_examples[split_idx:]

    train_path = output_path / "train.jsonl"
    val_path = output_path / "val.jsonl"
    train_parquet = output_path / "train.parquet"
    val_parquet = output_path / "val.parquet"

    if output_format in ("jsonl", "both"):
        _write_jsonl(train_path, train)
        _write_jsonl(val_path, val)
    if output_format in ("parquet", "both"):
        if train:
            _write_parquet(train_parquet, [dict(x) for x in train])
        if val:
            _write_parquet(val_parquet, [dict(x) for x in val])

    # Also write a config for the training script
    config = {
        "train_file": str(train_path),
        "val_file": str(val_path),
        "train_count": len(train),
        "val_count": len(val),
        "max_seq_len": max_seq_len,
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "notes": (
            "Fine-tune on AMD MI300X with ROCm + PyTorch + PEFT. "
            "Use: torchrun --nproc_per_node=1 train.py --config training_config.json"
        ),
    }
    config_path = output_path / "training_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    stats = {"train": len(train), "val": len(
        val), "total_records": len(records)}
    logger.info("Dataset prepared: %s", stats)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(
        description="Prepare training data from scraped records")
    ap.add_argument("--data-dir", default="data/raw",
                    help="Raw scraped data directory")
    ap.add_argument("--output-dir", default="data/training",
                    help="Output directory")
    ap.add_argument("--manifest", default=None,
                    help="Optional JSONL manifest of curated uploads")
    ap.add_argument(
        "--format",
        choices=("jsonl", "parquet", "both"),
        default="jsonl",
        help="Output format",
    )
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    args = ap.parse_args()

    stats = prepare_dataset(
        args.data_dir,
        args.output_dir,
        args.val_split,
        args.max_seq_len,
        manifest_path=args.manifest,
        output_format=args.format,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
