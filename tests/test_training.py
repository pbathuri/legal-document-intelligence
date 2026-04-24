"""Test training data preparation pipeline."""
import json
import tempfile
from pathlib import Path

from legal_intel.training.prepare import (
    load_scraped_records,
    prepare_dataset,
    record_to_sft_examples,
)


def test_record_to_sft():
    record = {
        "doc_type": "sale_deed",
        "registration_date": "2024-03-15",
        "seller_names": ["Alice"],
        "buyer_names": ["Bob"],
        "survey_numbers": ["SY123/A"],
        "district": "Hyderabad",
        "consideration_amount": "5000000",
    }
    examples = record_to_sft_examples(record)
    assert len(examples) >= 1
    assert "instruction" in examples[0]
    assert "input" in examples[0]
    assert "output" in examples[0]


def test_load_scraped_records_includes_jsonl(tmp_path):
    bulk = tmp_path / "bulk"
    bulk.mkdir()
    (bulk / "kanoon_corpus.jsonl").write_text(
        json.dumps({"tid": 1, "title": "x", "headline": "property"}) + "\n",
        encoding="utf-8",
    )
    recs = load_scraped_records(str(tmp_path))
    assert len(recs) == 1
    assert recs[0]["tid"] == 1


def test_prepare_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "output"
        stats = prepare_dataset(
            data_dir=str(Path(tmpdir) / "nonexistent"),
            output_dir=str(out_dir),
        )
        assert stats["total_records"] == 0
