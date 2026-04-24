"""Parquet export for training prep (requires pyarrow)."""
import json

import pytest

from legal_intel.training.prepare import prepare_dataset


pytest.importorskip("pyarrow")


def test_prepare_parquet_and_manifest(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "doc_type": "sale_deed",
                "registration_date": "2024-01-01",
                "seller_names": ["A"],
                "buyer_names": ["B"],
                "survey_numbers": ["1/A"],
                "language": "en",
                "is_scanned": True,
                "source": "curated_upload",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    stats = prepare_dataset(
        data_dir=str(tmp_path / "nodir"),
        output_dir=str(out),
        manifest_path=str(manifest),
        output_format="both",
    )
    assert stats["train"] >= 1
    assert (out / "train.parquet").exists()
