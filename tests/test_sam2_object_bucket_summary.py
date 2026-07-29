import csv
import json

import pytest

from tools.benchmark.summarize_sam2_object_buckets import (
    comparison_rows,
    decision,
    main,
)


def aggregate_row(object_count, fps, milliseconds, relative, target_pass):
    return {
        "object_count": str(object_count),
        "samples": "4",
        "videos": "2",
        "median_propagation_ms_per_frame": str(milliseconds),
        "median_propagation_fps": str(fps),
        "relative_latency_vs_1": str(relative),
        "median_peak_memory_mb": "1000",
        "target_pass": target_pass,
    }


def summary(mode, aggregate, verification=None):
    return {
        "execution_mode": mode,
        "bucket_size": 4 if mode == "bucket" else None,
        "checkpoint": "/data/model.pt",
        "sam2_checkpoint": "/data/sam2.pt",
        "prompt_kind": "point",
        "videos": ["a", "b"],
        "aggregate": aggregate,
        "bucket_verification": verification,
    }


def test_bucket_summary_promotes_matched_faster_run():
    legacy = {
        1: aggregate_row(1, 50, 20, 1.0, ""),
        4: aggregate_row(4, 25, 40, 2.0, "0"),
    }
    bucket = {
        1: aggregate_row(1, 50, 20, 1.0, ""),
        4: aggregate_row(4, 40, 25, 1.25, "1"),
    }
    rows = comparison_rows(legacy, bucket)
    result = decision(
        summary("legacy", list(legacy.values())),
        summary(
            "bucket",
            list(bucket.values()),
            {"pass": True, "binary_mask_agreement": 1.0},
        ),
        rows,
    )

    assert rows[1]["fps_gain_pct"] == pytest.approx(60.0)
    assert rows[1]["latency_reduction_pct"] == pytest.approx(37.5)
    assert result["decision"] == "promote"


def test_bucket_summary_rejects_missing_equivalence():
    legacy = {
        1: aggregate_row(1, 50, 20, 1.0, ""),
        4: aggregate_row(4, 25, 40, 2.0, "0"),
    }
    bucket = {
        1: aggregate_row(1, 50, 20, 1.0, ""),
        4: aggregate_row(4, 40, 25, 1.25, "1"),
    }
    rows = comparison_rows(legacy, bucket)
    result = decision(
        summary("legacy", list(legacy.values())),
        summary("bucket", list(bucket.values()), None),
        rows,
    )

    assert result["decision"] == "reject"
    assert "binary_mask_equivalence" in result["failed_checks"]


def test_bucket_summary_cli_writes_csv_markdown_and_json(
    tmp_path,
    monkeypatch,
):
    legacy_dir = tmp_path / "legacy"
    bucket_dir = tmp_path / "bucket"
    out_dir = tmp_path / "comparison"
    legacy_dir.mkdir()
    bucket_dir.mkdir()
    legacy_rows = [
        aggregate_row(1, 50, 20, 1.0, ""),
        aggregate_row(4, 25, 40, 2.0, "0"),
    ]
    bucket_rows = [
        aggregate_row(1, 50, 20, 1.0, ""),
        aggregate_row(4, 40, 25, 1.25, "1"),
    ]
    for path, rows in (
        (legacy_dir / "aggregate.csv", legacy_rows),
        (bucket_dir / "aggregate.csv", bucket_rows),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (legacy_dir / "summary.json").write_text(
        json.dumps(summary("legacy", legacy_rows)),
        encoding="utf-8",
    )
    (bucket_dir / "summary.json").write_text(
        json.dumps(
            summary(
                "bucket",
                bucket_rows,
                {"pass": True, "binary_mask_agreement": 1.0},
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_sam2_object_buckets.py",
            "--legacy-dir",
            str(legacy_dir),
            "--bucket-dir",
            str(bucket_dir),
            "--out-dir",
            str(out_dir),
        ],
    )

    main()

    assert (out_dir / "comparison.csv").is_file()
    assert "PROMOTE" in (out_dir / "comparison.md").read_text(encoding="utf-8")
    assert json.loads(
        (out_dir / "summary.json").read_text(encoding="utf-8")
    )["decision"] == "promote"
