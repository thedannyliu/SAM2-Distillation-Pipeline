import csv
import json

from tools.benchmark.summarize_sam2_object_slots import (
    VARIANTS,
    build_rows,
    main,
)


def write_aggregate(path, n1_fps, n8_fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "object_count": 1,
            "median_propagation_fps": n1_fps,
            "median_propagation_ms_per_frame": 1000 / n1_fps,
            "median_peak_memory_mb": 9000,
        },
        {
            "object_count": 8,
            "median_propagation_fps": n8_fps,
            "median_propagation_ms_per_frame": 1000 / n8_fps,
            "median_peak_memory_mb": 10000,
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_object_slot_summary_applies_quality_and_latency_gates(tmp_path):
    run_root = tmp_path / "runs"
    run_root.mkdir()
    with (run_root / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("variant", "status", "val_J&F", "test_J&F"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "variant": VARIANTS[0],
                "status": "complete",
                "val_J&F": 70,
                "test_J&F": 72,
            }
        )
    reference = tmp_path / "reference"
    write_aggregate(reference / "aggregate.csv", 70, 20)
    latency_dir = (
        run_root
        / VARIANTS[0]
        / "main"
        / "multiobject_latency"
        / "point_n1-2-4-8"
    )
    write_aggregate(latency_dir / "aggregate.csv", 68, 30)
    (latency_dir / "summary.json").write_text(
        json.dumps({"bucket_verification": {"min_mask_iou": 0.96}}),
        encoding="utf-8",
    )

    rows = build_rows(
        run_root,
        reference,
        72.4,
        74.7,
        0.95,
        0.95,
        0.95,
    )

    assert rows[0]["quality_gate"] == 1
    assert rows[0]["n1_gate"] == 1
    assert rows[0]["n8_gate"] == 1
    assert rows[0]["verification_min_mask_iou"] == 0.96
    assert rows[0]["learned_mask_gate"] == 1
    assert rows[0]["promote"] == 1
    assert rows[0]["rank"] == 1
    assert rows[1]["promote"] == 0


def test_object_slot_summary_cli_writes_report(tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    run_root.mkdir()
    with (run_root / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("variant", "status", "val_J&F", "test_J&F"),
        )
        writer.writeheader()
    reference = tmp_path / "reference"
    write_aggregate(reference / "aggregate.csv", 70, 20)
    out_dir = tmp_path / "summary"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_sam2_object_slots.py",
            "--run-root",
            str(run_root),
            "--reference-latency-dir",
            str(reference),
            "--out-dir",
            str(out_dir),
        ],
    )

    main()

    assert (out_dir / "object_slot_results.csv").is_file()
    assert (out_dir / "object_slot_results.json").is_file()
    assert "95%" in (
        out_dir / "object_slot_results.md"
    ).read_text(encoding="utf-8")
