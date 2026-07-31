import csv
import json

from tools.benchmark.summarize_sam2_multiplex_screen import build_rows


def write_aggregate(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "object_count": count,
            "median_propagation_fps": fps,
            "median_propagation_ms_per_frame": 1000 / fps,
        }
        for count, fps in values.items()
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_multiplex_screen_promotes_quality_preserving_speedup(tmp_path):
    variants = ("fast", "accurate")
    run_root = tmp_path / "runs"
    reference = tmp_path / "reference"
    write_aggregate(reference / "aggregate.csv", {1: 70, 8: 20})
    for variant, jf, n8 in (
        ("fast", 69, 40),
        ("accurate", 70, 30),
    ):
        variant_dir = run_root / variant
        run_dir = variant_dir / "main"
        latency_dir = (
            run_dir / "multiobject_latency" / "point_n1-2-4-8"
        )
        latency_dir.mkdir(parents=True)
        (variant_dir / ".screen_complete").touch()
        (run_dir / "gate_status.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "metrics": {"J&F": jf, "mIoU": 0.84, "AP": 0.71},
                    "reference": {
                        "J&F": 70,
                        "mIoU": 0.84,
                        "AP": 0.71,
                    },
                }
            ),
            encoding="utf-8",
        )
        write_aggregate(
            latency_dir / "aggregate.csv",
            {1: 60, 2: 50, 4: 45, 8: n8},
        )
        (latency_dir / "summary.json").write_text(
            json.dumps(
                {"bucket_verification": {"min_mask_iou": 0.98}}
            ),
            encoding="utf-8",
        )

    rows = build_rows(
        run_root,
        reference,
        variants,
        gate_videos=32,
        min_quality_retention=0.95,
        min_learned_mask_iou=0.95,
    )

    assert [row["promote"] for row in rows] == [1, 1]
    assert [row["pareto"] for row in rows] == [1, 1]
    assert rows[0]["rank"] == 1
    assert rows[0]["n8_gain_pct"] == 100
