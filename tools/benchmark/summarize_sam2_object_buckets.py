#!/usr/bin/env python3
"""Compare matched SAM2 legacy and object-bucket latency runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "object_count",
    "samples",
    "videos",
    "legacy_ms_per_frame",
    "bucket_ms_per_frame",
    "latency_reduction_pct",
    "legacy_fps",
    "bucket_fps",
    "fps_gain_pct",
    "legacy_relative_latency",
    "bucket_relative_latency",
    "legacy_peak_memory_mb",
    "bucket_peak_memory_mb",
    "memory_delta_mb",
    "bucket_target_pass",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", required=True, type=Path)
    parser.add_argument("--bucket-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_aggregate(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            int(row["object_count"]): row
            for row in csv.DictReader(handle)
        }


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def matched_run_metadata(
    legacy: dict[str, Any],
    bucket: dict[str, Any],
) -> tuple[bool, list[str]]:
    mismatches = []
    for key in (
        "git_commit",
        "model_kind",
        "checkpoint",
        "sam2_checkpoint",
        "prompt_kind",
        "object_counts",
        "videos",
        "repetitions",
        "seed",
    ):
        if legacy.get(key) != bucket.get(key):
            mismatches.append(key)
    return not mismatches, mismatches


def comparison_rows(
    legacy: dict[int, dict[str, str]],
    bucket: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    if set(legacy) != set(bucket):
        raise ValueError(
            f"Object counts differ: legacy={sorted(legacy)}, bucket={sorted(bucket)}"
        )
    rows = []
    for object_count in sorted(legacy):
        legacy_row = legacy[object_count]
        bucket_row = bucket[object_count]
        legacy_ms = as_float(legacy_row, "median_propagation_ms_per_frame")
        bucket_ms = as_float(bucket_row, "median_propagation_ms_per_frame")
        legacy_fps = as_float(legacy_row, "median_propagation_fps")
        bucket_fps = as_float(bucket_row, "median_propagation_fps")
        legacy_memory = as_float(legacy_row, "median_peak_memory_mb")
        bucket_memory = as_float(bucket_row, "median_peak_memory_mb")
        rows.append(
            {
                "object_count": object_count,
                "samples": bucket_row["samples"],
                "videos": bucket_row["videos"],
                "legacy_ms_per_frame": legacy_ms,
                "bucket_ms_per_frame": bucket_ms,
                "latency_reduction_pct": 100.0 * (1.0 - bucket_ms / legacy_ms),
                "legacy_fps": legacy_fps,
                "bucket_fps": bucket_fps,
                "fps_gain_pct": 100.0 * (bucket_fps / legacy_fps - 1.0),
                "legacy_relative_latency": as_float(
                    legacy_row, "relative_latency_vs_1"
                ),
                "bucket_relative_latency": as_float(
                    bucket_row, "relative_latency_vs_1"
                ),
                "legacy_peak_memory_mb": legacy_memory,
                "bucket_peak_memory_mb": bucket_memory,
                "memory_delta_mb": bucket_memory - legacy_memory,
                "bucket_target_pass": bucket_row.get("target_pass", ""),
            }
        )
    return rows


def decision(
    legacy_summary: dict[str, Any],
    bucket_summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    lineage_match, lineage_mismatches = matched_run_metadata(
        legacy_summary, bucket_summary
    )
    verification = bucket_summary.get("bucket_verification") or {}
    bucket_size = int(bucket_summary.get("bucket_size") or 0)
    by_count = {int(row["object_count"]): row for row in rows}
    promotion_count = bucket_size if bucket_size in by_count else max(by_count)
    promotion = by_count[promotion_count]
    checks = {
        "lineage_match": lineage_match,
        "execution_modes_match_request": (
            legacy_summary.get("execution_mode") == "legacy"
            and bucket_summary.get("execution_mode") == "bucket"
        ),
        "binary_mask_equivalence": verification.get("pass") is True,
        "promotion_count_faster": float(promotion["fps_gain_pct"]) > 0.0,
        "promotion_count_target_pass": str(promotion["bucket_target_pass"]) == "1",
        "sample_counts_match": all(
            legacy_summary["aggregate"][index]["samples"]
            == bucket_summary["aggregate"][index]["samples"]
            for index in range(len(rows))
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "decision": "promote" if not failed else "reject",
        "promotion_object_count": promotion_count,
        "checks": checks,
        "failed_checks": failed,
        "lineage_mismatches": lineage_mismatches,
        "bucket_verification": verification,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    legacy_dir: Path,
    bucket_dir: Path,
) -> str:
    lines = [
        "# SAM2 object-bucket comparison",
        "",
        f"- Decision: **{result['decision'].upper()}**",
        f"- Legacy: `{legacy_dir}`",
        f"- Bucket: `{bucket_dir}`",
        (
            "- Binary-mask agreement: "
            f"{result['bucket_verification'].get('binary_mask_agreement', 'missing')}"
        ),
        "",
        "| Objects | Legacy FPS | Bucket FPS | FPS gain | Legacy ms | Bucket ms | "
        "Latency reduction | Memory delta MB | Target |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['object_count']} | {row['legacy_fps']:.2f} | "
            f"{row['bucket_fps']:.2f} | {row['fps_gain_pct']:+.1f}% | "
            f"{row['legacy_ms_per_frame']:.2f} | "
            f"{row['bucket_ms_per_frame']:.2f} | "
            f"{row['latency_reduction_pct']:+.1f}% | "
            f"{row['memory_delta_mb']:+.0f} | "
            f"{row['bucket_target_pass']} |"
        )
    if result["failed_checks"]:
        lines.extend(
            [
                "",
                "Failed checks: " + ", ".join(result["failed_checks"]),
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    legacy_summary = read_json(args.legacy_dir / "summary.json")
    bucket_summary = read_json(args.bucket_dir / "summary.json")
    rows = comparison_rows(
        read_aggregate(args.legacy_dir / "aggregate.csv"),
        read_aggregate(args.bucket_dir / "aggregate.csv"),
    )
    result = decision(legacy_summary, bucket_summary, rows)
    result.update(
        {
            "legacy_dir": str(args.legacy_dir),
            "bucket_dir": str(args.bucket_dir),
            "rows": rows,
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "comparison.csv", rows)
    report = markdown_report(rows, result, args.legacy_dir, args.bucket_dir)
    (args.out_dir / "comparison.md").write_text(report, encoding="utf-8")
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report, end="")
    print(f"Summary: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
