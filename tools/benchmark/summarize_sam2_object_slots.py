#!/usr/bin/env python3
"""Rank learned SAM2 object-slot runs against the selected runtime baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VARIANTS = (
    "MX1_slot4_decoder_kd_3ep",
    "MX2_slot8_decoder_kd_3ep",
    "MX3_slot4_sharedkv_kd_3ep",
    "MX4_slot8_sharedkv_kd_3ep",
)

FIELDS = (
    "rank",
    "variant",
    "status",
    "val_jf",
    "test_jf",
    "val_retention",
    "test_retention",
    "n1_fps",
    "n1_fps_retention",
    "n8_fps",
    "n8_fps_gain_pct",
    "n8_ms_per_frame",
    "n8_peak_memory_mb",
    "verification_min_mask_iou",
    "quality_gate",
    "learned_mask_gate",
    "n1_gate",
    "n8_gate",
    "promote",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--reference-latency-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reference-val-jf", type=float, default=72.4)
    parser.add_argument("--reference-test-jf", type=float, default=74.7)
    parser.add_argument("--min-quality-retention", type=float, default=0.95)
    parser.add_argument("--min-learned-mask-iou", type=float, default=0.95)
    parser.add_argument("--min-n1-fps-retention", type=float, default=0.95)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated variant names to include.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def optional_float(value: Any) -> float | None:
    try:
        return float(value) if value not in ("", None) else None
    except (TypeError, ValueError):
        return None


def latency_by_count(path: Path) -> dict[int, dict[str, str]]:
    return {int(row["object_count"]): row for row in read_csv(path)}


def build_rows(
    run_root: Path,
    reference_latency_dir: Path,
    reference_val_jf: float,
    reference_test_jf: float,
    min_quality_retention: float,
    min_learned_mask_iou: float,
    min_n1_fps_retention: float,
    variants: tuple[str, ...] = VARIANTS,
) -> list[dict[str, Any]]:
    central_rows = {
        row["variant"]: row for row in read_csv(run_root / "summary.csv")
    }
    reference = latency_by_count(reference_latency_dir / "aggregate.csv")
    reference_n1 = float(reference[1]["median_propagation_fps"])
    reference_n8 = float(reference[8]["median_propagation_fps"])
    rows = []
    for variant in variants:
        central = central_rows.get(variant, {})
        latency_dir = (
            run_root
            / variant
            / "main"
            / "multiobject_latency"
            / "point_n1-2-4-8"
        )
        latency_path = latency_dir / "aggregate.csv"
        latency = latency_by_count(latency_path) if latency_path.is_file() else {}
        one = latency.get(1, {})
        eight = latency.get(8, {})
        val_jf = optional_float(central.get("val_J&F"))
        test_jf = optional_float(central.get("test_J&F"))
        n1_fps = optional_float(one.get("median_propagation_fps"))
        n8_fps = optional_float(eight.get("median_propagation_fps"))
        val_retention = (
            val_jf / reference_val_jf if val_jf is not None else None
        )
        test_retention = (
            test_jf / reference_test_jf if test_jf is not None else None
        )
        n1_retention = n1_fps / reference_n1 if n1_fps is not None else None
        n8_gain = (
            100.0 * (n8_fps / reference_n8 - 1.0)
            if n8_fps is not None
            else None
        )
        quality_gate = (
            val_retention is not None
            and test_retention is not None
            and min(val_retention, test_retention) >= min_quality_retention
        )
        n1_gate = (
            n1_retention is not None
            and n1_retention >= min_n1_fps_retention
        )
        n8_gate = n8_fps is not None and n8_fps > reference_n8
        verification = read_json(latency_dir / "summary.json").get(
            "bucket_verification"
        ) or {}
        verification_min_iou = optional_float(
            verification.get("min_mask_iou")
        )
        learned_mask_gate = (
            verification_min_iou is not None
            and verification_min_iou >= min_learned_mask_iou
        )
        rows.append(
            {
                "rank": "",
                "variant": variant,
                "status": central.get("status", "not_recorded"),
                "val_jf": val_jf,
                "test_jf": test_jf,
                "val_retention": val_retention,
                "test_retention": test_retention,
                "n1_fps": n1_fps,
                "n1_fps_retention": n1_retention,
                "n8_fps": n8_fps,
                "n8_fps_gain_pct": n8_gain,
                "n8_ms_per_frame": optional_float(
                    eight.get("median_propagation_ms_per_frame")
                ),
                "n8_peak_memory_mb": optional_float(
                    eight.get("median_peak_memory_mb")
                ),
                "verification_min_mask_iou": verification_min_iou,
                "quality_gate": int(quality_gate),
                "learned_mask_gate": int(learned_mask_gate),
                "n1_gate": int(n1_gate),
                "n8_gate": int(n8_gate),
                "promote": int(
                    quality_gate
                    and learned_mask_gate
                    and n1_gate
                    and n8_gate
                ),
            }
        )
    promoted = sorted(
        (row for row in rows if row["promote"]),
        key=lambda row: row["n8_fps"],
        reverse=True,
    )
    for rank, row in enumerate(promoted, 1):
        row["rank"] = rank
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def show(value: Any, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def markdown(
    rows: list[dict[str, Any]],
    reference_latency_dir: Path,
    reference_val_jf: float,
    reference_test_jf: float,
    min_quality_retention: float,
    min_learned_mask_iou: float,
    min_n1_fps_retention: float,
) -> str:
    lines = [
        "# SAM2 learned object-slot results",
        "",
        f"- Runtime reference: `{reference_latency_dir}`",
        f"- Quality reference: val J&F {reference_val_jf:.1f}, test J&F {reference_test_jf:.1f}",
        f"- Gates: full quality retention ≥ {min_quality_retention:.0%}; learned-path mask IoU ≥ {min_learned_mask_iou:.2f}; N=1 FPS retention ≥ {min_n1_fps_retention:.0%}; N=8 faster than reference",
        "",
        "| Rank | Variant | Status | Val J&F | Test J&F | Min quality | Learned mask IoU | N1 FPS | N1 retention | N8 FPS | N8 gain | N8 ms | Promote |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        quality_values = [
            value
            for value in (row["val_retention"], row["test_retention"])
            if value is not None
        ]
        minimum_quality = min(quality_values) if quality_values else None
        lines.append(
            f"| {row['rank']} | {row['variant']} | {row['status']} | "
            f"{show(row['val_jf'])} | {show(row['test_jf'])} | "
            f"{show(minimum_quality, 3)} | "
            f"{show(row['verification_min_mask_iou'], 3)} | "
            f"{show(row['n1_fps'])} | "
            f"{show(row['n1_fps_retention'], 3)} | {show(row['n8_fps'])} | "
            f"{show(row['n8_fps_gain_pct'], 1)}% | "
            f"{show(row['n8_ms_per_frame'])} | {row['promote']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    variants = tuple(
        variant.strip()
        for variant in args.variants.split(",")
        if variant.strip()
    )
    if not variants:
        raise SystemExit("--variants must contain at least one variant")
    rows = build_rows(
        args.run_root,
        args.reference_latency_dir,
        args.reference_val_jf,
        args.reference_test_jf,
        args.min_quality_retention,
        args.min_learned_mask_iou,
        args.min_n1_fps_retention,
        variants,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "object_slot_results.csv", rows)
    report = markdown(
        rows,
        args.reference_latency_dir,
        args.reference_val_jf,
        args.reference_test_jf,
        args.min_quality_retention,
        args.min_learned_mask_iou,
        args.min_n1_fps_retention,
    )
    (args.out_dir / "object_slot_results.md").write_text(
        report, encoding="utf-8"
    )
    payload = {
        "reference_latency_dir": str(args.reference_latency_dir),
        "reference_val_jf": args.reference_val_jf,
        "reference_test_jf": args.reference_test_jf,
        "min_quality_retention": args.min_quality_retention,
        "min_learned_mask_iou": args.min_learned_mask_iou,
        "min_n1_fps_retention": args.min_n1_fps_retention,
        "rows": rows,
    }
    (args.out_dir / "object_slot_results.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(report, end="")


if __name__ == "__main__":
    main()
