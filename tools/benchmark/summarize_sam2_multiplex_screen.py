#!/usr/bin/env python3
"""Summarize the fixed-cohort SAM2 multiplex overnight screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "rank",
    "pareto",
    "variant",
    "status",
    "gate_status",
    "mini_val_jf",
    "jf_retention",
    "mini_val_miou",
    "mini_val_ap",
    "verification_min_mask_iou",
    "n1_fps",
    "n1_retention",
    "n2_fps",
    "n4_fps",
    "n8_fps",
    "n8_gain_pct",
    "n8_ms_per_frame",
    "quality_gate",
    "learned_mask_gate",
    "n8_gate",
    "promote",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--reference-latency-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--gate-videos", type=int, default=32)
    parser.add_argument("--min-quality-retention", type=float, default=0.95)
    parser.add_argument("--min-learned-mask-iou", type=float, default=0.95)
    parser.add_argument("--max-promotions", type=int, default=4)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def optional_float(value: Any) -> float | None:
    try:
        return float(value) if value not in ("", None) else None
    except (TypeError, ValueError):
        return None


def latency_by_count(path: Path) -> dict[int, dict[str, str]]:
    return {
        int(row["object_count"]): row
        for row in read_csv(path)
    }


def is_pareto(row: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    quality = row["jf_retention"]
    speed = row["n8_fps"]
    if quality is None or speed is None:
        return False
    for other in rows:
        other_quality = other["jf_retention"]
        other_speed = other["n8_fps"]
        if other_quality is None or other_speed is None:
            continue
        weakly_better = (
            other_quality >= quality and other_speed >= speed
        )
        strictly_better = (
            other_quality > quality or other_speed > speed
        )
        if weakly_better and strictly_better:
            return False
    return True


def build_rows(
    run_root: Path,
    reference_latency_dir: Path,
    variants: tuple[str, ...],
    gate_videos: int,
    min_quality_retention: float,
    min_learned_mask_iou: float,
) -> list[dict[str, Any]]:
    reference = latency_by_count(
        reference_latency_dir / "aggregate.csv"
    )
    reference_n1 = float(reference[1]["median_propagation_fps"])
    reference_n8 = float(reference[8]["median_propagation_fps"])
    rows = []
    for variant in variants:
        variant_dir = run_root / variant
        run_dir = variant_dir / "main"
        gate = read_json(run_dir / "gate_status.json")
        metrics = gate.get("metrics") or {}
        reference_metrics = gate.get("reference") or {}
        jf = optional_float(metrics.get("J&F"))
        reference_jf = optional_float(reference_metrics.get("J&F"))
        retention = (
            jf / reference_jf
            if jf is not None and reference_jf
            else None
        )
        latency_dir = (
            run_dir / "multiobject_latency" / "point_n1-2-4-8"
        )
        latency = latency_by_count(latency_dir / "aggregate.csv")
        fps = {
            count: optional_float(
                latency.get(count, {}).get("median_propagation_fps")
            )
            for count in (1, 2, 4, 8)
        }
        n1_retention = (
            fps[1] / reference_n1 if fps[1] is not None else None
        )
        n8_gain = (
            100.0 * (fps[8] / reference_n8 - 1)
            if fps[8] is not None
            else None
        )
        verification = read_json(latency_dir / "summary.json").get(
            "bucket_verification"
        ) or {}
        min_mask_iou = optional_float(
            verification.get("min_mask_iou")
        )
        quality_gate = (
            gate.get("status") == "pass"
            and retention is not None
            and retention >= min_quality_retention
        )
        learned_mask_gate = (
            min_mask_iou is not None
            and min_mask_iou >= min_learned_mask_iou
        )
        n8_gate = fps[8] is not None and fps[8] > reference_n8
        complete = (
            (variant_dir / ".screen_complete").is_file()
            and gate
            and bool(latency)
        )
        rows.append(
            {
                "rank": "",
                "pareto": 0,
                "variant": variant,
                "status": "complete" if complete else "pending",
                "gate_status": gate.get("status", "pending"),
                "mini_val_jf": jf,
                "jf_retention": retention,
                "mini_val_miou": optional_float(metrics.get("mIoU")),
                "mini_val_ap": optional_float(metrics.get("AP")),
                "verification_min_mask_iou": min_mask_iou,
                "n1_fps": fps[1],
                "n1_retention": n1_retention,
                "n2_fps": fps[2],
                "n4_fps": fps[4],
                "n8_fps": fps[8],
                "n8_gain_pct": n8_gain,
                "n8_ms_per_frame": optional_float(
                    latency.get(8, {}).get(
                        "median_propagation_ms_per_frame"
                    )
                ),
                "quality_gate": int(quality_gate),
                "learned_mask_gate": int(learned_mask_gate),
                "n8_gate": int(n8_gate),
                "promote": int(
                    quality_gate and learned_mask_gate and n8_gate
                ),
            }
        )
    for row in rows:
        row["pareto"] = int(is_pareto(row, rows))
    promoted = sorted(
        (row for row in rows if row["promote"]),
        key=lambda row: (
            not row["pareto"],
            -row["n8_fps"],
            -row["jf_retention"],
        ),
    )
    for rank, row in enumerate(promoted, 1):
        row["rank"] = rank
    return rows


def show(value: Any, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def markdown(
    rows: list[dict[str, Any]],
    reference_latency_dir: Path,
    gate_videos: int,
    min_quality_retention: float,
    min_learned_mask_iou: float,
) -> str:
    lines = [
        "# SAM2 multiplex overnight screen",
        "",
        f"- Runtime reference: `{reference_latency_dir}`",
        f"- Quality cohort: fixed {gate_videos}-video SA-V val subset; each candidate is compared with MX5 on the same videos.",
        f"- Promotion gates: J&F retention ≥ {min_quality_retention:.0%}; learned-path mask IoU ≥ {min_learned_mask_iou:.2f}; N=8 faster than the runtime reference.",
        "- N=1 retention is reported but is not a screen blocker; deployment can route N<4 through the legacy path.",
        "",
        "| Rank | Pareto | Variant | Status | Gate | Mini J&F | Retention | Mask IoU | N1 FPS | N1 ret. | N2 FPS | N4 FPS | N8 FPS | N8 gain | N8 ms | Promote |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['pareto']} | {row['variant']} | "
            f"{row['status']} | {row['gate_status']} | "
            f"{show(row['mini_val_jf'])} | "
            f"{show(row['jf_retention'], 3)} | "
            f"{show(row['verification_min_mask_iou'], 3)} | "
            f"{show(row['n1_fps'])} | "
            f"{show(row['n1_retention'], 3)} | "
            f"{show(row['n2_fps'])} | {show(row['n4_fps'])} | "
            f"{show(row['n8_fps'])} | "
            f"{show(row['n8_gain_pct'], 1)}% | "
            f"{show(row['n8_ms_per_frame'])} | "
            f"{row['promote']} |"
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
        variants,
        args.gate_videos,
        args.min_quality_retention,
        args.min_learned_mask_iou,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "screen_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = markdown(
        rows,
        args.reference_latency_dir,
        args.gate_videos,
        args.min_quality_retention,
        args.min_learned_mask_iou,
    )
    (args.out_dir / "screen_results.md").write_text(
        report, encoding="utf-8"
    )
    (args.out_dir / "screen_results.json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    promotions = sorted(
        (row for row in rows if row["promote"]),
        key=lambda row: int(row["rank"]),
    )[: args.max_promotions]
    (args.out_dir / "promotion_candidates.txt").write_text(
        "".join(f"{row['variant']}\n" for row in promotions),
        encoding="utf-8",
    )
    print(report, end="")


if __name__ == "__main__":
    main()
