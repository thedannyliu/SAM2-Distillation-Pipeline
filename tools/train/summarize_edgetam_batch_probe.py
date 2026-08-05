#!/usr/bin/env python3
"""Summarize bounded EdgeTAM batch-capacity probe runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--max-reserved-gib", type=float, default=72.0)
    return parser.parse_args()


def read_candidate(path: Path, max_reserved_gib: float) -> dict:
    exit_code_path = path / "exit_code.txt"
    exit_code = (
        int(exit_code_path.read_text(encoding="utf-8").strip())
        if exit_code_path.is_file()
        else None
    )
    rank_files = sorted(path.glob("capacity_rank*.json"))
    ranks = [json.loads(item.read_text(encoding="utf-8")) for item in rank_files]
    timed_ranks = [item for item in ranks if item["step_seconds_mean"] is not None]
    expected_ranks = int(ranks[0]["world_size"]) if ranks else 0
    if ranks and timed_ranks:
        stage = ranks[0]["stage"].split("_batch", 1)[0]
        batch = int(ranks[0]["per_gpu_batch"])
        global_batch = int(ranks[0]["global_batch"])
        frames = int(ranks[0]["num_frames"])
        measured_steps = min(int(item["measured_steps"]) for item in ranks)
        step_seconds = max(float(item["step_seconds_mean"]) for item in timed_ranks)
        peak_reserved = max(float(item["peak_reserved_gib"]) for item in ranks)
        peak_allocated = max(float(item["peak_allocated_gib"]) for item in ranks)
        samples_per_second = global_batch / step_seconds
    else:
        parts = path.name.rsplit("_batch", 1)
        stage = parts[0]
        batch = (
            int(ranks[0]["per_gpu_batch"])
            if ranks
            else int(parts[1])
            if len(parts) == 2 and parts[1].isdigit()
            else 0
        )
        global_batch = int(ranks[0]["global_batch"]) if ranks else 0
        frames = int(ranks[0]["num_frames"]) if ranks else 0
        measured_steps = (
            min(int(item["measured_steps"]) for item in ranks) if ranks else 0
        )
        step_seconds = None
        peak_reserved = (
            max(float(item["peak_reserved_gib"]) for item in ranks)
            if ranks
            else None
        )
        peak_allocated = (
            max(float(item["peak_allocated_gib"]) for item in ranks)
            if ranks
            else None
        )
        samples_per_second = None

    passed = (
        exit_code == 0
        and len(ranks) == expected_ranks
        and len(timed_ranks) == expected_ranks
        and measured_steps >= 3
        and peak_reserved is not None
        and peak_reserved <= max_reserved_gib
    )
    return {
        "stage": stage,
        "per_gpu_batch": batch,
        "global_batch": global_batch,
        "frames": frames,
        "exit_code": exit_code,
        "ranks": len(ranks),
        "expected_ranks": expected_ranks,
        "measured_steps": measured_steps,
        "step_seconds": step_seconds,
        "samples_per_second": samples_per_second,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "passed": passed,
    }


def format_value(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    args = parse_args()
    candidates = [
        read_candidate(path, args.max_reserved_gib)
        for path in sorted(args.root.iterdir())
        if path.is_dir() and "_batch" in path.name
    ]
    if not candidates:
        raise SystemExit(f"No batch-probe candidates found under {args.root}")

    fieldnames = list(candidates[0])
    csv_path = args.root / "batch_probe_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    print(
        "stage batch/global T status steps step_s samples_s peak_alloc peak_reserved pass"
    )
    for item in candidates:
        status = "ok" if item["exit_code"] == 0 else f"exit={item['exit_code']}"
        print(
            f"{item['stage']:6s} "
            f"{item['per_gpu_batch']:>2}/{item['global_batch']:<2} "
            f"{item['frames']:>2} "
            f"{status:>7s} "
            f"{item['measured_steps']:>5} "
            f"{format_value(item['step_seconds']):>7s} "
            f"{format_value(item['samples_per_second']):>9s} "
            f"{format_value(item['peak_allocated_gib']):>10s} "
            f"{format_value(item['peak_reserved_gib']):>13s} "
            f"{str(item['passed']).lower()}"
        )

    print("recommended_per_gpu_batch")
    for stage in ("image", "t4", "t8", "t16"):
        passed = [
            item for item in candidates if item["stage"] == stage and item["passed"]
        ]
        if not passed:
            print(f"{stage}=none")
            continue
        fastest = max(passed, key=lambda item: item["samples_per_second"])
        print(
            f"{stage}={fastest['per_gpu_batch']} "
            f"global={fastest['global_batch']} "
            f"samples_s={fastest['samples_per_second']:.3f} "
            f"peak_reserved_gib={fastest['peak_reserved_gib']:.3f}"
        )
    print(f"summary_csv={csv_path}")


if __name__ == "__main__":
    main()
