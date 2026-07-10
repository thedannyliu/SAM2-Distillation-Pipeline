#!/usr/bin/env python3
"""Summarize raw SA-V image/VOS benchmark suite outputs."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--aggregate-csv", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def image_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    image_root = root / "image"
    if not image_root.exists():
        return rows
    for summary_path in sorted(image_root.glob("*/*/summary.json")):
        model = summary_path.parents[1].name
        prompt = summary_path.parent.name
        summary = read_json(summary_path)
        latency = summary.get("latency", {})
        rows.append(
            {
                "mode": "image",
                "prompt": prompt,
                "model": model,
                "status": summary.get("status", "unknown"),
                "num_images": summary.get("num_images"),
                "num_objects": summary.get("num_objects"),
                "mIoU": summary.get("mIoU"),
                "AP": summary.get("AP"),
                "AP50": summary.get("AP50"),
                "AP75": summary.get("AP75"),
                "J&F": "",
                "J": "",
                "F": "",
                "mean_set_image_seconds": latency.get("mean_set_image_seconds"),
                "mean_prompt_seconds": latency.get("mean_prompt_seconds"),
                "mean_total_object_seconds": latency.get("mean_total_object_seconds"),
                "elapsed_sec": "",
                "sec_per_video": "",
                "summary_path": str(summary_path),
            }
        )
    return rows


def vos_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    vos_root = root / "vos"
    if not vos_root.exists():
        return rows
    for eval_path in sorted(vos_root.glob("*/*/eval_summary.json")):
        model = eval_path.parents[1].name
        prompt = eval_path.parent.name
        run_summary_path = eval_path.parent / "run_summary.json"
        if not run_summary_path.exists():
            run_summary_path = eval_path.parent / "pred" / "summary.json"
        if not run_summary_path.exists():
            continue
        eval_summary = read_json(eval_path)
        run_summary = read_json(run_summary_path)
        metrics = eval_summary.get("metrics", {})
        rows.append(
            {
                "mode": "video_tracking",
                "prompt": prompt,
                "model": model,
                "status": eval_summary.get("status", "unknown"),
                "num_images": "",
                "num_objects": "",
                "mIoU": "",
                "AP": "",
                "AP50": "",
                "AP75": "",
                "J&F": metrics.get("J&F"),
                "J": metrics.get("J"),
                "F": metrics.get("F"),
                "mean_set_image_seconds": "",
                "mean_prompt_seconds": "",
                "mean_total_object_seconds": "",
                "elapsed_sec": run_summary.get("elapsed_sec"),
                "sec_per_video": run_summary.get("sec_per_video"),
                "summary_path": str(eval_path),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = image_rows(args.root) + vos_rows(args.root)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "mode",
        "prompt",
        "model",
        "status",
        "num_images",
        "num_objects",
        "mIoU",
        "AP",
        "AP50",
        "AP75",
        "J&F",
        "J",
        "F",
        "mean_set_image_seconds",
        "mean_prompt_seconds",
        "mean_total_object_seconds",
        "elapsed_sec",
        "sec_per_video",
        "summary_path",
    ]
    write_csv(args.out_csv, rows, fieldnames)
    if args.aggregate_csv:
        upsert_aggregate_csv(args.aggregate_csv, rows, fieldnames)
    print(json.dumps({"rows": rows}, indent=2))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def upsert_aggregate_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = []
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as f:
                existing = list(csv.DictReader(f))
        keys = {(str(row["model"]), str(row["mode"]), str(row["prompt"])) for row in rows}
        merged = [
            row
            for row in existing
            if (row.get("model", ""), row.get("mode", ""), row.get("prompt", "")) not in keys
        ]
        merged.extend(rows)
        merged.sort(key=lambda row: (str(row.get("model", "")), str(row.get("mode", ""))))
        write_csv(path, merged, fieldnames)
        fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
