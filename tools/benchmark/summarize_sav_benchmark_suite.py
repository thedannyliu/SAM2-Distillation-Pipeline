#!/usr/bin/env python3
"""Summarize raw SA-V image/VOS benchmark suite outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
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
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
