#!/usr/bin/env python3
"""Merge video-sharded SA-V image or VOS benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("image", "vos"), required=True)
    parser.add_argument("--shard-root", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--video-list-file", required=True, type=Path)
    parser.add_argument("--max-videos", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def expected_videos(path: Path, max_videos: int) -> list[str]:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return names[:max_videos] if max_videos > 0 else names


def average_precision(ious: list[float], scores: list[float], threshold: float) -> float:
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray([ious[index] >= threshold for index in order], dtype=np.float32)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(len(ious), 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    result = 0.0
    for recall_threshold in np.linspace(0.0, 1.0, 101):
        valid = precisions[recalls >= recall_threshold]
        result += float(valid.max()) if valid.size else 0.0
    return result / 101.0


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def require_coverage(actual: list[str], expected: list[str], label: str) -> None:
    duplicates = sorted({name for name in actual if actual.count(name) > 1})
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if duplicates or missing or extra:
        raise RuntimeError(
            f"{label} coverage mismatch: duplicates={duplicates[:10]}, "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )


def merge_image(shard_roots: list[Path], out_dir: Path, expected: list[str]) -> None:
    summaries = [read_json(root / "summary.json") for root in shard_roots]
    if any(summary.get("status") != "pass" for summary in summaries):
        raise RuntimeError("one or more image shards failed")
    checkpoints = {str(summary.get("checkpoint")) for summary in summaries}
    if len(checkpoints) != 1:
        raise RuntimeError(f"image shards used different checkpoints: {checkpoints}")

    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for root in shard_roots:
        with (root / "per_object_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            fieldnames = fieldnames or list(reader.fieldnames or [])
            rows.extend(reader)
    if not rows or not fieldnames:
        raise RuntimeError("image shards produced no per-object rows")
    keys = [
        (row["video"], row["object_id"], row["frame_stem"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate image per-object rows across shards")
    require_coverage(sorted({row["video"] for row in rows}), expected, "image")

    ious = [float(row["iou"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    prompts = [float(row["prompt_seconds"]) for row in rows]
    totals = [float(row["total_object_seconds"]) for row in rows]
    set_image_by_path: dict[str, float] = {}
    for row in rows:
        set_image_by_path.setdefault(row["image_path"], float(row["set_image_seconds"]))
    set_images = list(set_image_by_path.values())
    thresholds = [round(value, 2) for value in np.arange(0.50, 0.96, 0.05)]
    ap_by_threshold = {
        f"AP{int(threshold * 100)}": average_precision(ious, scores, threshold)
        for threshold in thresholds
    }
    summary = dict(summaries[0])
    summary.update(
        {
            "out_dir": str(out_dir),
            "num_images": len(set_image_by_path),
            "num_objects": len(rows),
            "mIoU": float(np.mean(ious)),
            "median_IoU": percentile(ious, 50),
            "AP": float(np.mean(list(ap_by_threshold.values()))),
            **ap_by_threshold,
            "latency": {
                "mean_set_image_seconds": float(np.mean(set_images)),
                "p50_set_image_seconds": percentile(set_images, 50),
                "p95_set_image_seconds": percentile(set_images, 95),
                "mean_prompt_seconds": float(np.mean(prompts)),
                "p50_prompt_seconds": percentile(prompts, 50),
                "p95_prompt_seconds": percentile(prompts, 95),
                "mean_total_object_seconds": float(np.mean(totals)),
            },
            "artifacts_saved": sum(int(item.get("artifacts_saved", 0)) for item in summaries),
            "frame_artifacts_saved": sum(
                int(item.get("frame_artifacts_saved", 0)) for item in summaries
            ),
            "parallel_shards": len(shard_roots),
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_object_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def merge_vos(shard_roots: list[Path], out_dir: Path, expected: list[str]) -> None:
    summaries = [read_json(root / "summary.json") for root in shard_roots]
    if any(summary.get("status") != "pass" for summary in summaries):
        raise RuntimeError("one or more VOS shards failed")
    checkpoints = {str(summary.get("checkpoint")) for summary in summaries}
    if len(checkpoints) != 1:
        raise RuntimeError(f"VOS shards used different checkpoints: {checkpoints}")
    videos = [name for summary in summaries for name in summary.get("video_names", [])]
    require_coverage(videos, expected, "VOS")

    rows = [row for summary in summaries for row in summary.get("video_summaries", [])]
    rows_by_video = {row["video"]: row for row in rows}
    if len(rows_by_video) != len(rows):
        raise RuntimeError("duplicate VOS video summaries across shards")
    out_dir.mkdir(parents=True, exist_ok=True)
    for root, summary in zip(shard_roots, summaries):
        for video in summary["video_names"]:
            source = root / video
            destination = out_dir / video
            if not source.is_dir():
                raise FileNotFoundError(source)
            if destination.exists():
                raise FileExistsError(destination)
            shutil.move(str(source), str(destination))

    elapsed_values = [float(summary.get("elapsed_sec", 0.0)) for summary in summaries]
    elapsed_gpu_seconds = sum(elapsed_values)
    summary = dict(summaries[0])
    summary.update(
        {
            "prediction_root": str(out_dir),
            "video_names": expected,
            "videos": len(expected),
            "elapsed_sec": elapsed_gpu_seconds,
            "sec_per_video": elapsed_gpu_seconds / max(len(expected), 1),
            "parallel_wall_seconds": max(elapsed_values, default=0.0),
            "parallel_throughput_videos_per_second": len(expected)
            / max(max(elapsed_values, default=0.0), 1e-12),
            "parallel_shards": len(shard_roots),
            "num_prediction_pngs": sum(
                int(item.get("num_prediction_pngs", 0)) for item in summaries
            ),
            "num_zero_fallback_objects": sum(
                int(item.get("num_zero_fallback_objects", 0)) for item in summaries
            ),
            "video_summaries": [rows_by_video[name] for name in expected],
        }
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    expected = expected_videos(args.video_list_file, args.max_videos)
    if not expected:
        raise SystemExit("video list is empty")
    if args.mode == "image":
        merge_image(args.shard_root, args.out_dir, expected)
    else:
        merge_vos(args.shard_root, args.out_dir, expected)
    print(
        json.dumps(
            {
                "status": "pass",
                "mode": args.mode,
                "shards": len(args.shard_root),
                "videos": len(expected),
                "out_dir": str(args.out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
