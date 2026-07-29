#!/usr/bin/env python3
"""Select and deterministically repeat dense SA-V training videos."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sam2_distill.data.sav_task_dataset import resolve_sav_train_annotation_path


def frame_object_counts(payload: Any) -> list[int]:
    masklets = payload.get("masklet") if isinstance(payload, dict) else payload
    if not isinstance(masklets, list):
        raise ValueError("annotation has no masklet list")
    return [
        sum(item is not None for item in frame)
        for frame in masklets
        if isinstance(frame, list)
    ]


def repeat_to_length(values: list[str], target: int, seed: int) -> list[str]:
    if not values:
        raise ValueError("cannot repeat an empty video cohort")
    if target <= 0:
        return list(values)
    rng = random.Random(seed)
    output = []
    while len(output) < target:
        cycle = list(values)
        rng.shuffle(cycle)
        output.extend(cycle)
    return output[:target]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sav-root", required=True, type=Path)
    parser.add_argument("--output-video-ids", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-summary", required=True, type=Path)
    parser.add_argument("--min-objects", type=int, default=8)
    parser.add_argument("--min-dense-frames", type=int, default=4)
    parser.add_argument("--target-samples", type=int, default=50337)
    parser.add_argument("--seed", type=int, default=310107256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_objects < 1 or args.min_dense_frames < 1:
        raise SystemExit("--min-objects and --min-dense-frames must be positive")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    frame = pd.read_parquet(
        args.manifest,
        columns=["video_id", "annotation_path", "split"],
    )
    frame = frame[frame["split"] == "train"]
    rows = []
    eligible = []
    missing_annotations = []
    invalid_annotations = []
    for video_id, video_rows in frame.groupby("video_id", sort=True):
        annotation_values = [
            value
            for value in video_rows["annotation_path"].tolist()
            if isinstance(value, str) and value.strip()
        ]
        annotation = resolve_sav_train_annotation_path(
            str(video_id),
            annotation_values[0] if annotation_values else None,
            args.sav_root,
        )
        if annotation is None:
            missing_annotations.append(str(video_id))
            continue
        try:
            counts = frame_object_counts(
                json.loads(annotation.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, json.JSONDecodeError):
            invalid_annotations.append(str(video_id))
            continue
        dense_frames = sum(count >= args.min_objects for count in counts)
        row = {
            "video_id": str(video_id),
            "annotation_path": str(annotation),
            "annotated_frames": len(counts),
            "max_visible_objects": max(counts, default=0),
            "dense_frames": dense_frames,
            "eligible": int(dense_frames >= args.min_dense_frames),
        }
        rows.append(row)
        if row["eligible"]:
            eligible.append(str(video_id))

    sampled = repeat_to_length(eligible, args.target_samples, args.seed)
    summary = {
        "status": "pass",
        "manifest": str(args.manifest),
        "sav_root": str(args.sav_root),
        "min_objects": args.min_objects,
        "min_dense_frames": args.min_dense_frames,
        "videos_scanned": len(rows),
        "eligible_unique_videos": len(eligible),
        "output_samples": len(sampled),
        "target_samples": args.target_samples,
        "seed": args.seed,
        "missing_annotations": len(missing_annotations),
        "invalid_annotations": len(invalid_annotations),
        "output_video_ids": str(args.output_video_ids),
    }

    for path in (args.output_video_ids, args.out_csv, args.out_summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_ids = args.output_video_ids.with_suffix(
        args.output_video_ids.suffix + ".tmp"
    )
    temporary_ids.write_text(
        "".join(f"{video_id}\n" for video_id in sampled),
        encoding="utf-8",
    )
    temporary_ids.replace(args.output_video_ids)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_id",
                "annotation_path",
                "annotated_frames",
                "max_visible_objects",
                "dense_frames",
                "eligible",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    args.out_summary.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
