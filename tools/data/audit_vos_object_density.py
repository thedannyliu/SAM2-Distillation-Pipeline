#!/usr/bin/env python3
"""Audit per-video object density and select a shared-frame VOS cohort."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_THRESHOLDS = (1, 2, 4, 8, 16, 32)


def video_names(image_root: Path, video_list_file: Path | None) -> list[str]:
    if video_list_file is None:
        return sorted(path.name for path in image_root.iterdir() if path.is_dir())
    return [
        Path(line.strip()).stem
        for line in video_list_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mask_is_nonempty(path: Path) -> bool:
    with Image.open(path) as image:
        return bool(np.asarray(image).any())


def shared_frame_prompts(
    ann_video_dir: Path,
    min_objects: int,
) -> tuple[int, list[tuple[str, Path]]] | None:
    """Find a frame with at least ``min_objects`` non-empty object masks."""
    by_frame: dict[int, list[tuple[str, Path]]] = defaultdict(list)
    for object_dir in sorted(path for path in ann_video_dir.iterdir() if path.is_dir()):
        for mask_path in object_dir.glob("*.png"):
            try:
                frame_idx = int(mask_path.stem)
            except ValueError:
                continue
            by_frame[frame_idx].append((object_dir.name, mask_path))

    candidates = sorted(by_frame.items(), key=lambda item: (-len(item[1]), item[0]))
    for frame_idx, records in candidates:
        if len(records) < min_objects:
            break
        nonempty = [
            (object_id, mask_path)
            for object_id, mask_path in sorted(records)
            if mask_is_nonempty(mask_path)
        ]
        if len(nonempty) >= min_objects:
            return frame_idx, nonempty
    return None


def count_tracks(ann_video_dir: Path) -> int:
    return sum(
        1
        for object_dir in ann_video_dir.iterdir()
        if object_dir.is_dir() and next(object_dir.glob("*.png"), None) is not None
    )


def count_images(image_video_dir: Path) -> int:
    return sum(
        1
        for path in image_video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def percentile(values: list[int], percentile_value: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), percentile_value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--ann-root", required=True, type=Path)
    parser.add_argument("--video-list-file", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-shared-objects", type=int, default=8)
    parser.add_argument("--max-cohort-videos", type=int, default=16)
    parser.add_argument("--seed", type=int, default=310107256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.image_root, args.ann_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if args.video_list_file is not None and not args.video_list_file.is_file():
        raise FileNotFoundError(args.video_list_file)
    if args.min_shared_objects < 1:
        raise SystemExit("--min-shared-objects must be positive")

    rows = []
    missing_images = []
    missing_annotations = []
    eligible = []
    for video in video_names(args.image_root, args.video_list_file):
        image_video_dir = args.image_root / video
        ann_video_dir = args.ann_root / video
        if not image_video_dir.is_dir():
            missing_images.append(video)
            continue
        if not ann_video_dir.is_dir():
            missing_annotations.append(video)
            continue
        shared = shared_frame_prompts(ann_video_dir, args.min_shared_objects)
        row = {
            "video": video,
            "image_frames": count_images(image_video_dir),
            "track_count": count_tracks(ann_video_dir),
            "shared_frame": shared[0] if shared is not None else "",
            "shared_nonempty_objects": len(shared[1]) if shared is not None else 0,
            "eligible": int(shared is not None),
        }
        rows.append(row)
        if shared is not None:
            eligible.append(video)

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    cohort = sorted(
        eligible[: args.max_cohort_videos]
        if args.max_cohort_videos > 0
        else eligible
    )
    track_counts = [int(row["track_count"]) for row in rows]
    shared_counts = [int(row["shared_nonempty_objects"]) for row in rows]
    summary = {
        "status": "pass" if cohort else "fail",
        "image_root": str(args.image_root),
        "ann_root": str(args.ann_root),
        "video_list_file": str(args.video_list_file)
        if args.video_list_file is not None
        else None,
        "videos_scanned": len(rows),
        "missing_image_videos": missing_images,
        "missing_annotation_videos": missing_annotations,
        "min_shared_objects": args.min_shared_objects,
        "eligible_videos": len(eligible),
        "cohort_videos": cohort,
        "seed": args.seed,
        "track_count": {
            "median": percentile(track_counts, 50),
            "p90": percentile(track_counts, 90),
            "max": max(track_counts, default=0),
            "at_least": {
                str(threshold): sum(value >= threshold for value in track_counts)
                for threshold in DEFAULT_THRESHOLDS
            },
        },
        "validated_shared_frame_count": {
            "median": percentile(shared_counts, 50),
            "p90": percentile(shared_counts, 90),
            "max": max(shared_counts, default=0),
            "at_least": {
                str(threshold): sum(value >= threshold for value in shared_counts)
                for threshold in DEFAULT_THRESHOLDS
            },
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video",
        "image_frames",
        "track_count",
        "shared_frame",
        "shared_nonempty_objects",
        "eligible",
    ]
    with (args.out_dir / "per_video.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "cohort.txt").write_text(
        "".join(f"{video}\n" for video in cohort),
        encoding="utf-8",
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(
            f"No video has {args.min_shared_objects} non-empty masks on one frame"
        )


if __name__ == "__main__":
    main()
