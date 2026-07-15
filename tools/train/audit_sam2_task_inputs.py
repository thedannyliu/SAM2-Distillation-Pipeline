#!/usr/bin/env python3
"""Audit SA-V task data and progressive checkpoint inputs before GPU training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sam2_distill.data.sav_task_dataset import resolve_sav_train_annotation_path
from sam2_distill.models.stage1_checkpoint import extract_state_dict


def count_usable_frames(annotation_path: Path, sampled_frame_ids: set[int]) -> int:
    payload = json.loads(annotation_path.read_text())
    if isinstance(payload, dict):
        annotations = payload.get("masklet", payload.get("masks"))
        fps = payload.get("fps", 6)
        if isinstance(fps, list):
            fps = fps[0]
        ann_every = 24 // int(fps)
    else:
        annotations = payload
        ann_every = 4
    if not isinstance(annotations, list):
        return 0
    return sum(
        1
        for index, annotation in enumerate(annotations)
        if index * ann_every in sampled_frame_ids
        and annotation is not None
        and None not in annotation
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stage1-checkpoint", required=True, type=Path)
    parser.add_argument("--sav-root", required=True, type=Path)
    parser.add_argument("--sample-videos", type=int, default=100)
    parser.add_argument("--max-missing-annotation-videos", type=int, default=200)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    frame = pd.read_parquet(
        args.manifest,
        columns=["video_id", "frame_idx_24fps", "image_path", "annotation_path", "split"],
    )
    train = frame[frame["split"] == "train"]
    annotation_paths = {}
    missing_annotation_videos = []
    for video_id, rows in train.groupby("video_id", sort=True):
        values = [
            value
            for value in rows["annotation_path"].tolist()
            if isinstance(value, str) and value.strip()
        ]
        resolved = resolve_sav_train_annotation_path(
            str(video_id), values[0] if values else None, args.sav_root
        )
        if resolved is None:
            missing_annotation_videos.append(str(video_id))
        else:
            annotation_paths[str(video_id)] = resolved
    videos = pd.Series(annotation_paths.keys(), dtype="object").head(args.sample_videos)
    sample = train[train["video_id"].isin(videos)]
    missing_images = [
        path for path in sample["image_path"].astype(str) if not Path(path).is_file()
    ]
    invalid_frame_ids = int((sample["frame_idx_24fps"] % 4 != 0).sum())
    insufficient_videos = []
    invalid_annotations = []
    for video_id, rows in sample.groupby("video_id"):
        annotation_path = annotation_paths[str(video_id)]
        sampled_frame_ids = set(rows["frame_idx_24fps"].astype(int))
        try:
            usable = count_usable_frames(annotation_path, sampled_frame_ids)
        except Exception as exc:  # noqa: BLE001 - report malformed release data
            invalid_annotations.append(f"{annotation_path}: {exc}")
            continue
        if usable < 4:
            insufficient_videos.append({"video_id": str(video_id), "usable_frames": usable})
    checkpoint = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
    state = extract_state_dict(checkpoint)
    required_prefixes = ("backbone.", "projections.")
    missing_prefixes = [
        prefix
        for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in state)
    ]
    split_files = {
        split: (args.sav_root / split / f"{split}.txt").is_file()
        for split in ("sav_val", "sav_test")
    }
    failures = []
    if missing_images:
        failures.append(f"missing sampled images: {missing_images[:5]}")
    if len(missing_annotation_videos) > args.max_missing_annotation_videos:
        failures.append(
            f"{len(missing_annotation_videos)} train videos lack mounted manual JSON "
            f"(allowed {args.max_missing_annotation_videos}); "
            f"examples: {missing_annotation_videos[:10]}"
        )
    if invalid_annotations:
        failures.append(f"invalid sampled annotations: {invalid_annotations[:5]}")
    if invalid_frame_ids:
        failures.append(
            f"{invalid_frame_ids} sampled train frames are not on "
            "6fps annotation cadence"
        )
    if insufficient_videos:
        failures.append(
            "sampled videos with fewer than four fully annotated frames: "
            f"{insufficient_videos[:10]}"
        )
    if missing_prefixes:
        failures.append(f"Stage 1 checkpoint missing prefixes: {missing_prefixes}")
    if not all(split_files.values()):
        failures.append(f"missing validation/test file lists: {split_files}")
    summary = {
        "status": "fail" if failures else "pass",
        "manifest": str(args.manifest),
        "train_rows": len(train),
        "train_videos": int(train["video_id"].nunique()),
        "train_videos_with_manual_annotations": len(annotation_paths),
        "train_videos_without_manual_annotations": len(missing_annotation_videos),
        "missing_manual_annotation_examples": missing_annotation_videos[:10],
        "sampled_videos_checked": len(videos),
        "sampled_rows_checked": len(sample),
        "sampled_videos_with_fewer_than_four_usable_frames": len(insufficient_videos),
        "stage1_tensors": len(state),
        "split_files": split_files,
        "failures": failures,
    }
    if args.compact and not failures:
        print(
            " | ".join(
                [
                    "SA-V task audit: PASS",
                    f"candidates {summary['train_videos']:,}",
                    f"usable {summary['train_videos_with_manual_annotations']:,}",
                    f"excluded {summary['train_videos_without_manual_annotations']:,}",
                    f"sampled {summary['sampled_videos_checked']:,}",
                    "val/test lists PASS",
                ]
            )
        )
    else:
        print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
