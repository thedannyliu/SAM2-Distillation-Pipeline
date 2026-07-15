#!/usr/bin/env python3
"""Rebase the corrected SA-V Stage 1 manifest onto a mounted dataset release."""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sam2_distill.data.sav_task_dataset import resolve_sav_train_annotation_path


def replace_split_path(value: object, sav_root: Path, split: str) -> object:
    if not isinstance(value, str) or f"/{split}/" not in value:
        return value
    suffix = value.split(f"/{split}/", 1)[1]
    return str(sav_root / split / suffix)


def verify_image(path: str) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except Exception as exc:  # noqa: BLE001 - report invalid mounted images
        return f"{path}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--sav-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--decode-samples", type=int, default=200)
    parser.add_argument("--expected-train-rows", type=int, default=807248)
    parser.add_argument("--expected-train-videos", type=int, default=50453)
    parser.add_argument("--expected-train-frames-per-video", type=int, default=16)
    parser.add_argument("--expected-val-rows", type=int, default=1240)
    parser.add_argument("--expected-val-videos", type=int, default=155)
    parser.add_argument("--expected-val-frames-per-video", type=int, default=8)
    parser.add_argument("--max-missing-train-annotations", type=int, default=200)
    args = parser.parse_args()

    frame = pd.read_parquet(args.source_manifest)
    required = {"sample_id", "video_id", "image_path", "split"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise SystemExit(f"Missing manifest columns: {sorted(missing_columns)}")
    split_counts = frame["split"].value_counts().to_dict()
    if (
        split_counts.get("train") != args.expected_train_rows
        or split_counts.get("val_sav") != args.expected_val_rows
    ):
        raise SystemExit(f"Source is not the corrected manifest: {split_counts}")
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val_sav"]
    if (
        train["video_id"].nunique() != args.expected_train_videos
        or val["video_id"].nunique() != args.expected_val_videos
    ):
        raise SystemExit("Unexpected train/validation video counts")
    if set(train["video_id"]) & set(val["video_id"]):
        raise SystemExit("Train/validation video overlap detected")
    train_counts = train.groupby("video_id").size()
    val_counts = val.groupby("video_id").size()
    if not (train_counts == args.expected_train_frames_per_video).all() or not (
        val_counts == args.expected_val_frames_per_video
    ).all():
        raise SystemExit("Manifest does not have the expected per-video frame counts")

    rebased = frame.copy()
    train_mask = rebased["split"] == "train"
    val_mask = rebased["split"] == "val_sav"
    rebased.loc[train_mask, "image_path"] = [
        str(args.sav_root / "JPEGImages" / video / Path(path).name)
        for video, path in zip(
            rebased.loc[train_mask, "video_id"],
            rebased.loc[train_mask, "image_path"],
        )
    ]
    rebased.loc[val_mask, "image_path"] = [
        str(args.sav_root / "sav_val" / "JPEGImages_24fps" / video / Path(path).name)
        for video, path in zip(
            rebased.loc[val_mask, "video_id"],
            rebased.loc[val_mask, "image_path"],
        )
    ]
    for column in ("video_path", "annotation_path"):
        if column not in rebased.columns:
            continue
        rebased.loc[train_mask, column] = rebased.loc[train_mask, column].map(
            lambda value: replace_split_path(value, args.sav_root, "sav_train")
        )
        rebased.loc[val_mask, column] = rebased.loc[val_mask, column].map(
            lambda value: replace_split_path(value, args.sav_root, "sav_val")
        )

    if "annotation_path" not in rebased.columns:
        rebased["annotation_path"] = None
    train_annotation_paths = {}
    missing_train_annotations = []
    for video_id, rows in rebased.loc[train_mask].groupby("video_id", sort=True):
        values = [
            value
            for value in rows["annotation_path"].tolist()
            if isinstance(value, str) and value.strip()
        ]
        resolved = resolve_sav_train_annotation_path(
            str(video_id), values[0] if values else None, args.sav_root
        )
        if resolved is None:
            missing_train_annotations.append(str(video_id))
            train_annotation_paths[str(video_id)] = None
        else:
            train_annotation_paths[str(video_id)] = str(resolved)
    if len(missing_train_annotations) > args.max_missing_train_annotations:
        raise SystemExit(
            f"Mounted release is missing manual JSON for "
            f"{len(missing_train_annotations)} train videos; "
            f"examples: {missing_train_annotations[:10]}"
        )
    rebased.loc[train_mask, "annotation_path"] = rebased.loc[
        train_mask, "video_id"
    ].map(train_annotation_paths)

    if rebased["sample_id"].duplicated().any():
        raise SystemExit("Duplicate sample_id values found")
    paths = rebased["image_path"].astype(str).tolist()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        present = list(executor.map(lambda value: Path(value).is_file(), paths, chunksize=512))
    missing_paths = [path for path, exists in zip(paths, present) if not exists]
    if missing_paths:
        raise SystemExit(
            f"Mounted release is missing {len(missing_paths)} selected images; examples: {missing_paths[:10]}"
        )
    selected = random.Random(310107256).sample(paths, min(args.decode_samples, len(paths)))
    with ThreadPoolExecutor(max_workers=min(args.workers, 32)) as executor:
        decode_errors = [error for error in executor.map(verify_image, selected) if error]
    if decode_errors:
        raise SystemExit(f"Mounted image decode failures: {decode_errors[:10]}")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_manifest.with_suffix(".tmp.parquet")
    rebased.to_parquet(temporary, index=False)
    temporary.replace(args.output_manifest)
    summary = {
        "status": "pass",
        "source_manifest": str(args.source_manifest),
        "output_manifest": str(args.output_manifest),
        "sav_root": str(args.sav_root),
        "rows": len(rebased),
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "train_videos": int(train["video_id"].nunique()),
        "train_videos_with_manual_annotations": len(train_annotation_paths)
        - len(missing_train_annotations),
        "train_videos_without_manual_annotations": len(missing_train_annotations),
        "missing_manual_annotation_examples": missing_train_annotations[:10],
        "val_videos": int(val["video_id"].nunique()),
        "decoded_samples": len(selected),
    }
    args.output_manifest.with_suffix(".provenance.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
