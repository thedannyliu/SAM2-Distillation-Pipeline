#!/usr/bin/env python3
"""Audit a mounted SA-V release used by company training and evaluation runs."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def video_ids_from_list(path: Path) -> set[str]:
    return {
        Path(line.strip()).stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def decode_image(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except Exception as exc:  # noqa: BLE001 - report decoder failures
        return f"{path}: {exc}"


def parse_json(path: Path) -> str | None:
    try:
        with path.open(encoding="utf-8") as handle:
            json.load(handle)
        return None
    except Exception as exc:  # noqa: BLE001 - report parser failures
        return f"{path}: {exc}"


def select_for_validation(
    paths: list[Path], count: int, decode_all: bool, seed: int
) -> list[Path]:
    if decode_all or count >= len(paths):
        return paths
    if count <= 0:
        return []
    return random.Random(seed).sample(paths, count)


def validate_paths(
    paths: list[Path], worker, workers: int, count: int, decode_all: bool, seed: int
) -> tuple[int, list[str]]:
    selected = select_for_validation(paths, count, decode_all, seed)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        errors = [error for error in executor.map(worker, selected) if error]
    return len(selected), errors


def audit_train(root: Path, args: argparse.Namespace) -> tuple[dict[str, object], list[str]]:
    mp4_files = sorted(root.rglob("*.mp4"))
    manual_files = sorted(root.rglob("*_manual.json"))
    auto_files = sorted(root.rglob("*_auto.json"))
    mp4_ids = {path.stem for path in mp4_files}
    manual_ids = {
        path.name.removesuffix("_manual.json") for path in manual_files
    }
    zero_size = [str(path) for path in mp4_files + manual_files + auto_files if path.stat().st_size == 0]
    json_checked, json_errors = validate_paths(
        manual_files + auto_files,
        parse_json,
        args.workers,
        args.decode_samples,
        args.decode_all,
        args.seed,
    )
    checks = []
    if len(mp4_files) != args.expected_train_videos:
        checks.append(f"sav_train mp4 count is {len(mp4_files)}, expected {args.expected_train_videos}")
    if len(manual_files) != args.expected_manual_annotations:
        checks.append(
            f"manual annotation count is {len(manual_files)}, expected {args.expected_manual_annotations}"
        )
    if len(auto_files) != args.expected_auto_annotations:
        checks.append(
            f"auto annotation count is {len(auto_files)}, expected {args.expected_auto_annotations}"
        )
    if len(mp4_ids) != len(mp4_files):
        checks.append("duplicate train video IDs found")
    if manual_ids - mp4_ids:
        checks.append(f"{len(manual_ids - mp4_ids)} manual annotations have no MP4")
    if zero_size:
        checks.append(f"{len(zero_size)} zero-byte train files found")
    if json_errors:
        checks.append(f"{len(json_errors)} sampled train JSON files failed to parse")
    result = {
        "mp4_files": len(mp4_files),
        "unique_video_ids": len(mp4_ids),
        "manual_annotations": len(manual_files),
        "auto_annotations": len(auto_files),
        "mp4_without_manual_count": len(mp4_ids - manual_ids),
        "mp4_without_manual_examples": sorted(mp4_ids - manual_ids)[:20],
        "manual_without_mp4_count": len(manual_ids - mp4_ids),
        "zero_size_count": len(zero_size),
        "zero_size_examples": zero_size[:20],
        "json_files_checked": json_checked,
        "json_error_count": len(json_errors),
        "json_error_examples": json_errors[:20],
        "video_ids": mp4_ids,
    }
    return result, checks


def audit_sampled_frames(
    root: Path, train_video_ids: set[str], args: argparse.Namespace
) -> tuple[dict[str, object], list[str]]:
    video_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    image_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    counts = Counter(path.parent.name for path in image_files)
    frame_video_ids = set(counts)
    bad_per_video = {video: count for video, count in counts.items() if count != args.expected_frames_per_video}
    zero_size = [str(path) for path in image_files if path.stat().st_size == 0]
    images_checked, decode_errors = validate_paths(
        image_files,
        decode_image,
        args.workers,
        args.decode_samples,
        args.decode_all,
        args.seed + 1,
    )
    checks = []
    if len(video_dirs) != args.expected_train_videos:
        checks.append(f"JPEGImages video directory count is {len(video_dirs)}, expected {args.expected_train_videos}")
    if len(image_files) != args.expected_train_frames:
        checks.append(f"JPEGImages frame count is {len(image_files)}, expected {args.expected_train_frames}")
    if bad_per_video:
        checks.append(f"{len(bad_per_video)} JPEGImages videos do not have {args.expected_frames_per_video} frames")
    if frame_video_ids != train_video_ids:
        checks.append(
            f"JPEGImages/train video ID mismatch: {len(train_video_ids - frame_video_ids)} missing and "
            f"{len(frame_video_ids - train_video_ids)} extra"
        )
    if zero_size:
        checks.append(f"{len(zero_size)} zero-byte sampled train frames found")
    if decode_errors:
        checks.append(f"{len(decode_errors)} sampled train frames failed to decode")
    result = {
        "video_dirs": len(video_dirs),
        "image_files": len(image_files),
        "frames_per_video_min": min(counts.values(), default=0),
        "frames_per_video_max": max(counts.values(), default=0),
        "bad_frames_per_video_count": len(bad_per_video),
        "bad_frames_per_video_examples": dict(sorted(bad_per_video.items())[:20]),
        "train_video_ids_missing_count": len(train_video_ids - frame_video_ids),
        "train_video_ids_extra_count": len(frame_video_ids - train_video_ids),
        "zero_size_count": len(zero_size),
        "zero_size_examples": zero_size[:20],
        "images_checked": images_checked,
        "decode_error_count": len(decode_errors),
        "decode_error_examples": decode_errors[:20],
    }
    return result, checks


def audit_prepared_split(
    root: Path, split: str, expected_videos: int, args: argparse.Namespace
) -> tuple[dict[str, object], list[str]]:
    list_path = root / f"{split}.txt"
    image_root = root / "JPEGImages_24fps"
    annotation_root = root / "Annotations_6fps"
    listed = video_ids_from_list(list_path)
    image_dirs = {path.name for path in image_root.iterdir() if path.is_dir()}
    annotation_dirs = {path.name for path in annotation_root.iterdir() if path.is_dir()}
    image_files = sorted(
        path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    mask_files = sorted(annotation_root.rglob("*.png"))
    images_by_video: dict[str, set[str]] = {}
    for path in image_files:
        images_by_video.setdefault(path.parent.name, set()).add(path.stem)
    mask_frame_without_image = []
    for path in mask_files:
        video_id = path.relative_to(annotation_root).parts[0]
        if path.stem not in images_by_video.get(video_id, set()):
            mask_frame_without_image.append(str(path))
    zero_size = [str(path) for path in image_files + mask_files if path.stat().st_size == 0]
    image_checked, image_errors = validate_paths(
        image_files,
        decode_image,
        args.workers,
        args.decode_samples,
        args.decode_all,
        args.seed + 2,
    )
    mask_checked, mask_errors = validate_paths(
        mask_files,
        decode_image,
        args.workers,
        args.decode_samples,
        args.decode_all,
        args.seed + 3,
    )
    checks = []
    if len(listed) != expected_videos:
        checks.append(f"{split} list has {len(listed)} videos, expected {expected_videos}")
    if image_dirs != listed:
        checks.append(
            f"{split} image/list mismatch: {len(listed - image_dirs)} missing and {len(image_dirs - listed)} extra"
        )
    if annotation_dirs != listed:
        checks.append(
            f"{split} annotation/list mismatch: {len(listed - annotation_dirs)} missing and "
            f"{len(annotation_dirs - listed)} extra"
        )
    empty_image_videos = sorted(video for video in listed if not images_by_video.get(video))
    mask_video_ids = {path.relative_to(annotation_root).parts[0] for path in mask_files}
    empty_annotation_videos = sorted(listed - mask_video_ids)
    if empty_image_videos:
        checks.append(f"{len(empty_image_videos)} {split} videos contain no JPEG frames")
    if empty_annotation_videos:
        checks.append(f"{len(empty_annotation_videos)} {split} videos contain no PNG masks")
    if mask_frame_without_image:
        checks.append(f"{len(mask_frame_without_image)} {split} masks have no matching JPEG frame")
    if zero_size:
        checks.append(f"{len(zero_size)} zero-byte {split} image/mask files found")
    if image_errors or mask_errors:
        checks.append(f"{len(image_errors) + len(mask_errors)} sampled {split} files failed to decode")
    result = {
        "list_file": str(list_path),
        "listed_videos": len(listed),
        "image_video_dirs": len(image_dirs),
        "annotation_video_dirs": len(annotation_dirs),
        "image_files": len(image_files),
        "mask_files": len(mask_files),
        "listed_without_images_count": len(listed - image_dirs),
        "image_dirs_not_listed_count": len(image_dirs - listed),
        "listed_without_annotations_count": len(listed - annotation_dirs),
        "annotation_dirs_not_listed_count": len(annotation_dirs - listed),
        "empty_image_video_count": len(empty_image_videos),
        "empty_annotation_video_count": len(empty_annotation_videos),
        "mask_frame_without_image_count": len(mask_frame_without_image),
        "mask_frame_without_image_examples": mask_frame_without_image[:20],
        "zero_size_count": len(zero_size),
        "images_checked": image_checked,
        "masks_checked": mask_checked,
        "decode_error_count": len(image_errors) + len(mask_errors),
        "decode_error_examples": (image_errors + mask_errors)[:20],
    }
    return result, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sav-root", type=Path, default=Path("/mnt/data/danny-dataset/SA-V"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--decode-samples", type=int, default=200)
    parser.add_argument("--decode-all", action="store_true")
    parser.add_argument("--seed", type=int, default=310107256)
    parser.add_argument("--expected-train-videos", type=int, default=50453)
    parser.add_argument("--expected-manual-annotations", type=int, default=50452)
    parser.add_argument("--expected-auto-annotations", type=int, default=48306)
    parser.add_argument("--expected-train-frames", type=int, default=807248)
    parser.add_argument("--expected-frames-per-video", type=int, default=16)
    parser.add_argument("--expected-val-videos", type=int, default=155)
    parser.add_argument("--expected-test-videos", type=int, default=155)
    args = parser.parse_args()

    required = [
        args.sav_root / "sav_train",
        args.sav_root / "sav_val" / "JPEGImages_24fps",
        args.sav_root / "sav_val" / "Annotations_6fps",
        args.sav_root / "sav_val" / "sav_val.txt",
        args.sav_root / "sav_test" / "JPEGImages_24fps",
        args.sav_root / "sav_test" / "Annotations_6fps",
        args.sav_root / "sav_test" / "sav_test.txt",
        args.sav_root / "JPEGImages",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required SA-V paths:\n  " + "\n  ".join(missing))

    train, failures = audit_train(args.sav_root / "sav_train", args)
    train_ids = train.pop("video_ids")
    sampled_frames, frame_failures = audit_sampled_frames(
        args.sav_root / "JPEGImages", train_ids, args
    )
    val, val_failures = audit_prepared_split(
        args.sav_root / "sav_val", "sav_val", args.expected_val_videos, args
    )
    test, test_failures = audit_prepared_split(
        args.sav_root / "sav_test", "sav_test", args.expected_test_videos, args
    )
    failures.extend(frame_failures + val_failures + test_failures)
    report = {
        "status": "pass" if not failures else "fail",
        "sav_root": str(args.sav_root),
        "decode_mode": "all" if args.decode_all else f"sample_{args.decode_samples}",
        "failures": failures,
        "sav_train": train,
        "JPEGImages": sampled_frames,
        "sav_val": val,
        "sav_test": test,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
