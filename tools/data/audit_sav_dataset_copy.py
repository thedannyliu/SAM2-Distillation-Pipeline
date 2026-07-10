#!/usr/bin/env python3
"""Audit a staged SA-V copy and its video-balanced Stage 1 frame cache."""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image


def inventory(root: Path) -> dict[str, int]:
    result = {}
    for path in root.rglob("*"):
        if path.is_file():
            result[str(path.relative_to(root))] = path.stat().st_size
    return result


def list_video_ids(split_root: Path) -> set[str]:
    lists = sorted(split_root.glob("*.txt"))
    if not lists:
        return set()
    return {
        Path(line.strip()).stem
        for line in lists[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def decode_image(path: str) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except Exception as exc:  # noqa: BLE001 - audit records decoder failures
        return f"{path}: {exc}"


def compare_split(source: Path, target: Path) -> dict[str, object]:
    source_files = inventory(source)
    target_files = inventory(target)
    missing = sorted(path for path in source_files if path not in target_files)
    size_mismatch = sorted(
        path
        for path, size in source_files.items()
        if path in target_files and target_files[path] != size
    )
    extras = sorted(path for path in target_files if path not in source_files)
    return {
        "source_files": len(source_files),
        "source_bytes": sum(source_files.values()),
        "target_files": len(target_files),
        "target_bytes": sum(target_files.values()),
        "missing_count": len(missing),
        "missing_examples": missing[:20],
        "size_mismatch_count": len(size_mismatch),
        "size_mismatch_examples": size_mismatch[:20],
        "extra_count": len(extras),
        "extra_examples": extras[:20],
    }


def train_semantics(root: Path) -> dict[str, object]:
    mp4_ids = {path.stem for path in root.rglob("*.mp4")}
    manual_ids = {
        path.name.removesuffix("_manual.json") for path in root.rglob("*_manual.json")
    }
    auto_ids = {
        path.name.removesuffix("_auto.json") for path in root.rglob("*_auto.json")
    }
    return {
        "mp4": len(mp4_ids),
        "manual_annotations": len(manual_ids),
        "auto_annotations": len(auto_ids),
        "mp4_without_manual_count": len(mp4_ids - manual_ids),
        "mp4_without_manual_examples": sorted(mp4_ids - manual_ids)[:20],
        "manual_without_mp4_count": len(manual_ids - mp4_ids),
        "manual_without_mp4_examples": sorted(manual_ids - mp4_ids)[:20],
    }


def prepared_semantics(root: Path) -> dict[str, object]:
    listed = list_video_ids(root)
    image_root = root / "JPEGImages_24fps"
    annotation_root = root / "Annotations_6fps"
    image_ids = {path.name for path in image_root.iterdir() if path.is_dir()}
    annotation_ids = {path.name for path in annotation_root.iterdir() if path.is_dir()}
    image_files = sum(1 for path in image_root.rglob("*") if path.is_file())
    mask_files = sum(1 for path in annotation_root.rglob("*.png"))
    return {
        "listed_videos": len(listed),
        "image_video_dirs": len(image_ids),
        "annotation_video_dirs": len(annotation_ids),
        "image_files": image_files,
        "mask_files": mask_files,
        "listed_without_images_count": len(listed - image_ids),
        "listed_without_images_examples": sorted(listed - image_ids)[:20],
        "listed_without_annotations_count": len(listed - annotation_ids),
        "listed_without_annotations_examples": sorted(listed - annotation_ids)[:20],
    }


def processed_semantics(
    manifest_path: Path,
    target_root: Path,
    workers: int,
    decode_samples: int,
) -> dict[str, object]:
    frame = pd.read_parquet(manifest_path)
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val_sav"]
    paths = [str(path) for path in frame["image_path"]]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        exists = list(executor.map(lambda path: Path(path).is_file(), paths, chunksize=512))
    missing = [path for path, present in zip(paths, exists) if not present]
    external = [path for path in paths if not Path(path).is_relative_to(target_root)]
    rng = random.Random(310107256)
    present_paths = [path for path, present in zip(paths, exists) if present]
    selected = rng.sample(present_paths, min(decode_samples, len(present_paths)))
    with ThreadPoolExecutor(max_workers=min(workers, 32)) as executor:
        decode_errors = [error for error in executor.map(decode_image, selected) if error]
    return {
        "manifest": str(manifest_path),
        "rows": len(frame),
        "split_rows": {str(key): int(value) for key, value in frame["split"].value_counts().items()},
        "train_videos": int(train["video_id"].nunique()),
        "val_videos": int(val["video_id"].nunique()),
        "train_val_overlap": len(set(train["video_id"]) & set(val["video_id"])),
        "missing_image_count": len(missing),
        "missing_image_examples": missing[:20],
        "paths_outside_group_target_count": len(external),
        "paths_outside_group_target_examples": external[:20],
        "decoded_samples": len(selected),
        "decode_error_count": len(decode_errors),
        "decode_error_examples": decode_errors[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--decode-samples", type=int, default=200)
    args = parser.parse_args()

    required_splits = ("sav_train", "sav_val", "sav_test")
    missing_roots = [
        str(root / split)
        for root in (args.source_root, args.target_root)
        for split in required_splits
        if not (root / split).is_dir()
    ]
    if not args.manifest.is_file():
        missing_roots.append(str(args.manifest))
    if missing_roots:
        raise SystemExit("Missing required roots:\n  " + "\n  ".join(missing_roots))

    comparisons = {
        split: compare_split(args.source_root / split, args.target_root / split)
        for split in required_splits
    }
    semantics = {
        "sav_train": train_semantics(args.target_root / "sav_train"),
        "sav_val": prepared_semantics(args.target_root / "sav_val"),
        "sav_test": prepared_semantics(args.target_root / "sav_test"),
    }
    processed = processed_semantics(
        args.manifest,
        args.target_root.parent / "sam2_distill",
        args.workers,
        args.decode_samples,
    )

    comparison_ok = all(
        values["missing_count"] == 0 and values["size_mismatch_count"] == 0
        for values in comparisons.values()
    )
    prepared_ok = all(
        semantics[split]["listed_videos"] > 0
        and semantics[split]["listed_without_images_count"] == 0
        and semantics[split]["listed_without_annotations_count"] == 0
        and semantics[split]["image_files"] > 0
        and semantics[split]["mask_files"] > 0
        for split in ("sav_val", "sav_test")
    )
    train_ok = semantics["sav_train"]["mp4"] > 0
    processed_ok = (
        processed["split_rows"].get("train") == 807248
        and processed["split_rows"].get("val_sav") == 1240
        and processed["train_videos"] == 50453
        and processed["val_videos"] == 155
        and processed["train_val_overlap"] == 0
        and processed["missing_image_count"] == 0
        and processed["paths_outside_group_target_count"] == 0
        and processed["decode_error_count"] == 0
    )
    status = "pass" if comparison_ok and train_ok and prepared_ok and processed_ok else "fail"
    report = {
        "status": status,
        "source_root": str(args.source_root),
        "target_root": str(args.target_root),
        "comparisons": comparisons,
        "semantics": semantics,
        "processed": processed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.ready_marker.unlink(missing_ok=True)
    if status == "pass":
        args.ready_marker.parent.mkdir(parents=True, exist_ok=True)
        args.ready_marker.write_text(str(args.report) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
