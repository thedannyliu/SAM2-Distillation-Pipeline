#!/usr/bin/env python3
"""Prepare a raw SA-V train shard for image and VOS benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--max-videos", type=int, default=2)
    parser.add_argument("--max-objects-per-video", type=int, default=2)
    parser.add_argument("--ann-every", type=int, default=4)
    parser.add_argument("--frame-sample-rate", type=int, default=1)
    parser.add_argument("--use-auto", action="store_true", help="Use *_auto.json when *_manual.json is unavailable.")
    return parser.parse_args()


def first_existing_dir(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_dir():
            return path
    return None


def first_parent_dir_with_file(roots: list[Path], pattern: str) -> Path | None:
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob(pattern)):
            if path.is_file():
                return path.parent
    return None


def detect_video_root(shard: Path) -> Path:
    found = first_existing_dir([shard / "videos", shard / "train" / "videos", shard])
    if found is not None and any(found.rglob("*.mp4")):
        return found
    found = first_parent_dir_with_file([shard, shard.parent], "*.mp4")
    if found is None:
        raise FileNotFoundError(f"No mp4 files under {shard}")
    return found


def detect_ann_root(shard: Path) -> Path:
    found = first_existing_dir([shard / "annotations", shard / "train" / "annotations", shard])
    if found is not None and (any(found.glob("*_manual.json")) or any(found.glob("*_auto.json"))):
        return found
    found = first_parent_dir_with_file([shard, shard.parent], "*_manual.json")
    if found is None:
        found = first_parent_dir_with_file([shard, shard.parent], "*_auto.json")
    if found is None:
        raise FileNotFoundError(f"No *_manual.json or *_auto.json under {shard}")
    return found


def extract_frames(video_root: Path, image_root: Path, sample_rate: int, selected_videos: list[str]) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Frame extraction requires cv2/opencv-python.") from exc

    wanted = set(selected_videos)
    videos = [path for path in sorted(video_root.rglob("*.mp4")) if path.stem in wanted]
    if not videos:
        raise FileNotFoundError(f"No selected mp4 files under {video_root}")
    image_root.mkdir(parents=True, exist_ok=True)
    for video in videos:
        out_dir = image_root / video.stem
        if out_dir.exists() and any(out_dir.glob("*.jpg")):
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video}")
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_rate == 0:
                out = out_dir / f"{frame_idx:05d}.jpg"
                if not cv2.imwrite(str(out), frame):
                    raise RuntimeError(f"Could not write {out}")
            frame_idx += 1
        cap.release()


def decode_rle(rle: Any) -> np.ndarray | None:
    if not rle:
        return None
    if isinstance(rle, list) and not rle:
        return None
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise SystemExit("SA-V RLE decoding requires pycocotools.") from exc

    if isinstance(rle, dict):
        counts = rle.get("counts")
        size = rle.get("size")
        if counts is None or size is None:
            return None
        if isinstance(counts, list):
            rle = mask_utils.frPyObjects(rle, int(size[0]), int(size[1]))
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask.astype(bool)
    return None


def choose_annotation(ann_root: Path, video_id: str, use_auto: bool) -> Path | None:
    manual = ann_root / f"{video_id}_manual.json"
    auto = ann_root / f"{video_id}_auto.json"
    if manual.exists():
        return manual
    if use_auto and auto.exists():
        return auto
    return None


def discover_videos(video_root: Path, ann_root: Path, max_videos: int, use_auto: bool) -> list[str]:
    ids = []
    for video in sorted(video_root.rglob("*.mp4")):
        if choose_annotation(ann_root, video.stem, use_auto) is not None:
            ids.append(video.stem)
        if max_videos > 0 and len(ids) >= max_videos:
            break
    return ids


def write_masks(
    ann_path: Path,
    out_ann_root: Path,
    ann_every: int,
    max_objects: int,
) -> dict[str, Any]:
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    fallback_video_id = ann_path.name.removesuffix("_manual.json").removesuffix("_auto.json")
    video_id = str(payload.get("video_id") or fallback_video_id)
    masklets = payload.get("masklet")
    masklet_ids = payload.get("masklet_id") or []
    if not isinstance(masklets, list):
        raise ValueError(f"{ann_path} missing list field 'masklet'")

    if masklet_ids:
        selected_obj_indices = list(range(min(len(masklet_ids), max_objects if max_objects > 0 else len(masklet_ids))))
        object_names = {idx: f"{int(masklet_ids[idx]):03d}" for idx in selected_obj_indices}
    else:
        first_nonempty = next((frame for frame in masklets if isinstance(frame, list)), [])
        selected_obj_indices = list(range(min(len(first_nonempty), max_objects if max_objects > 0 else len(first_nonempty))))
        object_names = {idx: f"{idx:03d}" for idx in selected_obj_indices}

    written = 0
    for frame_idx_6fps, frame_rles in enumerate(masklets):
        if not isinstance(frame_rles, list):
            continue
        frame_idx_24fps = frame_idx_6fps * ann_every
        for obj_idx in selected_obj_indices:
            if obj_idx >= len(frame_rles):
                continue
            mask = decode_rle(frame_rles[obj_idx])
            if mask is None or not mask.any():
                continue
            obj_dir = out_ann_root / video_id / object_names[obj_idx]
            obj_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(obj_dir / f"{frame_idx_24fps:05d}.png")
            written += 1

    return {
        "video_id": video_id,
        "annotation": str(ann_path),
        "objects_selected": len(selected_obj_indices),
        "masks_written": written,
    }


def main() -> None:
    args = parse_args()
    video_root = detect_video_root(args.shard_root)
    ann_root = detect_ann_root(args.shard_root)
    videos = discover_videos(video_root, ann_root, args.max_videos, args.use_auto)
    if not videos:
        raise RuntimeError(f"No benchmarkable videos under {args.shard_root}")

    image_root = args.out_root / "JPEGImages_24fps"
    out_ann_root = args.out_root / "Annotations_6fps"
    args.out_root.mkdir(parents=True, exist_ok=True)
    out_ann_root.mkdir(parents=True, exist_ok=True)
    extract_frames(video_root, image_root, args.frame_sample_rate, videos)

    summaries = []
    kept = []
    for video_id in videos:
        ann_path = choose_annotation(ann_root, video_id, args.use_auto)
        if ann_path is None:
            continue
        summary = write_masks(ann_path, out_ann_root, args.ann_every, args.max_objects_per_video)
        if summary["masks_written"] > 0:
            kept.append(video_id)
            summaries.append(summary)

    if not kept:
        raise RuntimeError("Decoded no non-empty masks.")

    (args.out_root / "sav_train_benchmark.txt").write_text("".join(f"{video}\n" for video in kept), encoding="utf-8")
    summary = {
        "status": "pass",
        "shard_root": str(args.shard_root),
        "video_root": str(video_root),
        "ann_root": str(ann_root),
        "out_root": str(args.out_root),
        "image_root": str(image_root),
        "annotation_root": str(out_ann_root),
        "video_list": str(args.out_root / "sav_train_benchmark.txt"),
        "videos": len(kept),
        "ann_every": args.ann_every,
        "video_summaries": summaries,
    }
    (args.out_root / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
