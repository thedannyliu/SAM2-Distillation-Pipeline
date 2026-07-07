#!/usr/bin/env python3
"""Create overlay videos and mask artifacts for VOS predictions."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--pred-root", required=True, type=Path)
    parser.add_argument("--video-list-file", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-videos", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--fps", type=float, default=12.0)
    return parser.parse_args()


def video_names(image_root: Path, video_list_file: Path | None, max_videos: int) -> list[str]:
    if video_list_file is not None:
        names = [line.strip() for line in video_list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        names = sorted(path.name for path in image_root.iterdir() if path.is_dir())
    names = [name for name in names if (image_root / name).is_dir()]
    return names[:max_videos] if max_videos > 0 else names


def read_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image) > 0


def frame_for_mask(image_dir: Path, frame_stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = image_dir / f"{frame_stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def object_mask_pairs(gt_video: Path, pred_video: Path) -> list[tuple[str, Path, Path]]:
    pairs = []
    for gt_path in sorted(gt_video.glob("*/*.png")):
        pred_path = pred_video / gt_path.relative_to(gt_video)
        if pred_path.exists():
            pairs.append((gt_path.stem, gt_path, pred_path))
    return pairs


def group_pairs_by_frame(pairs: list[tuple[str, Path, Path]]) -> dict[str, list[tuple[Path, Path]]]:
    grouped: dict[str, list[tuple[Path, Path]]] = {}
    for frame_stem, gt_path, pred_path in pairs:
        grouped.setdefault(frame_stem, []).append((gt_path, pred_path))
    return grouped


def frame_sort_key(frame_stem: str) -> tuple[int, str]:
    return (int(frame_stem), frame_stem) if frame_stem.isdigit() else (10**12, frame_stem)


def make_overlay(image_bgr: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    gt_only = np.logical_and(gt, np.logical_not(pred))
    pred_only = np.logical_and(pred, np.logical_not(gt))
    both = np.logical_and(gt, pred)
    color = np.zeros_like(overlay)
    color[gt_only] = [0, 255, 0]
    color[pred_only] = [0, 0, 255]
    color[both] = [0, 220, 255]
    mask = np.logical_or(gt, pred)
    overlay[mask] = cv2.addWeighted(overlay, 0.45, color, 0.55, 0)[mask]
    return overlay


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for video in video_names(args.image_root, args.video_list_file, args.max_videos):
        image_dir = args.image_root / video
        gt_video = args.gt_root / video
        pred_video = args.pred_root / video
        pairs = object_mask_pairs(gt_video, pred_video)
        if not pairs:
            continue
        grouped = group_pairs_by_frame(pairs)
        first_image = None
        for frame_stem in sorted(grouped):
            first_image = frame_for_mask(image_dir, frame_stem)
            if first_image is not None:
                break
        if first_image is None:
            continue
        first_bgr = cv2.imread(str(first_image), cv2.IMREAD_COLOR)
        if first_bgr is None:
            continue
        h, w = first_bgr.shape[:2]
        video_out = args.out_dir / f"{video}_overlay.mp4"
        writer = cv2.VideoWriter(
            str(video_out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (w, h),
        )
        frames_written = 0
        mask_copy_root = args.out_dir / "masks" / video
        for frame_stem in sorted(grouped, key=frame_sort_key):
            if args.max_frames > 0 and frames_written >= args.max_frames:
                break
            image_path = frame_for_mask(image_dir, frame_stem)
            if image_path is None:
                continue
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            gt_union = np.zeros(image_bgr.shape[:2], dtype=bool)
            pred_union = np.zeros(image_bgr.shape[:2], dtype=bool)
            for gt_path, pred_path in grouped[frame_stem]:
                gt = read_binary_mask(gt_path)
                pred = read_binary_mask(pred_path)
                if gt.shape != image_bgr.shape[:2]:
                    gt = cv2.resize(gt.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                if pred.shape != image_bgr.shape[:2]:
                    pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                gt_union |= gt
                pred_union |= pred
                rel = gt_path.relative_to(gt_video)
                (mask_copy_root / "gt" / rel.parent).mkdir(parents=True, exist_ok=True)
                (mask_copy_root / "pred" / rel.parent).mkdir(parents=True, exist_ok=True)
                shutil.copy2(gt_path, mask_copy_root / "gt" / rel)
                shutil.copy2(pred_path, mask_copy_root / "pred" / rel)
            writer.write(make_overlay(image_bgr, gt_union, pred_union))
            frames_written += 1
        writer.release()
        summaries.append({"video": video, "overlay_video": str(video_out), "frames": frames_written})

    summary = {"status": "pass", "out_dir": str(args.out_dir), "videos": summaries}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
