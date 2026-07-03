#!/usr/bin/env python3
"""Prepare a symlinked SA-V shard range for SAM2 JSONRawDataset training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sav-root", type=Path, default=Path("/group-volume/danny-dataset/SA-V"))
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=5)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--extract-missing-frames", action="store_true")
    parser.add_argument("--frame-sample-rate", type=int, default=1)
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


def detect_image_root(shard: Path) -> Path:
    found = first_existing_dir(
        [
            shard / "JPEGImages_24fps",
            shard / "train" / "JPEGImages_24fps",
            shard / "JPEGImages",
            shard / "frames",
        ]
    )
    return found if found is not None else shard / "JPEGImages_24fps"


def detect_ann_root(shard: Path) -> Path:
    found = first_existing_dir(
        [
            shard / "annotations",
            shard / "train" / "annotations",
            shard.parent / "train" / "annotations",
            shard.parent / "annotations",
            shard / "Annotations",
            shard.parent / "Annotations",
            shard,
        ]
    )
    if found is not None and any(found.glob("*_manual.json")):
        return found
    found = first_parent_dir_with_file([shard, shard.parent], "*_manual.json")
    if found is None:
        raise FileNotFoundError(f"Could not find *_manual.json for {shard}")
    return found


def detect_video_root(shard: Path) -> Path:
    found = first_existing_dir(
        [
            shard / "videos",
            shard / "train" / "videos",
            shard.parent / "train" / "videos",
            shard.parent / "videos",
            shard,
        ]
    )
    if found is not None and any(found.rglob("*.mp4")):
        return found
    found = first_parent_dir_with_file([shard, shard.parent], "*.mp4")
    if found is None:
        raise FileNotFoundError(f"Could not find mp4 videos for {shard}")
    return found


def extract_frames(video_root: Path, image_root: Path, sample_rate: int) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Frame extraction requires cv2/opencv-python.") from exc

    videos = sorted(video_root.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No mp4 files under {video_root}")
    image_root.mkdir(parents=True, exist_ok=True)
    for video in videos:
        out_dir = image_root / video.stem
        if out_dir.exists() and any(out_dir.glob("*.jpg")):
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video}")
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


def symlink_force(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        dst.unlink()
    dst.symlink_to(src)


def main() -> None:
    args = parse_args()
    image_out = args.out_root / "JPEGImages_24fps"
    ann_out = args.out_root / "annotations"
    manifest = args.out_root / "manifests" / "sav_train_filelist.txt"
    image_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    videos: list[str] = []
    shard_summaries = []
    for shard_id in range(args.start_shard, args.end_shard + 1):
        shard = args.sav_root / f"sav_{shard_id:03d}"
        if not shard.is_dir():
            raise FileNotFoundError(shard)
        image_root = detect_image_root(shard)
        ann_root = detect_ann_root(shard)
        if not any(image_root.glob("*/*.jpg")):
            if not args.extract_missing_frames:
                raise FileNotFoundError(
                    f"No JPEG frames under {image_root}. Re-run with --extract-missing-frames."
                )
            extract_frames(detect_video_root(shard), image_root, args.frame_sample_rate)

        shard_videos = []
        for video_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
            ann = ann_root / f"{video_dir.name}_manual.json"
            if not ann.exists() or not any(video_dir.glob("*.jpg")):
                continue
            if (image_out / video_dir.name).exists() and not (image_out / video_dir.name).is_symlink():
                raise FileExistsError(image_out / video_dir.name)
            if video_dir.name in videos:
                raise ValueError(f"Duplicate video id across shards: {video_dir.name}")
            symlink_force(video_dir.resolve(), image_out / video_dir.name)
            symlink_force(ann.resolve(), ann_out / ann.name)
            videos.append(video_dir.name)
            shard_videos.append(video_dir.name)
        shard_summaries.append(
            {
                "shard": shard.name,
                "image_root": str(image_root),
                "ann_root": str(ann_root),
                "videos": len(shard_videos),
            }
        )

    if not videos:
        raise RuntimeError("No videos with frames and *_manual.json were found.")
    manifest.write_text("".join(f"{video}\n" for video in videos), encoding="utf-8")
    summary = {
        "sav_root": str(args.sav_root),
        "out_root": str(args.out_root),
        "start_shard": args.start_shard,
        "end_shard": args.end_shard,
        "videos": len(videos),
        "image_root": str(image_out),
        "ann_root": str(ann_out),
        "file_list": str(manifest),
        "shards": shard_summaries,
    }
    (args.out_root / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
