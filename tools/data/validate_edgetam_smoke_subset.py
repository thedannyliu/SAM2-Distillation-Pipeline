#!/usr/bin/env python3
"""Validate EdgeTAM smoke subsets without requiring GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def decode_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def validate_image_manifest(path: Path, max_items: int) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) > max_items:
        raise SystemExit(f"{path} has {len(rows)} rows, exceeds cap {max_items}")
    for row in rows:
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise SystemExit(f"missing image: {image_path}")
        width, height = decode_image(image_path)
        if width <= 0 or height <= 0:
            raise SystemExit(f"invalid image shape: {image_path}")
        ann = row.get("annotation_path")
        if ann and not Path(ann).exists():
            raise SystemExit(f"missing annotation: {ann}")
    return {"rows": len(rows), "manifest": str(path)}


def validate_sav_val(root: Path, max_frames: int) -> dict:
    image_root = root / "JPEGImages_24fps"
    ann_root = root / "Annotations_6fps"
    if not image_root.exists() or not ann_root.exists():
        raise SystemExit(f"{root} must contain JPEGImages_24fps and Annotations_6fps")
    total = 0
    videos = 0
    for video_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        frames = sorted(video_dir.glob("*.jpg"))
        if not frames:
            raise SystemExit(f"video has no frames: {video_dir}")
        for frame in frames:
            decode_image(frame)
        if not (ann_root / video_dir.name).exists():
            raise SystemExit(f"missing annotation dir for {video_dir.name}")
        total += len(frames)
        videos += 1
    if total > max_frames:
        raise SystemExit(f"SA-V subset has {total} frames, exceeds cap {max_frames}")
    return {"videos": videos, "frames": total, "root": str(root)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", action="append", default=[])
    parser.add_argument("--sav-val-root")
    parser.add_argument("--max-items", type=int, default=500)
    parser.add_argument("--max-frames", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {"image_manifests": []}
    for manifest in args.image_manifest:
        report["image_manifests"].append(validate_image_manifest(Path(manifest), args.max_items))
    if args.sav_val_root:
        report["sav_val"] = validate_sav_val(Path(args.sav_val_root), args.max_frames)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

