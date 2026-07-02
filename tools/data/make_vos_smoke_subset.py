#!/usr/bin/env python3
"""Create bounded VOS smoke subsets in DAVIS-style layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def require_cap(value: int, name: str) -> None:
    if value < 1 or value > 500:
        raise SystemExit(f"{name} must be in [1, 500] for smoke subsets; got {value}")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def pack_per_object_masks(object_mask_paths: list[Path], out_path: Path) -> None:
    packed = None
    palette = None
    for object_idx, mask_path in enumerate(object_mask_paths, start=1):
        with Image.open(mask_path) as mask_image:
            if palette is None:
                palette = mask_image.getpalette()
            mask = np.asarray(mask_image) > 0
        if packed is None:
            packed = np.zeros(mask.shape, dtype=np.uint8)
        packed[mask] = object_idx
    if packed is None:
        raise ValueError("cannot pack empty mask list")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(packed)
    if palette:
        image.putpalette(palette)
    image.save(out_path)


def select_videos(image_root: Path, max_frames: int) -> tuple[list[Path], int]:
    selected = []
    total = 0
    for video_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        frames = sorted(path for path in video_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
        if not frames:
            continue
        if selected and total + len(frames) > max_frames:
            continue
        selected.append(video_dir)
        total += min(len(frames), max_frames - total)
        if total >= max_frames:
            break
    if not selected:
        raise SystemExit(f"No videos found under {image_root}")
    return selected, total


def make_davis_style(args: argparse.Namespace) -> None:
    require_cap(args.max_frames, "--max-frames")
    image_root = Path(args.image_root).resolve()
    ann_root = Path(args.annotation_root).resolve()
    out = Path(args.out_root).resolve()
    selected, _ = select_videos(image_root, args.max_frames)
    rows = []
    total_frames = 0

    for video_dir in selected:
        frames = sorted(path for path in video_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
        for frame in frames:
            if total_frames >= args.max_frames:
                break
            ann = ann_root / video_dir.name / f"{frame.stem}.png"
            if not ann.exists():
                continue
            dst_image = out / "JPEGImages" / video_dir.name / frame.name
            dst_ann = out / "Annotations" / video_dir.name / ann.name
            copy_file(frame, dst_image)
            copy_file(ann, dst_ann)
            width, height = image_size(dst_image)
            rows.append(
                {
                    "video_id": video_dir.name,
                    "frame_id": frame.stem,
                    "image_path": str(dst_image),
                    "mask_path": str(dst_ann),
                    "height": height,
                    "width": width,
                }
            )
            total_frames += 1
        if total_frames >= args.max_frames:
            break

    write_manifest(out, rows)


def make_from_sav(args: argparse.Namespace) -> None:
    require_cap(args.max_frames, "--max-frames")
    sav_root = Path(args.sav_root).resolve()
    image_root = sav_root / "JPEGImages_24fps"
    ann_root = sav_root / "Annotations_6fps"
    out = Path(args.out_root).resolve()
    selected, _ = select_videos(image_root, args.max_frames)
    rows = []
    total_frames = 0

    for video_dir in selected:
        object_dirs = sorted(path for path in (ann_root / video_dir.name).iterdir() if path.is_dir())
        frame_ids = sorted({path.stem for object_dir in object_dirs for path in object_dir.glob("*.png")})
        for frame_id in frame_ids:
            if total_frames >= args.max_frames:
                break
            src_image = image_root / video_dir.name / f"{frame_id}.jpg"
            if not src_image.exists():
                continue
            object_masks = [object_dir / f"{frame_id}.png" for object_dir in object_dirs if (object_dir / f"{frame_id}.png").exists()]
            if not object_masks:
                continue
            dst_image = out / "JPEGImages" / video_dir.name / src_image.name
            dst_ann = out / "Annotations" / video_dir.name / f"{frame_id}.png"
            copy_file(src_image, dst_image)
            pack_per_object_masks(object_masks, dst_ann)
            width, height = image_size(dst_image)
            rows.append(
                {
                    "video_id": video_dir.name,
                    "frame_id": frame_id,
                    "image_path": str(dst_image),
                    "mask_path": str(dst_ann),
                    "height": height,
                    "width": width,
                    "source": "sav-packed",
                }
            )
            total_frames += 1
        if total_frames >= args.max_frames:
            break

    write_manifest(out, rows)


def write_manifest(out: Path, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit("No VOS rows selected")
    manifest = out / "manifests" / "vos_smoke_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    videos = sorted({row["video_id"] for row in rows})
    (out / "val.txt").write_text("".join(f"{video}\n" for video in videos), encoding="utf-8")
    print(json.dumps({"videos": len(videos), "frames": len(rows), "manifest": str(manifest)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    davis = sub.add_parser("davis-style")
    davis.add_argument("--image-root", required=True, help="Root containing {video}/{frame}.jpg.")
    davis.add_argument("--annotation-root", required=True, help="Root containing {video}/{frame}.png.")
    davis.add_argument("--out-root", required=True)
    davis.add_argument("--max-frames", type=int, default=500)
    davis.set_defaults(func=make_davis_style)

    sav = sub.add_parser("sav-to-davis-style")
    sav.add_argument("--sav-root", required=True, help="Root containing JPEGImages_24fps and Annotations_6fps.")
    sav.add_argument("--out-root", required=True)
    sav.add_argument("--max-frames", type=int, default=500)
    sav.set_defaults(func=make_from_sav)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
