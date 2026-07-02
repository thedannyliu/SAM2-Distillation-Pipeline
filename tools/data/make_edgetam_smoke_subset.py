#!/usr/bin/env python3
"""Create <=500-item real-data smoke subsets for EdgeTAM development."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def require_cap(value: int, name: str) -> None:
    if value < 1 or value > 500:
        raise SystemExit(f"{name} must be in [1, 500] for PACE smoke subsets; got {value}")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def make_sa1b(args: argparse.Namespace) -> None:
    require_cap(args.max_items, "--max-items")
    src = Path(args.source_root).expanduser().resolve()
    out = Path(args.out_root).expanduser().resolve()
    rows = []

    train_count = min(args.train_count, args.max_items)
    val_count = min(args.val_count, args.max_items - train_count)
    for split, count in (("train", train_count), ("val", val_count)):
        images = sorted((src / "images" / split).glob("*.jpg"))[:count]
        for image_path in images:
            stem = image_path.stem
            ann_path = src / "annotations" / split / f"{stem}.json"
            if not ann_path.exists():
                continue
            dst_image = out / "images" / split / image_path.name
            dst_ann = out / "annotations" / split / ann_path.name
            copy_file(image_path, dst_image)
            copy_file(ann_path, dst_ann)
            width, height = image_size(dst_image)
            rows.append(
                {
                    "sample_id": stem,
                    "source": "sa1b_smoke",
                    "image_path": str(dst_image),
                    "annotation_path": str(dst_ann),
                    "height": height,
                    "width": width,
                    "split": split,
                }
            )

    manifest = out / "manifests" / "sa1b_smoke_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({"dataset": "sa1b", "rows": len(rows), "manifest": str(manifest)}, indent=2))


def make_sav_val(args: argparse.Namespace) -> None:
    require_cap(args.max_frames, "--max-frames")
    src = Path(args.source_root).expanduser().resolve()
    out = Path(args.out_root).expanduser().resolve()
    src_images = src / "JPEGImages_24fps"
    src_anns = src / "Annotations_6fps"
    selected = []
    total_frames = 0

    for video_dir in sorted(p for p in src_images.iterdir() if p.is_dir()):
        frames = sorted(video_dir.glob("*.jpg"))
        if not frames:
            continue
        if total_frames + len(frames) > args.max_frames and selected:
            continue
        if len(frames) > args.max_frames:
            frames = frames[: args.max_frames]
        dst_video = out / "JPEGImages_24fps" / video_dir.name
        for frame in frames:
            copy_file(frame, dst_video / frame.name)
        ann_dir = src_anns / video_dir.name
        if ann_dir.exists():
            shutil.copytree(ann_dir, out / "Annotations_6fps" / video_dir.name, dirs_exist_ok=True)
        selected.append(video_dir.name)
        total_frames += len(frames)
        if total_frames >= args.max_frames:
            break

    filelist = out / "sav_val.txt"
    filelist.write_text("".join(f"{name}\n" for name in selected), encoding="utf-8")
    manifest = out / "official_subset_manifest.json"
    manifest.write_text(json.dumps({"videos": selected, "frames": total_frames}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": "sav_val", "videos": len(selected), "frames": total_frames, "filelist": str(filelist)}, indent=2))


def make_coco(args: argparse.Namespace) -> None:
    require_cap(args.max_items, "--max-items")
    src = Path(args.source_root).expanduser().resolve()
    out = Path(args.out_root).expanduser().resolve()
    rows = []
    image_root = src / "images" / "val2017"
    if not image_root.exists():
        image_root = src / "val2017"
    for image_path in sorted(p for p in image_root.iterdir() if p.suffix.lower() in IMAGE_EXTS)[: args.max_items]:
        dst = out / "images" / "val2017" / image_path.name
        copy_file(image_path, dst)
        width, height = image_size(dst)
        rows.append({"sample_id": image_path.stem, "image_path": str(dst), "height": height, "width": width, "split": "val"})
    if (src / "annotations").exists():
        shutil.copytree(src / "annotations", out / "annotations", dirs_exist_ok=True)
    manifest = out / "manifests" / "coco_smoke_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"dataset": "coco", "rows": len(rows), "manifest": str(manifest)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sa1b = sub.add_parser("sa1b")
    sa1b.add_argument("--source-root", required=True)
    sa1b.add_argument("--out-root", required=True)
    sa1b.add_argument("--max-items", type=int, default=500)
    sa1b.add_argument("--train-count", type=int, default=400)
    sa1b.add_argument("--val-count", type=int, default=100)
    sa1b.set_defaults(func=make_sa1b)

    sav = sub.add_parser("sav-val")
    sav.add_argument("--source-root", required=True, help="Root containing JPEGImages_24fps and Annotations_6fps.")
    sav.add_argument("--out-root", required=True)
    sav.add_argument("--max-frames", type=int, default=500)
    sav.set_defaults(func=make_sav_val)

    coco = sub.add_parser("coco")
    coco.add_argument("--source-root", required=True)
    coco.add_argument("--out-root", required=True)
    coco.add_argument("--max-items", type=int, default=500)
    coco.set_defaults(func=make_coco)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

