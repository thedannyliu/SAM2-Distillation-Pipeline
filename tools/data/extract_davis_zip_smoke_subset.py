#!/usr/bin/env python3
"""Extract a bounded DAVIS 2017 smoke subset directly from the official zip."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from PIL import Image


def require_cap(value: int) -> None:
    if value < 1 or value > 500:
        raise SystemExit(f"--max-frames must be in [1, 500] for smoke subsets; got {value}")


def parse_video_frame(path: str, prefix: str, suffix: str) -> tuple[str, str] | None:
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    rest = path[len(prefix) :]
    parts = rest.split("/")
    if len(parts) != 2:
        return None
    return parts[0], Path(parts[1]).stem


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument(
        "--resolution-dir",
        default="auto",
        help="DAVIS resolution folder to read, e.g. 480p or Full-Resolution. Use auto to detect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_cap(args.max_frames)
    args.out_root.mkdir(parents=True, exist_ok=True)

    images = {}
    anns = {}
    with zipfile.ZipFile(args.zip) as archive:
        names = archive.namelist()
        if args.resolution_dir == "auto":
            resolution_dir = None
            for candidate in ("480p", "Full-Resolution"):
                image_prefix = f"DAVIS/JPEGImages/{candidate}/"
                ann_prefix = f"DAVIS/Annotations/{candidate}/"
                has_images = any(name.startswith(image_prefix) and name.endswith(".jpg") for name in names)
                has_anns = any(name.startswith(ann_prefix) and name.endswith(".png") for name in names)
                if has_images and has_anns:
                    resolution_dir = candidate
                    break
            if resolution_dir is None:
                raise SystemExit(f"Could not find DAVIS JPEGImages/Annotations resolution folders in {args.zip}")
        else:
            resolution_dir = args.resolution_dir

        image_prefix = f"DAVIS/JPEGImages/{resolution_dir}/"
        ann_prefix = f"DAVIS/Annotations/{resolution_dir}/"
        for name in names:
            image_key = parse_video_frame(name, image_prefix, ".jpg")
            if image_key:
                images[image_key] = name
                continue
            ann_key = parse_video_frame(name, ann_prefix, ".png")
            if ann_key:
                anns[ann_key] = name

        rows = []
        for video_id, frame_id in sorted(set(images) & set(anns)):
            image_dst = args.out_root / "JPEGImages" / video_id / f"{frame_id}.jpg"
            ann_dst = args.out_root / "Annotations" / video_id / f"{frame_id}.png"
            image_dst.parent.mkdir(parents=True, exist_ok=True)
            ann_dst.parent.mkdir(parents=True, exist_ok=True)
            if not image_dst.exists():
                image_dst.write_bytes(archive.read(images[(video_id, frame_id)]))
            if not ann_dst.exists():
                ann_dst.write_bytes(archive.read(anns[(video_id, frame_id)]))
            width, height = image_size(image_dst)
            rows.append(
                {
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "image_path": str(image_dst),
                    "mask_path": str(ann_dst),
                    "height": height,
                    "width": width,
                    "source": f"davis2017-trainval-{resolution_dir}",
                }
            )
            if len(rows) >= args.max_frames:
                break

    if not rows:
        raise SystemExit(f"No paired DAVIS frames found in {args.zip}")
    manifest = args.out_root / "manifests" / "vos_smoke_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    videos = sorted({row["video_id"] for row in rows})
    (args.out_root / "val.txt").write_text("".join(f"{video}\n" for video in videos), encoding="utf-8")
    metadata = args.out_root / "metadata.json"
    metadata.write_text(
        json.dumps({"resolution_dir": resolution_dir, "videos": len(videos), "frames": len(rows)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "videos": len(videos),
                "frames": len(rows),
                "resolution_dir": resolution_dir,
                "manifest": str(manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
