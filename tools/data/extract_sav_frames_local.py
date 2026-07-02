#!/usr/bin/env python3
"""Extract JPEG frames from a local SA-V video shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=1)
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all videos.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_rate < 1:
        raise SystemExit("--sample-rate must be >= 1")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("extract_sav_frames_local.py requires opencv-python / cv2") from exc

    videos = sorted(args.video_root.rglob("*.mp4"))
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    if not videos:
        raise SystemExit(f"No .mp4 videos found under {args.video_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {"video_root": str(args.video_root), "output_root": str(args.output_root), "videos": []}
    for video_path in videos:
        out_dir = args.output_root / video_path.stem
        if out_dir.exists() and not args.overwrite and any(out_dir.glob("*.jpg")):
            summary["videos"].append({"video": video_path.stem, "status": "skip-existing"})
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        frame_idx = 0
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.sample_rate == 0:
                out_path = out_dir / f"{frame_idx:05d}.jpg"
                if not cv2.imwrite(str(out_path), frame):
                    raise RuntimeError(f"Failed to write {out_path}")
                written += 1
            frame_idx += 1
        cap.release()
        summary["videos"].append(
            {"video": video_path.stem, "frames_read": frame_idx, "frames_written": written}
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
