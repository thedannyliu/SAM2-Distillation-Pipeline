#!/usr/bin/env python3
"""Audit the immutable inputs for SAM2.1-L two-clock Wave 1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sav-root", required=True, type=Path)
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--sam2-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sample-videos", type=int, default=32)
    parser.add_argument("--decode-videos", type=int, default=2)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = (
        args.manifest,
        args.sav_root / "sav_train",
        args.sam2_root / "training/model/sam2.py",
        args.sam2_config,
        args.checkpoint,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(args.sam2_root))
    import sam2

    from sam2_distill.two_clock.data import (
        SAV24FPSRawDataset,
        decode_contiguous_rgb,
    )

    dataset = SAV24FPSRawDataset(
        manifest=args.manifest,
        sav_root=args.sav_root,
        split="train",
        verify_paths=False,
    )
    sample_count = min(args.sample_videos, len(dataset))
    indices = sorted(
        {
            round(index * (len(dataset) - 1) / max(sample_count - 1, 1))
            for index in range(sample_count)
        }
    )
    failures = []
    decoded = []
    usable = 0
    for sample_index, index in enumerate(indices):
        video, loader = dataset.get_video(index)
        video_path = Path(video.frames[0].image_path)
        if not video_path.is_file():
            failures.append(f"missing video: {video_path}")
            continue
        anchors = loader.valid_anchor_frames(len(video.frames), 8)
        if not anchors:
            failures.append(f"no T8 annotated anchor: {video.video_name}")
            continue
        usable += 1
        if sample_index < args.decode_videos:
            frames = decode_contiguous_rgb(video_path, list(range(anchors[0], anchors[0] + 8)))
            decoded.append(
                {
                    "video": video.video_name,
                    "anchor": anchors[0],
                    "frames": len(frames),
                    "size": list(frames[0].size),
                }
            )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    expected_prefixes = (
        "image_encoder.",
        "memory_attention.",
        "memory_encoder.",
        "sam_mask_decoder.",
    )
    missing_prefixes = [
        prefix for prefix in expected_prefixes if not any(key.startswith(prefix) for key in state)
    ]
    if missing_prefixes:
        failures.append(f"checkpoint missing prefixes: {missing_prefixes}")

    payload = {
        "status": "pass" if not failures else "fail",
        "torch_version": torch.__version__,
        "sam2_package": str(Path(sam2.__file__).resolve()),
        "manifest": str(args.manifest),
        "dataset_records": len(dataset),
        "sampled_records": len(indices),
        "usable_sampled_records": usable,
        "decoded": decoded,
        "checkpoint": str(args.checkpoint),
        "checkpoint_tensors": len(state),
        "failures": failures[:50],
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
