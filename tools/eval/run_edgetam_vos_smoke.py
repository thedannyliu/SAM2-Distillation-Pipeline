#!/usr/bin/env python3
"""Run official EdgeTAM VOS inference on the bounded SA-V smoke subset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edgetam-root", required=True, type=Path)
    parser.add_argument("--sam2-cfg", default="configs/edgetam.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sav-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--video-list-file", type=Path, default=None)
    parser.add_argument("--use-all-masks", action="store_true")
    parser.add_argument("--track-object-appearing-later-in-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = args.edgetam_root / "tools" / "vos_inference.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing EdgeTAM VOS script: {script}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing EdgeTAM checkpoint: {args.checkpoint}")

    base_video_dir = args.sav_root / "JPEGImages_24fps"
    input_mask_dir = args.sav_root / "Annotations_6fps"
    if not base_video_dir.exists() or not input_mask_dir.exists():
        raise FileNotFoundError(f"SA-V smoke root must contain JPEGImages_24fps and Annotations_6fps: {args.sav_root}")

    command = [
        sys.executable,
        str(script),
        "--sam2_cfg",
        args.sam2_cfg,
        "--sam2_checkpoint",
        str(args.checkpoint),
        "--base_video_dir",
        str(base_video_dir),
        "--input_mask_dir",
        str(input_mask_dir),
        "--output_mask_dir",
        str(args.out_dir),
        "--per_obj_png_file",
    ]
    if args.video_list_file is not None:
        command.extend(["--video_list_file", str(args.video_list_file)])
    if args.use_all_masks:
        command.append("--use_all_masks")
    if args.track_object_appearing_later_in_video:
        command.append("--track_object_appearing_later_in_video")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        check=False,
        cwd=str(args.edgetam_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    summary = {
        "status": "pass" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "command": " ".join(command),
        "output_tail": result.stdout[-4000:],
        "prediction_root": str(args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
