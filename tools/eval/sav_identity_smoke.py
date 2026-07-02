#!/usr/bin/env python3
"""Copy SA-V annotations as predictions and validate the eval file layout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def mask_iou(a_path: Path, b_path: Path) -> float:
    with Image.open(a_path) as a_image, Image.open(b_path) as b_image:
        a = np.asarray(a_image) > 0
        b = np.asarray(b_image) > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sav-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--filelist", type=Path, default=None)
    parser.add_argument("--evaluator", type=Path, default=None, help="Optional path to SAM2 sav_evaluator.py.")
    parser.add_argument("--num-processes", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_frames < 1 or args.max_frames > 500:
        raise SystemExit("--max-frames must be in [1, 500] for smoke tests")

    ann_root = args.sav_root / "Annotations_6fps"
    if not ann_root.exists():
        raise FileNotFoundError(f"Missing SA-V annotation root: {ann_root}")

    if args.filelist and args.filelist.exists():
        videos = [line.strip() for line in args.filelist.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        videos = sorted(path.name for path in ann_root.iterdir() if path.is_dir())

    copied = []
    for video in videos:
        for mask_path in sorted((ann_root / video).glob("*/*.png")):
            rel = mask_path.relative_to(ann_root)
            dst = args.out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(mask_path, dst)
            copied.append((mask_path, dst))
            if len(copied) >= args.max_frames:
                break
        if len(copied) >= args.max_frames:
            break

    if not copied:
        raise RuntimeError(f"No masks found under {ann_root}")

    check_count = min(20, len(copied))
    ious = [mask_iou(gt, pred) for gt, pred in copied[:check_count]]
    summary = {
        "status": "pass",
        "videos": videos,
        "masks_copied": len(copied),
        "checked_masks": check_count,
        "mean_iou_checked": sum(ious) / len(ious),
        "prediction_root": str(args.out_dir),
    }
    if args.evaluator is not None:
        command = [
            sys.executable,
            str(args.evaluator),
            "--gt_root",
            str(ann_root),
            "--pred_root",
            str(args.out_dir),
            "--num_processes",
            str(args.num_processes),
            "--strict",
        ]
        result = subprocess.run(
            command,
            check=False,
            cwd=str(args.evaluator.parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        summary["official_evaluator"] = {
            "command": " ".join(command),
            "returncode": result.returncode,
            "output_tail": result.stdout[-4000:],
        }
        if result.returncode != 0:
            summary["status"] = "official_evaluator_failed"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary["status"] != "pass":
        raise SystemExit(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
