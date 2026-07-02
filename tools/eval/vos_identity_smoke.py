#!/usr/bin/env python3
"""Copy DAVIS-style VOS annotations as predictions and score mask IoU."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def mean_object_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    object_ids = sorted(int(value) for value in np.unique(gt) if value > 0)
    if not object_ids:
        return 1.0
    scores = []
    for object_id in object_ids:
        gt_mask = gt == object_id
        pred_mask = pred == object_id
        union = np.logical_or(gt_mask, pred_mask).sum()
        scores.append(1.0 if union == 0 else float(np.logical_and(gt_mask, pred_mask).sum() / union))
    return float(sum(scores) / len(scores))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_frames < 1 or args.max_frames > 500:
        raise SystemExit("--max-frames must be in [1, 500] for smoke tests")
    if not args.annotation_root.exists():
        raise FileNotFoundError(f"Missing annotation root: {args.annotation_root}")

    copied = []
    for video_dir in sorted(path for path in args.annotation_root.iterdir() if path.is_dir()):
        for mask_path in sorted(video_dir.glob("*.png")):
            rel = mask_path.relative_to(args.annotation_root)
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
        raise RuntimeError(f"No masks found under {args.annotation_root}")

    check_count = min(50, len(copied))
    ious = [mean_object_iou(load_mask(gt), load_mask(pred)) for gt, pred in copied[:check_count]]
    summary = {
        "status": "pass",
        "frames_copied": len(copied),
        "checked_frames": check_count,
        "mean_object_iou_checked": sum(ious) / len(ious),
        "prediction_root": str(args.out_dir),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
