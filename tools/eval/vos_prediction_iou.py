#!/usr/bin/env python3
"""Score packed indexed VOS predictions with mean per-object IoU."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--pred-root", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--max-frames", type=int, default=500)
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def object_iou(gt: np.ndarray, pred: np.ndarray, object_id: int) -> float:
    gt_mask = gt == object_id
    pred_mask = pred == object_id
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(gt_mask, pred_mask).sum() / union)


def main() -> None:
    args = parse_args()
    if args.max_frames < 1 or args.max_frames > 500:
        raise SystemExit("--max-frames must be in [1, 500] for smoke tests")
    for path in (args.gt_root, args.pred_root):
        if not path.exists():
            raise FileNotFoundError(path)

    rows = []
    for gt_path in sorted(args.gt_root.glob("*/*.png")):
        if len(rows) >= args.max_frames:
            break
        pred_path = args.pred_root / gt_path.relative_to(args.gt_root)
        if not pred_path.exists():
            continue
        gt = load_mask(gt_path)
        pred = load_mask(pred_path)
        object_ids = [int(value) for value in np.unique(gt) if value > 0]
        for object_id in object_ids:
            rows.append(
                {
                    "video": gt_path.parent.name,
                    "frame": gt_path.stem,
                    "object_id": object_id,
                    "iou": object_iou(gt, pred, object_id),
                }
            )

    if not rows:
        raise RuntimeError(f"No matching indexed PNG predictions under {args.pred_root}")

    mean_iou = float(sum(row["iou"] for row in rows) / len(rows))
    videos = sorted({row["video"] for row in rows})
    summary = {
        "status": "pass",
        "gt_root": str(args.gt_root),
        "pred_root": str(args.pred_root),
        "videos": len(videos),
        "objects_scored": len(rows),
        "mean_object_iou": mean_iou,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["video", "frame", "object_id", "iou"])
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
