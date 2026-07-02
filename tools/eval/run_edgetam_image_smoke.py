#!/usr/bin/env python3
"""Run official EdgeTAM image predictor on one real smoke image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edgetam-root", required=True, type=Path)
    parser.add_argument("--sam2-cfg", default="configs/edgetam.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.edgetam_root.exists():
        raise FileNotFoundError(f"Missing EdgeTAM root: {args.edgetam_root}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    if not args.image.exists():
        raise FileNotFoundError(f"Missing image: {args.image}")

    sys.path.insert(0, str(args.edgetam_root))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device(args.device)
    model = build_sam2(args.sam2_cfg, str(args.checkpoint), device=str(device))
    predictor = SAM2ImagePredictor(model)
    with Image.open(args.image) as image:
        image = image.convert("RGB")
        width, height = image.size
        image_np = np.asarray(image)

    predictor.set_image(image_np)
    point_coords = np.array([[width / 2.0, height / 2.0]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    masks, scores, low_res_masks = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
        normalize_coords=False,
    )

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(np.uint8) * 255
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.out_dir / "best_mask.png"
    Image.fromarray(best_mask).save(mask_path)
    summary = {
        "status": "pass",
        "image": str(args.image),
        "device": str(device),
        "masks_shape": list(masks.shape),
        "low_res_masks_shape": list(low_res_masks.shape),
        "scores": [float(score) for score in scores],
        "best_idx": best_idx,
        "best_mask_area": int(best_mask.astype(bool).sum()),
        "mask_path": str(mask_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
