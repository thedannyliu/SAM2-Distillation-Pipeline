#!/usr/bin/env python3
"""Run image-predictor smoke on an exported EdgeTAM TinyViT checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--edgetam-root", required=True, type=Path)
    parser.add_argument("--sam2-training-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def add_import_roots(edgetam_root: Path, sam2_training_root: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    for root in (sam2_training_root, edgetam_root):
        if not root.exists():
            raise FileNotFoundError(root)
        sys.path.insert(0, str(root))


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    model = instantiate(cfg.model, _recursive_=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"]
    incompatible = model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, {
        "num_tensors": len(state_dict),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def main() -> None:
    args = parse_args()
    add_import_roots(args.edgetam_root, args.sam2_training_root)

    if not args.model_config.exists():
        raise FileNotFoundError(args.model_config)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.image.exists():
        raise FileNotFoundError(args.image)

    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device(args.device)
    model, load_summary = load_model(args.model_config, args.checkpoint, device)
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
        "model_config": str(args.model_config),
        "checkpoint": str(args.checkpoint),
        "image": str(args.image),
        "device": str(device),
        "load": load_summary,
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
