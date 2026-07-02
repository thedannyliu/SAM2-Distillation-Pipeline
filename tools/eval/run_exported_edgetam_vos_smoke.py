#!/usr/bin/env python3
"""Run VOS smoke on an exported EdgeTAM TinyViT checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sav-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--edgetam-root", required=True, type=Path)
    parser.add_argument("--sam2-training-root", required=True, type=Path)
    parser.add_argument("--video-list-file", type=Path)
    parser.add_argument("--use-all-masks", action="store_true")
    parser.add_argument("--per-obj-png-file", action="store_true", default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def add_import_roots(edgetam_root: Path, sam2_training_root: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    for root in (sam2_training_root, edgetam_root):
        if not root.exists():
            raise FileNotFoundError(root)
        sys.path.insert(0, str(root))


def load_vos_module(edgetam_root: Path):
    script = edgetam_root / "tools" / "vos_inference.py"
    if not script.exists():
        raise FileNotFoundError(script)
    spec = importlib.util.spec_from_file_location("edgetam_vos_inference", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_video_names(sav_root: Path, video_list_file: Path | None) -> list[str]:
    if video_list_file is not None:
        return [
            line.strip()
            for line in video_list_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return sorted(path.name for path in (sav_root / "JPEGImages_24fps").iterdir() if path.is_dir())


def load_video_predictor(config_path: Path, checkpoint_path: Path, device: torch.device):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    cfg.model._target_ = "sam2.sam2_video_predictor.SAM2VideoPredictor"
    cfg.model.fill_hole_area = 8
    cfg.model.binarize_mask_from_pts_for_mem_enc = True
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


def count_pngs(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.png"))


def main() -> None:
    args = parse_args()
    add_import_roots(args.edgetam_root, args.sam2_training_root)

    base_video_dir = args.sav_root / "JPEGImages_24fps"
    input_mask_dir = args.sav_root / "Annotations_6fps"
    if not base_video_dir.exists() or not input_mask_dir.exists():
        raise FileNotFoundError(f"SA-V smoke root must contain JPEGImages_24fps and Annotations_6fps: {args.sav_root}")

    device = torch.device(args.device)
    predictor, load_summary = load_video_predictor(args.model_config, args.checkpoint, device)
    vos_module = load_vos_module(args.edgetam_root)
    video_names = load_video_names(args.sav_root, args.video_list_file)
    if not video_names:
        raise RuntimeError("No videos selected for VOS smoke")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for video_name in video_names:
        vos_module.vos_inference(
            predictor=predictor,
            base_video_dir=str(base_video_dir),
            input_mask_dir=str(input_mask_dir),
            output_mask_dir=str(args.out_dir),
            video_name=video_name,
            use_all_masks=args.use_all_masks,
            per_obj_png_file=args.per_obj_png_file,
        )

    summary: dict[str, Any] = {
        "status": "pass",
        "model_config": str(args.model_config),
        "checkpoint": str(args.checkpoint),
        "sav_root": str(args.sav_root),
        "prediction_root": str(args.out_dir),
        "device": str(device),
        "load": load_summary,
        "video_names": video_names,
        "num_prediction_pngs": count_pngs(args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
