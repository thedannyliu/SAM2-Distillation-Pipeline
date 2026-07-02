#!/usr/bin/env python3
"""Run official EdgeTAM VOS inference on a generic VOS dataset layout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edgetam-root", required=True, type=Path)
    parser.add_argument("--sam2-cfg", default="configs/edgetam.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--input-mask-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--video-list-file", type=Path)
    parser.add_argument("--use-all-masks", action="store_true")
    parser.add_argument("--per-obj-png-file", action="store_true")
    parser.add_argument("--track-object-appearing-later-in-video", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def count_pngs(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.png"))


def add_import_roots(edgetam_root: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(edgetam_root))


def load_vos_module(edgetam_root: Path):
    script = edgetam_root / "tools" / "vos_inference.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing EdgeTAM VOS script: {script}")
    spec = importlib.util.spec_from_file_location("edgetam_vos_inference", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_video_names(image_root: Path, video_list_file: Path | None) -> list[str]:
    if video_list_file is not None:
        return [
            line.strip()
            for line in video_list_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return sorted(path.name for path in image_root.iterdir() if path.is_dir())


def main() -> None:
    args = parse_args()
    for path in (args.edgetam_root, args.checkpoint, args.image_root, args.input_mask_root):
        if not path.exists():
            raise FileNotFoundError(path)

    add_import_roots(args.edgetam_root)
    from sam2.build_sam import build_sam2_video_predictor
    from sam2_distill.edgetam.compat import patch_edgetam_perceiver_view

    patch_edgetam_perceiver_view()
    vos_module = load_vos_module(args.edgetam_root)
    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("false" if args.per_obj_png_file else "true")
    ]
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=str(args.checkpoint),
        device=args.device,
        apply_postprocessing=False,
        hydra_overrides_extra=hydra_overrides_extra,
    )
    video_names = load_video_names(args.image_root, args.video_list_file)
    if not video_names:
        raise RuntimeError(f"No videos selected under {args.image_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    inference_fn = (
        vos_module.vos_separate_inference_per_object
        if args.track_object_appearing_later_in_video
        else vos_module.vos_inference
    )
    for video_name in video_names:
        inference_fn(
            predictor=predictor,
            base_video_dir=str(args.image_root),
            input_mask_dir=str(args.input_mask_root),
            output_mask_dir=str(args.out_dir),
            video_name=video_name,
            use_all_masks=args.use_all_masks,
            per_obj_png_file=args.per_obj_png_file,
        )

    summary = {
        "status": "pass",
        "sam2_cfg": args.sam2_cfg,
        "checkpoint": str(args.checkpoint),
        "image_root": str(args.image_root),
        "input_mask_root": str(args.input_mask_root),
        "prediction_root": str(args.out_dir),
        "video_names": video_names,
        "device": args.device,
        "per_obj_png_file": args.per_obj_png_file,
        "track_object_appearing_later_in_video": args.track_object_appearing_later_in_video,
        "num_prediction_pngs": count_pngs(args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
