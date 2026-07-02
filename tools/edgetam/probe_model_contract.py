#!/usr/bin/env python3
"""Validate the EdgeTAM TinyViT model contract and image encoder shapes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--edgetam-root", required=True, type=Path)
    parser.add_argument("--sam2-training-root", type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--tinyvit-checkpoint", type=Path)
    parser.add_argument("--model-checkpoint", type=Path)
    return parser.parse_args()


def add_import_roots(edgetam_root: Path, sam2_training_root: Path | None) -> None:
    for root in (edgetam_root, sam2_training_root):
        if root is not None and not root.exists():
            raise FileNotFoundError(root)
    sys.path.insert(0, str(REPO_ROOT))
    if sam2_training_root is not None:
        sys.path.insert(0, str(sam2_training_root))
    sys.path.insert(0, str(edgetam_root))


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(tensor).all().item()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "finite": bool(finite),
    }


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    args = parse_args()
    add_import_roots(args.edgetam_root, args.sam2_training_root)

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam2_distill.edgetam.compat import patch_edgetam_perceiver_view

    patch_edgetam_perceiver_view()

    cfg = OmegaConf.load(args.config)
    model_cfg = cfg.model
    if args.tinyvit_checkpoint is not None:
        if not args.tinyvit_checkpoint.exists():
            raise FileNotFoundError(args.tinyvit_checkpoint)
        model_cfg.image_encoder.trunk.checkpoint_path = str(args.tinyvit_checkpoint)
        model_cfg.image_encoder.trunk.pretrained = False

    device = torch.device(args.device)
    previous_default_device = torch.get_default_device()
    if device.type == "cuda":
        torch.set_default_device(device)
    try:
        model = instantiate(model_cfg, _recursive_=True)
    finally:
        torch.set_default_device(previous_default_device)
    model.eval()
    model = model.to(device)

    checkpoint_summary = None
    if args.model_checkpoint is not None:
        if not args.model_checkpoint.exists():
            raise FileNotFoundError(args.model_checkpoint)
        checkpoint = torch.load(args.model_checkpoint, map_location="cpu", weights_only=True)
        state_dict = checkpoint["model"]
        incompatible = model.load_state_dict(state_dict, strict=True)
        checkpoint_summary = {
            "path": str(args.model_checkpoint),
            "num_tensors": len(state_dict),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }

    image_size = int(args.image_size or model.image_size)
    expected_stride16 = image_size // 16
    expected_high_res = [
        [32, image_size // 4, image_size // 4],
        [64, image_size // 8, image_size // 8],
        [256, image_size // 16, image_size // 16],
    ]

    with torch.inference_mode():
        images = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
        backbone_out = model.forward_image(images)

    memory_attention_layers = get_attr(model.memory_attention, "num_layers")
    if memory_attention_layers is None and hasattr(model.memory_attention, "layers"):
        memory_attention_layers = len(model.memory_attention.layers)

    perceiver = model.spatial_perceiver
    static = {
        "num_maskmem": int(model.num_maskmem),
        "max_obj_ptrs_in_encoder": int(model.max_obj_ptrs_in_encoder),
        "use_obj_ptrs_in_encoder": bool(model.use_obj_ptrs_in_encoder),
        "memory_attention_layers": int(memory_attention_layers),
        "mem_dim": int(model.mem_dim),
        "hidden_dim": int(model.hidden_dim),
        "spatial_perceiver_class": type(perceiver).__name__ if perceiver is not None else None,
        "spatial_perceiver_num_latents": int(get_attr(perceiver, "num_latents", -1)),
        "spatial_perceiver_num_latents_2d": int(get_attr(perceiver, "num_latents_2d", -1)),
        "image_size": int(model.image_size),
        "forward_image_size": image_size,
        "trunk_name": str(get_attr(model.image_encoder.trunk, "name", model_cfg.image_encoder.trunk.name)),
        "trunk_checkpoint_path": str(args.tinyvit_checkpoint) if args.tinyvit_checkpoint is not None else None,
    }
    outputs = {
        "vision_features": tensor_summary(backbone_out["vision_features"]),
        "backbone_fpn": [tensor_summary(tensor) for tensor in backbone_out["backbone_fpn"]],
        "vision_pos_enc": [tensor_summary(tensor) for tensor in backbone_out["vision_pos_enc"]],
    }

    failures: list[str] = []
    check(static["num_maskmem"] == 7, "num_maskmem must be 7", failures)
    check(static["max_obj_ptrs_in_encoder"] == 16, "max_obj_ptrs_in_encoder must be 16", failures)
    check(static["use_obj_ptrs_in_encoder"], "object pointers must be enabled", failures)
    check(static["memory_attention_layers"] == 2, "memory attention must have 2 layers", failures)
    check(static["mem_dim"] == 64, "memory encoder output dim must be 64", failures)
    check(static["hidden_dim"] == 256, "model hidden dim must be 256", failures)
    check(static["spatial_perceiver_num_latents"] == 256, "spatial perceiver global latents must be 256", failures)
    check(static["spatial_perceiver_num_latents_2d"] == 256, "spatial perceiver 2D latents must be 256", failures)
    check(outputs["vision_features"]["shape"] == [args.batch_size, 256, expected_stride16, expected_stride16], "vision_features must be stride-16 256-channel features", failures)
    check(len(outputs["backbone_fpn"]) == 3, "backbone_fpn must contain 3 feature levels after scalp", failures)
    for idx, expected_hw in enumerate(expected_high_res):
        shape = outputs["backbone_fpn"][idx]["shape"]
        expected_channels, expected_height, expected_width = expected_hw
        check(
            shape == [args.batch_size, expected_channels, expected_height, expected_width],
            f"backbone_fpn[{idx}] shape must be [B,{expected_channels},{expected_height},{expected_width}]",
            failures,
        )
    check(outputs["vision_features"]["finite"], "vision_features contains NaN/Inf", failures)
    for idx, entry in enumerate(outputs["backbone_fpn"]):
        check(entry["finite"], f"backbone_fpn[{idx}] contains NaN/Inf", failures)

    summary = {
        "result": "pass" if not failures else "fail",
        "config": str(args.config),
        "device": str(device),
        "static_contract": static,
        "outputs": outputs,
        "checkpoint_load": checkpoint_summary,
        "failures": failures,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
