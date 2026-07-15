#!/usr/bin/env python3
"""Smoke-test a pretrained RepViT Stage 1 student on one synthetic image."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sam2_distill.models.stage1_student import build_stage1_student
from sam2_distill.training.stage1_losses import stage1_feature_distillation_loss


EXPECTED_SHAPES = {
    "high_res_s0": (1, 32, 256, 256),
    "high_res_s1": (1, 64, 128, 128),
    "image_embed": (1, 256, 64, 64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing RepViT checkpoint: {args.checkpoint}")
    device = torch.device(args.device)
    model = build_stage1_student(
        student_family="repvit",
        model_name=args.model_name,
        checkpoint_path=str(args.checkpoint),
        adapter_mode="projection",
    ).to(device)
    model.train()
    images = torch.randn(1, 3, 1024, 1024, device=device)
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with amp:
        student = model(images)
        teacher = {
            name: torch.randn(shape, device=device, dtype=student[name].dtype)
            for name, shape in EXPECTED_SHAPES.items()
        }
        loss, metrics = stage1_feature_distillation_loss(
            student,
            teacher,
            lambda_mse=1.0,
            lambda_hr=1.0,
            lambda_cos=0.25,
            lambda_l1=0.10,
        )
    loss.backward()

    actual_shapes = {name: tuple(value.shape) for name, value in student.items()}
    if actual_shapes != EXPECTED_SHAPES:
        raise SystemExit(f"Unexpected Stage 1 shapes: {actual_shapes}")
    missing_grad = [
        name for name, param in model.named_parameters() if param.requires_grad and param.grad is None
    ]
    if missing_grad:
        raise SystemExit(f"Trainable parameters without gradients: {missing_grad[:20]}")
    load_summary = model.backbone.checkpoint_load_summary
    if not load_summary or int(load_summary["loaded_tensors"]) == 0:
        raise SystemExit("RepViT pretrained checkpoint did not load any tensors")

    print(
        json.dumps(
            {
                "status": "pass",
                "model_name": args.model_name,
                "checkpoint": str(args.checkpoint),
                "parameters": sum(param.numel() for param in model.parameters()),
                "shapes": {name: list(shape) for name, shape in actual_shapes.items()},
                "losses": {name: float(value) for name, value in metrics.items()},
                "checkpoint_load": load_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
