#!/usr/bin/env python3
"""One-batch SAM3.1 teacher and TinyViT Stage 1 compatibility smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sam2_distill.models.sam31_teacher import SAM31VisionTeacher
from sam2_distill.models.tinyvit_sam3_adapter import TinyViTSAM3Adapter
from sam2_distill.training.sam31_stage1_losses import sam31_feature_distillation_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--tinyvit-checkpoint", required=True)
    parser.add_argument("--model-name", default="tiny_vit_21m_512.dist_in22k_ft_in1k")
    parser.add_argument("--adapter-mode", default="residual_dwconv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the default smoke test")
    device = torch.device(args.device)
    teacher = SAM31VisionTeacher(args.teacher_checkpoint).to(device).eval()
    student = TinyViTSAM3Adapter(
        model_name=args.model_name,
        checkpoint_path=args.tinyvit_checkpoint,
        adapter_mode=args.adapter_mode,
        freeze_backbone_bn=True,
    ).to(device).train()
    image = torch.zeros(1, 3, 1008, 1008, device=device)

    amp_enabled = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=amp_enabled,
    ):
        target = teacher(image)
    target = target.detach().clone()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=amp_enabled,
    ):
        prediction = student(image)
        loss, metrics = sam31_feature_distillation_loss(prediction, target)
    loss.backward()

    loaded = student.backbone.checkpoint_load_summary or {}
    summary = {
        "status": "pass",
        "torch_version": torch.__version__,
        "device": str(device),
        "teacher_checkpoint_prefix": teacher.checkpoint_prefix,
        "teacher_shape": list(target.shape),
        "student_shape": list(prediction.shape),
        "loss": float(loss.detach().cpu()),
        "loss_feature_mse": float(metrics["loss_feature_mse"].cpu()),
        "loss_feature_cos": float(metrics["loss_feature_cos"].cpu()),
        "tinyvit_loaded_tensors": loaded.get("loaded_tensors"),
        "tinyvit_target_tensors": loaded.get("target_tensors"),
        "student_trainable_parameters": sum(
            parameter.numel() for parameter in student.parameters() if parameter.requires_grad
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
