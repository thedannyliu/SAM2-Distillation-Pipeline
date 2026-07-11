"""Load a SAM3.1 Stage 1 TinyViT checkpoint into the official predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from sam2_distill.models.tinyvit_sam3_adapter import TinyViTSAM3Adapter


class SAM31StudentTrunk(nn.Module):
    """Adapt the Stage 1 tensor output to the official ViT list contract."""

    channel_list = [1024]

    def __init__(self, student: TinyViTSAM3Adapter) -> None:
        super().__init__()
        self.student = student

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        return [self.student(images)]


def load_sam31_student(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[SAM31StudentTrunk, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state")
    if not isinstance(state_dict, dict):
        raise KeyError("SAM3.1 Stage 1 checkpoint is missing model_state")
    saved_args = checkpoint.get("args", {})
    model_name = saved_args.get(
        "model_name", "tiny_vit_21m_512.dist_in22k_ft_in1k"
    )
    adapter_mode = saved_args.get("adapter_mode", "residual_dwconv")
    student = TinyViTSAM3Adapter(
        model_name=model_name,
        checkpoint_path=None,
        adapter_mode=adapter_mode,
        freeze_backbone_bn=True,
    )
    student.load_state_dict(state_dict, strict=True)
    student.to(device).eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    return SAM31StudentTrunk(student), {
        "student_checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "model_name": model_name,
        "adapter_mode": adapter_mode,
    }


def patch_multiplex_predictor_trunk(
    predictor,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    trunk, summary = load_sam31_student(checkpoint_path, device)
    vision_backbone = predictor.model.detector.backbone.vision_backbone
    if not hasattr(vision_backbone, "trunk"):
        raise TypeError("Official SAM3.1 detector does not expose vision_backbone.trunk")
    vision_backbone.trunk = trunk
    predictor.model._stage1_student_trunk = trunk
    return summary
