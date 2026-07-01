"""Stage 1 feature-distillation losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def cosine_feature_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student_flat = student.flatten(2)
    teacher_flat = teacher.flatten(2)
    return 1.0 - F.cosine_similarity(student_flat, teacher_flat, dim=1).mean()


def stage1_feature_distillation_loss(
    student: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    lambda_mse: float = 1.0,
    lambda_l1: float = 0.5,
    lambda_cos: float = 0.1,
    lambda_hr: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    required = ("image_embed", "high_res_s0", "high_res_s1")
    missing = [name for name in required if name not in student or name not in teacher]
    if missing:
        raise KeyError(f"missing feature targets: {missing}")

    image_mse = F.mse_loss(student["image_embed"], teacher["image_embed"])
    image_l1 = F.smooth_l1_loss(student["image_embed"], teacher["image_embed"])
    image_cos = cosine_feature_loss(student["image_embed"], teacher["image_embed"])
    high_res_mse = F.mse_loss(student["high_res_s0"], teacher["high_res_s0"]) + F.mse_loss(
        student["high_res_s1"], teacher["high_res_s1"]
    )

    total = (
        lambda_mse * image_mse
        + lambda_l1 * image_l1
        + lambda_cos * image_cos
        + lambda_hr * high_res_mse
    )
    metrics = {
        "loss_stage1_total": total.detach(),
        "loss_image_mse": image_mse.detach(),
        "loss_image_l1": image_l1.detach(),
        "loss_image_cos": image_cos.detach(),
        "loss_high_res_mse": high_res_mse.detach(),
    }
    return total, metrics
