"""Losses for SAM3.1 raw vision-trunk feature distillation."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def sam31_feature_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    lambda_mse: float = 1.0,
    lambda_cos: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if student.shape != teacher.shape:
        raise ValueError(
            f"Student/teacher shape mismatch: {tuple(student.shape)} vs {tuple(teacher.shape)}"
        )
    student_float = student.float()
    teacher_float = teacher.float()
    mse = F.mse_loss(student_float, teacher_float)
    cosine = 1.0 - F.cosine_similarity(
        student_float, teacher_float, dim=1, eps=1e-6
    ).mean()
    total = lambda_mse * mse + lambda_cos * cosine
    return total, {
        "loss_stage1_total": total.detach(),
        "loss_feature_mse": mse.detach(),
        "loss_feature_cos": cosine.detach(),
        "teacher_feature_mean": teacher_float.mean().detach(),
        "teacher_feature_std": teacher_float.std().detach(),
    }
