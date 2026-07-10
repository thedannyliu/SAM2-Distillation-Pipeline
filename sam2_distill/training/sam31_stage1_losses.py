"""Losses for SAM3.1 raw vision-trunk feature distillation."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def spatial_relation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    grid_size: int = 18,
) -> torch.Tensor:
    """Match pooled spatial-token cosine affinity matrices."""
    student_tokens = F.adaptive_avg_pool2d(student, grid_size).flatten(2).transpose(1, 2)
    teacher_tokens = F.adaptive_avg_pool2d(teacher, grid_size).flatten(2).transpose(1, 2)
    student_tokens = F.normalize(student_tokens, dim=-1, eps=1e-6)
    teacher_tokens = F.normalize(teacher_tokens, dim=-1, eps=1e-6)
    student_relations = student_tokens @ student_tokens.transpose(1, 2)
    teacher_relations = teacher_tokens @ teacher_tokens.transpose(1, 2)
    return F.mse_loss(student_relations, teacher_relations)


def sam31_feature_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    lambda_mse: float = 1.0,
    lambda_cos: float = 0.25,
    lambda_relation: float = 0.0,
    relation_grid_size: int = 18,
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
    if lambda_relation > 0:
        relation = spatial_relation_loss(
            student_float, teacher_float, grid_size=relation_grid_size
        )
    else:
        relation = student_float.new_zeros(())
    total = lambda_mse * mse + lambda_cos * cosine + lambda_relation * relation
    return total, {
        "loss_stage1_total": total.detach(),
        "loss_feature_mse": mse.detach(),
        "loss_feature_cos": cosine.detach(),
        "loss_spatial_relation": relation.detach(),
        "teacher_feature_mean": teacher_float.mean().detach(),
        "teacher_feature_std": teacher_float.std().detach(),
    }
