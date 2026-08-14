"""Loss helpers whose weighting semantics are independent of SAM2 internals."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_group_mean(terms: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if terms.shape != valid.shape:
        raise ValueError(
            f"term/valid shape mismatch: {tuple(terms.shape)} != {tuple(valid.shape)}"
        )
    valid = valid.to(dtype=torch.bool)
    if not valid.any():
        return terms.sum() * 0.0
    return terms[valid].mean()


def group_normalized_supervision(
    gt_terms: torch.Tensor,
    gt_valid: torch.Tensor,
    pseudo_terms: torch.Tensor,
    pseudo_valid: torch.Tensor,
    *,
    pseudo_weight: float = 0.25,
) -> torch.Tensor:
    if pseudo_weight < 0:
        raise ValueError("pseudo_weight must be non-negative")
    return masked_group_mean(gt_terms, gt_valid) + pseudo_weight * masked_group_mean(
        pseudo_terms, pseudo_valid
    )


def soft_dice_loss_per_mask(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    epsilon: float = 1.0,
) -> torch.Tensor:
    if probabilities.shape != targets.shape or probabilities.ndim < 3:
        raise ValueError("probabilities and targets must have matching mask shapes")
    probabilities = probabilities.float().flatten(-2)
    targets = targets.float().flatten(-2)
    intersection = (probabilities * targets).sum(-1)
    denominator = probabilities.sum(-1) + targets.sum(-1)
    return 1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)


def soft_mask_distillation_per_mask(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher mask shapes differ")
    teacher_probabilities = teacher_logits.detach().float().sigmoid()
    bce = F.binary_cross_entropy_with_logits(
        student_logits.float(), teacher_probabilities, reduction="none"
    ).flatten(-2).mean(-1)
    dice = soft_dice_loss_per_mask(student_logits.float().sigmoid(), teacher_probabilities)
    return bce + dice


def discounted_refresh_value(
    reuse_cost: torch.Tensor,
    refresh_cost: torch.Tensor,
    *,
    discounts: tuple[float, ...] = (1.0, 0.8, 0.64),
) -> torch.Tensor:
    if reuse_cost.shape != refresh_cost.shape:
        raise ValueError("counterfactual cost shapes differ")
    if reuse_cost.shape[-1] != len(discounts):
        raise ValueError("counterfactual horizon does not match discounts")
    weights = reuse_cost.new_tensor(discounts)
    return ((reuse_cost - refresh_cost) * weights).sum(dim=-1)
