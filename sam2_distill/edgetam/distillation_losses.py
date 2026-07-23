"""Distillation losses for EdgeTAM-style SAM2 training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EdgeTAMDistillationWeights:
    lambda_task: float = 1.0
    lambda_img: float = 1.0
    lambda_mem: float = 1.0
    core_loss_key: str = "core_loss"


def mse_feature_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError(f"feature shape mismatch: student={tuple(student.shape)} teacher={tuple(teacher.shape)}")
    return F.mse_loss(student.float(), teacher.float())


def edgetam_distillation_loss(
    task_losses: dict[str, torch.Tensor],
    student_features: dict[str, torch.Tensor],
    teacher_features: dict[str, torch.Tensor],
    weights: EdgeTAMDistillationWeights = EdgeTAMDistillationWeights(),
) -> dict[str, torch.Tensor]:
    if weights.core_loss_key not in task_losses:
        raise KeyError(f"task loss dict is missing {weights.core_loss_key!r}")

    losses = dict(task_losses)
    total = weights.lambda_task * losses[weights.core_loss_key]

    if weights.lambda_img:
        loss_img = mse_feature_loss(student_features["F16"], teacher_features["F16"])
        losses["loss_img_distill"] = loss_img
        total = total + weights.lambda_img * loss_img

    if weights.lambda_mem:
        loss_mem = mse_feature_loss(student_features["F_M"], teacher_features["F_M"])
        losses["loss_mem_distill"] = loss_mem
        total = total + weights.lambda_mem * loss_mem

    losses[weights.core_loss_key] = total
    return losses


class EdgeTAMMultiStepDistillationLoss(nn.Module):
    """Wrap a SAM2 task loss and add EdgeTAM feature distillation terms."""

    def __init__(
        self,
        task_loss: nn.Module,
        lambda_task: float = 1.0,
        lambda_img: float = 1.0,
        lambda_mem: float = 1.0,
        core_loss_key: str = "core_loss",
    ) -> None:
        super().__init__()
        self.task_loss = task_loss
        self.weights = EdgeTAMDistillationWeights(
            lambda_task=lambda_task,
            lambda_img=lambda_img,
            lambda_mem=lambda_mem,
            core_loss_key=core_loss_key,
        )

    def forward(self, outs_batch: list[dict], targets_batch: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = self.task_loss(outs_batch, targets_batch)
        total = self.weights.lambda_task * losses[self.weights.core_loss_key]

        if self.weights.lambda_img:
            img_terms = self._collect_terms(outs_batch, "distill_F16", "teacher_distill_F16")
            if not img_terms:
                raise KeyError("lambda_img > 0 but outputs do not contain teacher_distill_F16")
            loss_img = torch.stack(img_terms).mean()
            losses["loss_img_distill"] = loss_img
            total = total + self.weights.lambda_img * loss_img

        if self.weights.lambda_mem:
            mem_terms = self._collect_terms(outs_batch, "distill_F_M", "teacher_distill_F_M")
            if not mem_terms:
                raise KeyError("lambda_mem > 0 but outputs do not contain teacher_distill_F_M")
            loss_mem = torch.stack(mem_terms).mean()
            losses["loss_mem_distill"] = loss_mem
            total = total + self.weights.lambda_mem * loss_mem

        losses[self.weights.core_loss_key] = total
        return losses

    @staticmethod
    def _collect_terms(
        outs_batch: list[dict],
        student_key: str,
        teacher_key: str,
    ) -> list[torch.Tensor]:
        terms = []
        for out in outs_batch:
            if student_key not in out or teacher_key not in out:
                continue
            terms.append(mse_feature_loss(out[student_key], out[teacher_key].detach()))
        return terms
