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
    lambda_mask_logits: float = 0.0
    lambda_obj_ptr: float = 0.0
    core_loss_key: str = "core_loss"


def mse_feature_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError(
            "feature shape mismatch: "
            f"student={tuple(student.shape)} teacher={tuple(teacher.shape)}"
        )
    return F.mse_loss(student.float(), teacher.float())


def mask_logit_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError(
            "mask-logit shape mismatch: "
            f"student={tuple(student.shape)} teacher={tuple(teacher.shape)}"
        )
    return F.binary_cross_entropy_with_logits(
        student.float(),
        teacher.detach().float().sigmoid(),
    )


def cosine_feature_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError(
            "cosine-feature shape mismatch: "
            f"student={tuple(student.shape)} teacher={tuple(teacher.shape)}"
        )
    return (
        1.0
        - F.cosine_similarity(
            student.float(),
            teacher.detach().float(),
            dim=-1,
        )
    ).mean()


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
        lambda_mask_logits: float = 0.0,
        lambda_obj_ptr: float = 0.0,
        normalize_task_by_num_frames: bool = False,
        temporal_kd_on_propagated_frames_only: bool = False,
        core_loss_key: str = "core_loss",
    ) -> None:
        super().__init__()
        self.task_loss = task_loss
        self.weights = EdgeTAMDistillationWeights(
            lambda_task=lambda_task,
            lambda_img=lambda_img,
            lambda_mem=lambda_mem,
            lambda_mask_logits=lambda_mask_logits,
            lambda_obj_ptr=lambda_obj_ptr,
            core_loss_key=core_loss_key,
        )
        self.normalize_task_by_num_frames = normalize_task_by_num_frames
        self.temporal_kd_on_propagated_frames_only = (
            temporal_kd_on_propagated_frames_only
        )

    def forward(
        self,
        outs_batch: list[dict],
        targets_batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        losses = self.task_loss(outs_batch, targets_batch)
        task_core = losses[self.weights.core_loss_key]
        if self.normalize_task_by_num_frames:
            if not outs_batch:
                raise ValueError(
                    "cannot normalize task loss for an empty output batch"
                )
            losses["loss_task_raw"] = task_core.detach()
            task_core = task_core / len(outs_batch)
            losses["loss_task_normalized"] = task_core
        total = self.weights.lambda_task * task_core

        if self.weights.lambda_img:
            img_terms = self._collect_terms(
                outs_batch,
                "distill_F16",
                "teacher_distill_F16",
            )
            if not img_terms:
                raise KeyError(
                    "lambda_img > 0 but outputs do not contain "
                    "teacher_distill_F16"
                )
            loss_img = torch.stack(img_terms).mean()
            losses["loss_img_distill"] = loss_img
            total = total + self.weights.lambda_img * loss_img

        if self.weights.lambda_mem:
            mem_terms = self._collect_terms(
                outs_batch,
                "distill_F_M",
                "teacher_distill_F_M",
                propagated_only=self.temporal_kd_on_propagated_frames_only,
            )
            if not mem_terms:
                raise KeyError(
                    "lambda_mem > 0 but outputs do not contain "
                    "teacher_distill_F_M"
                )
            loss_mem = torch.stack(mem_terms).mean()
            losses["loss_mem_distill"] = loss_mem
            total = total + self.weights.lambda_mem * loss_mem

        if self.weights.lambda_mask_logits:
            logit_terms = self._collect_terms(
                outs_batch,
                "pred_masks",
                "teacher_pred_masks",
                loss_fn=mask_logit_distillation_loss,
                propagated_only=self.temporal_kd_on_propagated_frames_only,
            )
            if not logit_terms:
                raise KeyError(
                    "lambda_mask_logits > 0 but outputs do not contain "
                    "teacher_pred_masks"
                )
            loss_logits = torch.stack(logit_terms).mean()
            losses["loss_mask_logit_distill"] = loss_logits
            total = total + self.weights.lambda_mask_logits * loss_logits

        if self.weights.lambda_obj_ptr:
            obj_ptr_terms = self._collect_terms(
                outs_batch,
                "obj_ptr",
                "teacher_obj_ptr",
                loss_fn=cosine_feature_loss,
                propagated_only=self.temporal_kd_on_propagated_frames_only,
            )
            if not obj_ptr_terms:
                raise KeyError(
                    "lambda_obj_ptr > 0 but outputs do not contain "
                    "teacher_obj_ptr"
                )
            loss_obj_ptr = torch.stack(obj_ptr_terms).mean()
            losses["loss_obj_ptr_distill"] = loss_obj_ptr
            total = total + self.weights.lambda_obj_ptr * loss_obj_ptr

        losses[self.weights.core_loss_key] = total
        prompt_matches = [
            float(out["teacher_prompt_match"])
            for out in outs_batch
            if "teacher_prompt_match" in out
        ]
        if prompt_matches:
            losses["prompt_match_rate"] = total.new_tensor(prompt_matches).mean()
        return losses

    @staticmethod
    def _collect_terms(
        outs_batch: list[dict],
        student_key: str,
        teacher_key: str,
        loss_fn=mse_feature_loss,
        propagated_only: bool = False,
    ) -> list[torch.Tensor]:
        terms = []
        for out in outs_batch:
            if propagated_only:
                if "distill_is_init_cond_frame" not in out:
                    raise KeyError(
                        "propagated-only KD requires "
                        "distill_is_init_cond_frame in every output"
                    )
                if out["distill_is_init_cond_frame"]:
                    continue
            if student_key not in out or teacher_key not in out:
                continue
            terms.append(loss_fn(out[student_key], out[teacher_key].detach()))
        return terms
