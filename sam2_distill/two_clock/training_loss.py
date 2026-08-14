"""Group-normalized GT/pseudo and privileged-state losses for two-clock SAM2."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from sam2_distill.two_clock.losses import (
    group_normalized_supervision,
    masked_group_mean,
    soft_mask_distillation_per_mask,
)


class TwoClockSAM2Loss(nn.Module):
    def __init__(
        self,
        *,
        pseudo_weight: float = 0.25,
        focal_weight: float = 20.0,
        dice_weight: float = 1.0,
        iou_weight: float = 1.0,
        object_weight: float = 1.0,
        state_weight: float = 0.0,
        pointer_weight: float = 0.0,
        score_weight: float = 0.0,
        core_loss_key: str = "core_loss",
    ) -> None:
        super().__init__()
        self.pseudo_weight = pseudo_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.object_weight = object_weight
        self.state_weight = state_weight
        self.pointer_weight = pointer_weight
        self.score_weight = score_weight
        self.core_loss_key = core_loss_key

    def forward(self, outputs: list[dict], targets: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(outputs) != len(targets):
            raise ValueError("output/target frame count differs")
        gt_terms, gt_validity = [], []
        pseudo_terms, pseudo_validity = [], []
        state_terms, pointer_terms, score_terms, reuse_validity = [], [], [], []
        components = defaultdict(list)
        for frame_output, target in zip(outputs, targets):
            gt_valid = frame_output["two_clock_gt_valid"].bool()
            exclude_anchor_mask = bool(frame_output["two_clock_is_anchor"]) and int(
                frame_output["two_clock_prompt_kind"]
            ) == 0
            if exclude_anchor_mask:
                gt_valid = torch.zeros_like(gt_valid)
            gt, frame_components = self._gt_terms(frame_output, target)
            teacher_masks = frame_output["teacher_pred_masks"]
            pseudo = soft_mask_distillation_per_mask(
                frame_output["pred_masks"], teacher_masks
            ).squeeze(1)
            pseudo_valid = ~frame_output["two_clock_gt_valid"].bool()
            gt_terms.append(gt)
            gt_validity.append(gt_valid)
            pseudo_terms.append(pseudo)
            pseudo_validity.append(pseudo_valid)
            for name, value in frame_components.items():
                components[name].append(value)

            reuse = frame_output["two_clock_age"].gt(0)
            reuse_validity.append(reuse)
            state_terms.append(
                (frame_output["distill_F_M"].float() - frame_output["teacher_distill_F_M"].float())
                .square()
                .flatten(1)
                .mean(1)
            )
            pointer_terms.append(
                1.0
                - F.cosine_similarity(
                    frame_output["obj_ptr"].float(),
                    frame_output["teacher_obj_ptr"].float(),
                    dim=-1,
                )
            )
            score_terms.append(
                F.binary_cross_entropy_with_logits(
                    frame_output["object_score_logits"].float(),
                    frame_output["teacher_object_score_logits"].float().sigmoid(),
                    reduction="none",
                ).flatten(1).mean(1)
            )

        gt_terms_tensor = torch.stack(gt_terms)
        gt_valid_tensor = torch.stack(gt_validity)
        pseudo_terms_tensor = torch.stack(pseudo_terms)
        pseudo_valid_tensor = torch.stack(pseudo_validity)
        loss_gt = masked_group_mean(gt_terms_tensor, gt_valid_tensor)
        loss_pseudo = masked_group_mean(pseudo_terms_tensor, pseudo_valid_tensor)
        total = group_normalized_supervision(
            gt_terms_tensor,
            gt_valid_tensor,
            pseudo_terms_tensor,
            pseudo_valid_tensor,
            pseudo_weight=self.pseudo_weight,
        )
        reuse_valid = torch.stack(reuse_validity)
        loss_state = masked_group_mean(torch.stack(state_terms), reuse_valid)
        loss_pointer = masked_group_mean(torch.stack(pointer_terms), reuse_valid)
        loss_score = masked_group_mean(torch.stack(score_terms), reuse_valid)
        total = (
            total
            + self.state_weight * loss_state
            + self.pointer_weight * loss_pointer
            + self.score_weight * loss_score
        )
        result = {
            self.core_loss_key: total,
            "loss_gt_group": loss_gt,
            "loss_pseudo_group": loss_pseudo,
            "loss_state_distill": loss_state,
            "loss_pointer_distill": loss_pointer,
            "loss_score_distill": loss_score,
            "gt_pairs": gt_valid_tensor.sum().to(total.dtype),
            "pseudo_pairs": pseudo_valid_tensor.sum().to(total.dtype),
            "reuse_pairs": reuse_valid.sum().to(total.dtype),
        }
        for name, values in components.items():
            result[name] = masked_group_mean(torch.stack(values), gt_valid_tensor)
        return result

    def _gt_terms(
        self, output: dict, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = output["multistep_pred_multimasks_high_res"][-1].float()
        target = target[:, None].float().expand_as(logits)
        probabilities = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = probabilities * target + (1 - probabilities) * (1 - target)
        alpha_t = 0.25 * target + 0.75 * (1 - target)
        focal = (ce * (1 - p_t).square() * alpha_t).flatten(2).mean(-1)
        intersection = (probabilities * target).flatten(2).sum(-1)
        denominator = probabilities.flatten(2).sum(-1) + target.flatten(2).sum(-1)
        dice = 1.0 - (2 * intersection + 1) / (denominator + 1)
        best = (self.focal_weight * focal + self.dice_weight * dice).argmin(-1)
        batch = torch.arange(logits.shape[0], device=logits.device)
        focal = focal[batch, best]
        dice = dice[batch, best]
        selected_logits = logits[batch, best]
        predicted_mask = selected_logits.flatten(1) > 0
        target_mask = target[batch, best].flatten(1) > 0
        intersection_hard = (predicted_mask & target_mask).sum(-1).float()
        union = (predicted_mask | target_mask).sum(-1).float().clamp_min(1)
        actual_iou = intersection_hard / union
        predicted_iou = output["multistep_pred_ious"][-1].float()[batch, best]
        iou = F.l1_loss(predicted_iou, actual_iou, reduction="none")
        object_target = target_mask.any(-1).float()
        object_loss = F.binary_cross_entropy_with_logits(
            output["multistep_object_score_logits"][-1].float().flatten(),
            object_target,
            reduction="none",
        )
        total = (
            self.focal_weight * focal
            + self.dice_weight * dice
            + self.iou_weight * iou
            + self.object_weight * object_loss
        )
        return total, {
            "loss_gt_focal": focal,
            "loss_gt_dice": dice,
            "loss_gt_iou": iou,
            "loss_gt_object": object_loss,
        }
