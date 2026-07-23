#!/usr/bin/env python
"""Smoke test EdgeTAM SAM2 task loss plus feature distillation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam2-training-root", type=Path, required=True)
    parser.add_argument("--edgetam-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=250107256)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--objects", type=int, default=1)
    parser.add_argument("--masks", type=int, default=3)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--feature-size", type=int, default=8)
    return parser.parse_args()


def add_import_roots(edgetam_root: Path, sam2_training_root: Path) -> None:
    # Keep EdgeTAM first so `sam2.modeling.*` resolves to EdgeTAM modules while
    # `training.*` resolves from the official SAM2 training checkout.
    for root in (sam2_training_root, edgetam_root):
        if not root.exists():
            raise FileNotFoundError(root)
    sys.path.insert(0, str(sam2_training_root))
    sys.path.insert(0, str(edgetam_root))


def make_outputs(args: argparse.Namespace) -> tuple[list[dict], torch.Tensor, list[torch.Tensor]]:
    outputs = []
    grad_tensors = []

    for _ in range(args.frames):
        mask_logits = torch.randn(args.objects, args.masks, args.height, args.width, requires_grad=True)
        pred_ious = torch.randn(args.objects, args.masks, requires_grad=True)
        object_logits = torch.randn(args.objects, 1, requires_grad=True)
        distill_f16 = torch.randn(args.objects, 256, args.feature_size, args.feature_size, requires_grad=True)
        distill_fm = torch.randn(args.objects, 256, args.feature_size, args.feature_size, requires_grad=True)
        teacher_f16 = torch.randn_like(distill_f16)
        teacher_fm = torch.randn_like(distill_fm)

        outputs.append(
            {
                "multistep_pred_multimasks_high_res": [mask_logits],
                "multistep_pred_ious": [pred_ious],
                "multistep_object_score_logits": [object_logits],
                "distill_F16": distill_f16,
                "distill_F_M": distill_fm,
                "teacher_distill_F16": teacher_f16,
                "teacher_distill_F_M": teacher_fm,
            }
        )
        grad_tensors.extend([mask_logits, pred_ious, object_logits, distill_f16, distill_fm])

    targets = torch.zeros(args.frames, args.objects, args.height, args.width)
    targets[:, :, args.height // 4 : args.height // 2, args.width // 4 : args.width // 2] = 1.0
    return outputs, targets, grad_tensors


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(REPO_ROOT))
    add_import_roots(args.edgetam_root, args.sam2_training_root)

    from sam2_distill.edgetam.distillation_losses import EdgeTAMMultiStepDistillationLoss
    from training.loss_fns import MultiStepMultiMasksAndIous
    from sam2_distill.edgetam.teacher_features import (
        attach_synthetic_teacher_features,
        attach_teacher_features,
    )

    task_loss = MultiStepMultiMasksAndIous(
        weight_dict={
            "loss_mask": 20,
            "loss_dice": 1,
            "loss_iou": 1,
            "loss_class": 1,
        },
        supervise_all_iou=True,
        iou_use_l1_loss=True,
        pred_obj_scores=True,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1.0,
    )
    loss_fn = EdgeTAMMultiStepDistillationLoss(
        task_loss=task_loss,
        lambda_img=0.5,
        lambda_mem=0.25,
    )

    outputs, targets, grad_tensors = make_outputs(args)
    losses = loss_fn(outputs, targets)
    losses["core_loss"].backward()

    missing_grad = [idx for idx, tensor in enumerate(grad_tensors) if tensor.grad is None]
    if missing_grad:
        raise RuntimeError(f"missing gradients for tensors: {missing_grad}")

    alignment_loss_fn = EdgeTAMMultiStepDistillationLoss(
        task_loss=task_loss,
        lambda_task=0,
        lambda_img=0,
        lambda_mem=1,
    )
    alignment_outputs, alignment_targets, alignment_tensors = make_outputs(args)
    alignment_losses = alignment_loss_fn(alignment_outputs, alignment_targets)
    alignment_losses["core_loss"].backward()
    memory_tensors = alignment_tensors[4::5]
    if any(tensor.grad is None or not tensor.grad.any() for tensor in memory_tensors):
        raise RuntimeError("pure memory alignment did not backpropagate to F_M")
    non_memory_tensors = [
        tensor
        for index, tensor in enumerate(alignment_tensors)
        if index % 5 != 4
    ]
    if any(tensor.grad is not None and tensor.grad.any() for tensor in non_memory_tensors):
        raise RuntimeError("pure memory alignment leaked nonzero task/image gradients")

    student_outputs = [
        {
            "distill_F16": torch.randn(1, 256, 8, 8, requires_grad=True),
            "distill_F_M": torch.randn(1, 256, 8, 8, requires_grad=True),
        }
        for _ in range(2)
    ]
    teacher_outputs = [
        {
            "distill_F16": torch.randn(1, 256, 8, 8, requires_grad=True),
            "distill_F_M": torch.randn(1, 256, 8, 8, requires_grad=True),
        }
        for _ in range(2)
    ]
    attach_teacher_features(student_outputs, teacher_outputs)
    if any(frame["teacher_distill_F16"].requires_grad for frame in student_outputs):
        raise RuntimeError("attached teacher image features should be detached")
    attach_synthetic_teacher_features(student_outputs, offset=0.125)

    summary = {
        "seed": args.seed,
        "frames": args.frames,
        "objects": args.objects,
        "losses": {key: float(value.detach().cpu()) for key, value in losses.items()},
        "checked_grad_tensors": len(grad_tensors),
        "pure_memory_alignment": "pass",
        "teacher_injection": "pass",
        "result": "pass",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
