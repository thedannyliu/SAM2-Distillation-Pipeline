from __future__ import annotations

import torch

from sam2_distill.two_clock.training_loss import TwoClockSAM2Loss


def _output(*, gt_valid: bool, anchor: bool, prompt_kind: int, age: int) -> dict:
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    score = torch.zeros(1, 1, requires_grad=True)
    pointer = torch.tensor([[1.0, 0.0]], requires_grad=True)
    state = torch.zeros(1, 2, 2, 2, requires_grad=True)
    return {
        "two_clock_gt_valid": torch.tensor([gt_valid]),
        "two_clock_is_anchor": anchor,
        "two_clock_prompt_kind": prompt_kind,
        "two_clock_age": torch.tensor([age]),
        "multistep_pred_multimasks_high_res": [logits],
        "multistep_pred_ious": [torch.zeros(1, 1, requires_grad=True)],
        "multistep_object_score_logits": [score],
        "pred_masks": logits,
        "teacher_pred_masks": torch.ones_like(logits),
        "distill_F_M": state,
        "teacher_distill_F_M": torch.ones_like(state),
        "obj_ptr": pointer,
        "teacher_obj_ptr": torch.tensor([[0.0, 1.0]]),
        "object_score_logits": score,
        "teacher_object_score_logits": torch.ones_like(score),
    }


def test_mask_prompt_anchor_has_no_supervision_pair() -> None:
    output = _output(gt_valid=True, anchor=True, prompt_kind=0, age=0)
    losses = TwoClockSAM2Loss()([output], torch.ones(1, 1, 4, 4))
    assert losses["gt_pairs"].item() == 0
    assert losses["pseudo_pairs"].item() == 0
    assert losses["core_loss"].item() == 0


def test_box_anchor_uses_gt_and_not_teacher() -> None:
    output = _output(gt_valid=True, anchor=True, prompt_kind=2, age=0)
    losses = TwoClockSAM2Loss()([output], torch.ones(1, 1, 4, 4))
    assert losses["gt_pairs"].item() == 1
    assert losses["pseudo_pairs"].item() == 0
    assert losses["loss_gt_group"].item() > 0


def test_unannotated_reuse_uses_pseudo_and_privileged_state() -> None:
    output = _output(gt_valid=False, anchor=False, prompt_kind=1, age=2)
    loss_module = TwoClockSAM2Loss(
        state_weight=0.25,
        pointer_weight=0.1,
        score_weight=0.1,
    )
    losses = loss_module([output], torch.zeros(1, 1, 4, 4))
    assert losses["gt_pairs"].item() == 0
    assert losses["pseudo_pairs"].item() == 1
    assert losses["reuse_pairs"].item() == 1
    assert losses["loss_pseudo_group"].item() > 0
    assert losses["loss_state_distill"].item() == 1
    losses["core_loss"].backward()
    assert output["distill_F_M"].grad is not None
