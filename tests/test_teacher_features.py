import pytest
import torch
from torch import nn

from sam2_distill.edgetam.distillation_losses import (
    EdgeTAMMultiStepDistillationLoss,
)
from sam2_distill.edgetam.teacher_features import (
    attach_teacher_features,
    extract_teacher_model_state,
    flatten_tracking_outputs,
)


def test_extract_teacher_model_state_prefers_full_exported_task_state():
    encoder_weight = torch.ones(1)
    task_weight = torch.ones(2)
    checkpoint = {
        "step": 10,
        "model_state": {"encoder.weight": encoder_weight},
        "task_model_state": {"module.task.weight": task_weight},
        "args": {},
    }

    state = extract_teacher_model_state(checkpoint)

    assert set(state) == {"task.weight"}
    assert state["task.weight"] is task_weight


def test_extract_teacher_model_state_accepts_raw_state_dict():
    weight = torch.ones(1)

    state = extract_teacher_model_state({"module.weight": weight})

    assert state == {"weight": weight}


def test_extract_teacher_model_state_rejects_metadata_only_checkpoint():
    with pytest.raises(KeyError, match="task_model_state"):
        extract_teacher_model_state({"step": 10, "args": {}})


def test_attach_teacher_features_skips_absent_optional_targets():
    student = {}
    teacher = {
        "distill_F16": torch.ones(1),
        "distill_F_M": torch.ones(1),
        "pred_masks": torch.ones(1),
    }

    attach_teacher_features([student], [teacher])

    assert set(student) == {
        "teacher_distill_F16",
        "teacher_distill_F_M",
        "teacher_pred_masks",
    }
    assert "teacher_obj_ptr" not in student


def test_attach_teacher_features_detaches_present_targets():
    student = {}
    teacher = {
        "distill_F16": torch.ones(1, requires_grad=True),
        "distill_F_M": torch.ones(1, requires_grad=True),
        "pred_masks": torch.ones(1, requires_grad=True),
        "obj_ptr": torch.ones(1, requires_grad=True),
    }

    attach_teacher_features([student], [teacher])

    assert all(not value.requires_grad for value in student.values())


def test_flatten_tracking_outputs_retains_obj_ptr_for_kd():
    outputs = {
        "cond_frame_outputs": {
            0: {"pred_masks": torch.ones(1), "obj_ptr": torch.ones(2)}
        },
        "non_cond_frame_outputs": {
            1: {"pred_masks": torch.ones(1), "obj_ptr": torch.ones(2)}
        },
    }

    with_ptr = flatten_tracking_outputs(
        outputs,
        num_frames=2,
        keep_obj_ptr=True,
    )
    without_ptr = flatten_tracking_outputs(
        outputs,
        num_frames=2,
        keep_obj_ptr=False,
    )

    assert all("obj_ptr" in output for output in with_ptr)
    assert all("obj_ptr" not in output for output in without_ptr)


def test_obj_ptr_distillation_backpropagates_through_student_pointer():
    class TaskLoss(nn.Module):
        def forward(self, outputs, targets):
            del targets
            return {"core_loss": outputs[0]["obj_ptr"].sum() * 0}

    student_ptr = torch.tensor([[1.0, 0.0]], requires_grad=True)
    loss = EdgeTAMMultiStepDistillationLoss(
        task_loss=TaskLoss(),
        lambda_img=0,
        lambda_mem=0,
        lambda_obj_ptr=0.25,
    )

    losses = loss(
        [
            {
                "obj_ptr": student_ptr,
                "teacher_obj_ptr": torch.tensor([[0.0, 1.0]]),
            }
        ],
        targets_batch=torch.empty(0),
    )
    losses["core_loss"].backward()

    assert losses["loss_obj_ptr_distill"].item() == pytest.approx(1.0)
    assert student_ptr.grad is not None
    assert torch.count_nonzero(student_ptr.grad).item() > 0


def test_task_loss_normalization_is_invariant_to_repeated_frames():
    class SummedTaskLoss(nn.Module):
        def forward(self, outputs, targets):
            del targets
            return {
                "core_loss": torch.stack(
                    [output["task"] for output in outputs]
                ).sum()
            }

    loss = EdgeTAMMultiStepDistillationLoss(
        task_loss=SummedTaskLoss(),
        lambda_img=0,
        lambda_mem=0,
        normalize_task_by_num_frames=True,
    )

    t4 = loss(
        [{"task": torch.tensor(2.0)} for _ in range(4)],
        targets_batch=torch.empty(0),
    )
    t8 = loss(
        [{"task": torch.tensor(2.0)} for _ in range(8)],
        targets_batch=torch.empty(0),
    )

    assert t4["core_loss"].item() == pytest.approx(2.0)
    assert t8["core_loss"].item() == pytest.approx(2.0)
    assert t4["loss_task_raw"].item() == pytest.approx(8.0)
    assert t8["loss_task_raw"].item() == pytest.approx(16.0)


def test_temporal_kd_excludes_initial_conditioning_frames():
    class ZeroTaskLoss(nn.Module):
        def forward(self, outputs, targets):
            del targets
            return {"core_loss": outputs[0]["distill_F_M"].sum() * 0}

    loss = EdgeTAMMultiStepDistillationLoss(
        task_loss=ZeroTaskLoss(),
        lambda_img=0,
        lambda_mem=1,
        temporal_kd_on_propagated_frames_only=True,
    )
    outputs = [
        {
            "distill_is_init_cond_frame": True,
            "distill_F_M": torch.tensor([100.0]),
            "teacher_distill_F_M": torch.tensor([0.0]),
        },
        {
            "distill_is_init_cond_frame": False,
            "distill_F_M": torch.tensor([2.0]),
            "teacher_distill_F_M": torch.tensor([0.0]),
        },
    ]

    losses = loss(outputs, targets_batch=torch.empty(0))

    assert losses["loss_mem_distill"].item() == pytest.approx(4.0)
    assert losses["core_loss"].item() == pytest.approx(4.0)


def test_paired_prompt_match_rate_is_reported():
    class ZeroTaskLoss(nn.Module):
        def forward(self, outputs, targets):
            del targets
            return {"core_loss": outputs[0]["task"].sum() * 0}

    loss = EdgeTAMMultiStepDistillationLoss(
        task_loss=ZeroTaskLoss(),
        lambda_img=0,
        lambda_mem=0,
    )
    losses = loss(
        [
            {"task": torch.ones(1), "teacher_prompt_match": True},
            {"task": torch.ones(1), "teacher_prompt_match": True},
        ],
        targets_batch=torch.empty(0),
    )

    assert losses["prompt_match_rate"].item() == pytest.approx(1.0)
