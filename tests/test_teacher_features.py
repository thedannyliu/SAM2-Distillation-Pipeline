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
