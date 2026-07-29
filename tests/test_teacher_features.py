import pytest
import torch

from sam2_distill.edgetam.teacher_features import (
    attach_teacher_features,
    extract_teacher_model_state,
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
