import torch

from sam2_distill.edgetam.teacher_features import attach_teacher_features


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
