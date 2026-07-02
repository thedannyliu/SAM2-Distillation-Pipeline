"""Teacher feature attachment helpers for EdgeTAM distillation."""

from __future__ import annotations

import torch


def attach_teacher_features(
    student_outputs: list[dict],
    teacher_outputs: list[dict],
) -> None:
    if len(student_outputs) != len(teacher_outputs):
        raise ValueError(
            f"student/teacher frame count mismatch: {len(student_outputs)} vs {len(teacher_outputs)}"
        )

    for student, teacher in zip(student_outputs, teacher_outputs):
        student["teacher_distill_F16"] = teacher["distill_F16"].detach()
        student["teacher_distill_F_M"] = teacher["distill_F_M"].detach()


def attach_synthetic_teacher_features(
    student_outputs: list[dict],
    offset: float = 0.01,
) -> None:
    for student in student_outputs:
        student["teacher_distill_F16"] = student["distill_F16"].detach() + offset
        student["teacher_distill_F_M"] = student["distill_F_M"].detach() + offset
