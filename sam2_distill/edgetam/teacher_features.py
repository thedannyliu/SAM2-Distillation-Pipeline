"""Teacher feature attachment helpers for EdgeTAM distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


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
        student["teacher_pred_masks"] = teacher["pred_masks"].detach()
        student["teacher_obj_ptr"] = teacher["obj_ptr"].detach()


def attach_synthetic_teacher_features(
    student_outputs: list[dict],
    offset: float = 0.01,
) -> None:
    for student in student_outputs:
        student["teacher_distill_F16"] = student["distill_F16"].detach() + offset
        student["teacher_distill_F_M"] = student["distill_F_M"].detach() + offset
        student["teacher_pred_masks"] = student["pred_masks"].detach() + offset
        student["teacher_obj_ptr"] = student["obj_ptr"].detach() + offset


class TeacherFeatureCache:
    """Frame-ordered EdgeTAM teacher feature cache.

    The expected cache file is a torch checkpoint with:

    ``teacher_distill_F16`` or ``F16``:
        Tensor/list with frame-major features, each frame shaped ``[C, H, W]``
        or ``[N, C, H, W]``.

    ``teacher_distill_F_M`` or ``F_M``:
        Same contract for memory-attended features.
    """

    def __init__(self, path: str | Path, map_location: str = "cpu") -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        payload = torch.load(self.path, map_location=map_location, weights_only=True)
        self.f16 = self._read_feature(payload, "teacher_distill_F16", "F16")
        self.fm = self._read_feature(payload, "teacher_distill_F_M", "F_M")
        if len(self.f16) != len(self.fm):
            raise ValueError(f"teacher cache length mismatch: F16={len(self.f16)} F_M={len(self.fm)}")

    def attach(self, student_outputs: list[dict]) -> None:
        if len(student_outputs) > len(self.f16):
            raise ValueError(
                f"teacher cache has {len(self.f16)} frames but outputs require {len(student_outputs)}"
            )
        for frame_idx, student in enumerate(student_outputs):
            student["teacher_distill_F16"] = self._match_student_shape(
                self.f16[frame_idx],
                student["distill_F16"],
            )
            student["teacher_distill_F_M"] = self._match_student_shape(
                self.fm[frame_idx],
                student["distill_F_M"],
            )

    @staticmethod
    def _read_feature(payload: dict[str, Any], primary: str, fallback: str) -> list[torch.Tensor]:
        if primary in payload:
            value = payload[primary]
        elif fallback in payload:
            value = payload[fallback]
        else:
            raise KeyError(f"teacher cache missing {primary!r} or {fallback!r}")

        if isinstance(value, torch.Tensor):
            if value.dim() < 4:
                raise ValueError(f"teacher cache tensor for {primary} must be frame-major, got {tuple(value.shape)}")
            return [frame.detach().cpu() for frame in value]
        return [torch.as_tensor(frame).detach().cpu() for frame in value]

    @staticmethod
    def _match_student_shape(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
        teacher = teacher.to(device=student.device, dtype=student.dtype)
        if teacher.dim() == 3:
            teacher = teacher.unsqueeze(0)
        if teacher.dim() != 4:
            raise ValueError(f"teacher feature must be [C,H,W] or [N,C,H,W], got {tuple(teacher.shape)}")

        if teacher.shape[-2:] != student.shape[-2:]:
            teacher = F.interpolate(
                teacher,
                size=student.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if teacher.shape[1] != student.shape[1]:
            raise ValueError(
                f"teacher/student channel mismatch: teacher={teacher.shape[1]} student={student.shape[1]}"
            )
        if teacher.shape[0] == 1 and student.shape[0] != 1:
            teacher = teacher.expand(student.shape[0], -1, -1, -1)
        if teacher.shape != student.shape:
            raise ValueError(f"teacher/student shape mismatch: teacher={tuple(teacher.shape)} student={tuple(student.shape)}")
        return teacher.detach()
