"""Teacher feature attachment helpers for EdgeTAM distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


def extract_teacher_model_state(
    checkpoint: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Extract a full task-model state dict from supported checkpoint schemas."""
    for key in ("model", "task_model_state", "model_state", "state_dict"):
        state = checkpoint.get(key)
        if (
            isinstance(state, dict)
            and state
            and all(isinstance(value, torch.Tensor) for value in state.values())
        ):
            return {
                name.removeprefix("module."): value
                for name, value in state.items()
            }

    if checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return {
            name.removeprefix("module."): value
            for name, value in checkpoint.items()
        }

    raise KeyError(
        "teacher checkpoint does not contain a tensor state dict under "
        "'model', 'task_model_state', 'model_state', or 'state_dict'"
    )


def attach_teacher_features(
    student_outputs: list[dict],
    teacher_outputs: list[dict],
) -> None:
    if len(student_outputs) != len(teacher_outputs):
        raise ValueError(
            f"student/teacher frame count mismatch: {len(student_outputs)} vs {len(teacher_outputs)}"
        )

    for student, teacher in zip(student_outputs, teacher_outputs):
        for student_key, teacher_key in (
            ("teacher_distill_F16", "distill_F16"),
            ("teacher_distill_F_M", "distill_F_M"),
            ("teacher_pred_masks", "pred_masks"),
            ("teacher_obj_ptr", "obj_ptr"),
        ):
            if teacher_key in teacher:
                student[student_key] = teacher[teacher_key].detach()


def flatten_tracking_outputs(
    output_dict: dict[str, dict[int, dict]],
    num_frames: int,
    *,
    keep_obj_ptr: bool,
) -> list[dict]:
    """Flatten SAM2 frame outputs, optionally retaining pointer KD targets."""
    all_frame_outputs = {}
    all_frame_outputs.update(output_dict["cond_frame_outputs"])
    all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
    outputs = [all_frame_outputs[frame_idx] for frame_idx in range(num_frames)]
    if keep_obj_ptr:
        return outputs
    return [
        {key: value for key, value in output.items() if key != "obj_ptr"}
        for output in outputs
    ]


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
