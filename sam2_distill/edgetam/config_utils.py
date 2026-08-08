"""Small helpers for converting trainer configs to inference configs."""

from __future__ import annotations

from typing import Any


TEACHER_ONLY_MODEL_KEYS = (
    "freeze_teacher",
    "synthetic_teacher",
    "synthetic_teacher_offset",
    "teacher_checkpoint",
    "teacher_feature_cache_path",
    "teacher_model",
    "teacher_model_config",
    "pair_teacher_student_prompts",
)


def strip_teacher_only_model_config(model_config: Any) -> None:
    """Remove teacher-wrapper arguments before inference instantiation."""

    for key in TEACHER_ONLY_MODEL_KEYS:
        if key in model_config:
            del model_config[key]
