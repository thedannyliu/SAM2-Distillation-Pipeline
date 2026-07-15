"""Build SAM2-compatible Stage 1 students across supported backbone families."""

from __future__ import annotations

from torch import nn


def build_stage1_student(
    student_family: str,
    model_name: str,
    checkpoint_path: str | None,
    adapter_mode: str,
) -> nn.Module:
    if student_family == "tinyvit":
        from sam2_distill.models.tinyvit_adapter import TinyViTSAM2Adapter

        return TinyViTSAM2Adapter(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            adapter_mode=adapter_mode,
        )
    if student_family == "repvit":
        if adapter_mode != "projection":
            raise ValueError("RepViT Stage 1 currently supports projection mode only")
        from sam2_distill.models.repvit_adapter import RepViTSAM2Adapter

        return RepViTSAM2Adapter(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
        )
    raise ValueError(f"Unsupported Stage 1 student family: {student_family}")
