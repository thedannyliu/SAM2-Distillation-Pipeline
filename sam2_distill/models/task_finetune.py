"""Model components and checkpoint conversion for progressive SAM2 task tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from sam2_distill.models.stage1_checkpoint import extract_state_dict
from sam2_distill.models.stage1_student import build_stage1_student


class _DirectFeatureNeck(nn.Module):
    def __init__(self, position_encoding: nn.Module, d_model: int = 256) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.d_model = d_model


class Stage1StudentImageEncoder(nn.Module):
    """Present a Stage 1 student as a SAM2 ImageEncoder-compatible module."""

    outputs_preprojected_sam_features = True

    def __init__(
        self,
        position_encoding: nn.Module,
        student_family: str = "tinyvit",
        model_name: str = "tiny_vit_21m_512.dist_in22k_ft_in1k",
        checkpoint_path: str | None = None,
        adapter_mode: str = "projection",
    ) -> None:
        super().__init__()
        self.student = build_stage1_student(
            student_family=student_family,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            adapter_mode=adapter_mode,
        )
        self.neck = _DirectFeatureNeck(position_encoding)

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        features = self.student(images)
        backbone_fpn = [
            features["high_res_s0"],
            features["high_res_s1"],
            features["image_embed"],
        ]
        expected_shapes = ((32, 256, 256), (64, 128, 128), (256, 64, 64))
        actual_shapes = [tuple(feature.shape[1:]) for feature in backbone_fpn]
        if actual_shapes != list(expected_shapes):
            raise ValueError(
                "Stage 1 student emitted incompatible SAM features: "
                f"actual={actual_shapes}, expected={list(expected_shapes)}"
            )
        vision_pos_enc = [
            self.neck.position_encoding(feature).to(feature.dtype)
            for feature in backbone_fpn
        ]
        return {
            "vision_features": backbone_fpn[-1],
            "vision_pos_enc": vision_pos_enc,
            "backbone_fpn": backbone_fpn,
        }


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def initialize_task_model(
    model: nn.Module,
    base_sam2_checkpoint: str,
    stage1_checkpoint: str,
    previous_task_checkpoint: str | None = None,
) -> nn.Module:
    """Strictly initialize a task model from Stage 1 or a previous task stage."""

    if previous_task_checkpoint:
        checkpoint = _load_checkpoint(previous_task_checkpoint)
        state = checkpoint.get("model", checkpoint.get("task_model_state"))
        if not isinstance(state, dict):
            raise KeyError(f"No model state in {previous_task_checkpoint}")
        model.load_state_dict(state, strict=True)
        return model

    base_checkpoint = _load_checkpoint(base_sam2_checkpoint)
    base_state = base_checkpoint.get("model", base_checkpoint)
    if not isinstance(base_state, dict):
        raise TypeError(f"Unsupported SAM2 checkpoint: {base_sam2_checkpoint}")
    stage1_payload = _load_checkpoint(stage1_checkpoint)
    stage1_state = extract_state_dict(stage1_payload)
    target_state = model.state_dict()
    merged: dict[str, torch.Tensor] = {}
    for key, target in target_state.items():
        if key.startswith("image_encoder.student."):
            source_key = key.removeprefix("image_encoder.student.")
            source = stage1_state.get(source_key)
        else:
            source = base_state.get(key)
        if source is None:
            raise KeyError(f"Missing initializer tensor for {key}")
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(
                f"Initializer shape mismatch for {key}: "
                f"{tuple(source.shape)} != {tuple(target.shape)}"
            )
        merged[key] = source
    model.load_state_dict(merged, strict=True)
    return model


def export_task_checkpoint(
    trainer_checkpoint: str | Path,
    output_path: str | Path,
    stage_name: str,
    trainable_mode: str,
    source_stage1_checkpoint: str,
) -> dict[str, Any]:
    checkpoint = _load_checkpoint(trainer_checkpoint)
    task_state = checkpoint.get("model")
    if not isinstance(task_state, dict):
        raise KeyError(f"No model state in {trainer_checkpoint}")
    prefix = "image_encoder.student."
    student_state = {
        key.removeprefix(prefix): value
        for key, value in task_state.items()
        if key.startswith(prefix)
    }
    if not student_state:
        raise KeyError("Trainer checkpoint has no Stage 1 student image encoder")
    payload = {
        "step": checkpoint.get("steps", {}).get("train"),
        "epoch": checkpoint.get("epoch"),
        "model_state": student_state,
        "task_model_state": task_state,
        "args": {
            "student_family": "tinyvit",
            "model_name": "tiny_vit_21m_512.dist_in22k_ft_in1k",
            "adapter_mode": "projection",
            "task_stage": stage_name,
            "trainable_mode": trainable_mode,
            "source_stage1_checkpoint": source_stage1_checkpoint,
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return {
        "output": str(output),
        "stage": stage_name,
        "epoch": payload["epoch"],
        "step": payload["step"],
        "student_tensors": len(student_state),
        "task_tensors": len(task_state),
    }
