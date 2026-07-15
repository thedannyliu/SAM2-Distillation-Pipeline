"""Helpers for loading Stage 1 student checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


MODEL_BY_IMAGE_PROJ_CHANNELS = {
    384: "tiny_vit_21m_512.dist_in22k_ft_in1k",
    256: "tiny_vit_11m_224.dist_in22k_ft_in1k",
    160: "tiny_vit_5m_224.dist_in22k_ft_in1k",
}

CKPT_BY_MODEL = {
    "tiny_vit_21m_512.dist_in22k_ft_in1k": "tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors",
    "tiny_vit_11m_224.dist_in22k_ft_in1k": "tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors",
    "tiny_vit_5m_224.dist_in22k_ft_in1k": "tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors",
    "repvit_m0_9.dist_450e_in1k": "repvit_m0_9.dist_450e_in1k.safetensors",
    "repvit_m2_3.dist_450e_in1k": "repvit_m2_3.dist_450e_in1k.safetensors",
}


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model", "model_state", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return strip_module_prefix(value)
    raise KeyError("checkpoint must contain one of: model, model_state, state_dict")


def infer_tinyvit_model_name(state_dict: dict[str, torch.Tensor], fallback: str) -> str:
    weight = state_dict.get("projections.image_embed.weight")
    if torch.is_tensor(weight) and weight.ndim == 4:
        model_name = MODEL_BY_IMAGE_PROJ_CHANNELS.get(int(weight.shape[1]))
        if model_name is not None:
            return model_name
    return fallback


def infer_stage1_model_name(
    checkpoint: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    fallback: str,
) -> str:
    args = checkpoint.get("args")
    if isinstance(args, dict) and isinstance(args.get("model_name"), str):
        return str(args["model_name"])
    return infer_tinyvit_model_name(state_dict, fallback)


def infer_student_family(
    checkpoint: dict[str, Any], model_name: str, fallback: str = "tinyvit"
) -> str:
    args = checkpoint.get("args")
    if isinstance(args, dict) and args.get("student_family") in {"tinyvit", "repvit"}:
        return str(args["student_family"])
    if model_name.startswith("repvit_"):
        return "repvit"
    return fallback


def infer_adapter_mode(checkpoint: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> str:
    args = checkpoint.get("args")
    if isinstance(args, dict) and args.get("adapter_mode") in {"projection", "residual_dwconv"}:
        return str(args["adapter_mode"])
    if any(key.startswith("adapters.") for key in state_dict):
        return "residual_dwconv"
    return "projection"


def resolve_tinyvit_checkpoint(model_name: str, requested_checkpoint: Path) -> Path:
    expected_name = CKPT_BY_MODEL.get(model_name)
    if expected_name is None:
        return requested_checkpoint
    candidate = requested_checkpoint.parent / expected_name
    if candidate.exists():
        return candidate
    return requested_checkpoint


def resolve_student_checkpoint(model_name: str, requested_checkpoint: Path) -> Path:
    return resolve_tinyvit_checkpoint(model_name, requested_checkpoint)


def load_task_non_image_state(
    model: torch.nn.Module, checkpoint: dict[str, Any]
) -> dict[str, Any] | None:
    """Load decoder/memory weights from a progressive task-tuning checkpoint."""

    task_state = checkpoint.get("task_model_state")
    if not isinstance(task_state, dict):
        return None
    non_image_state = {
        key: value
        for key, value in strip_module_prefix(task_state).items()
        if not key.startswith("image_encoder.")
    }
    target_non_image = {
        key for key in model.state_dict() if not key.startswith("image_encoder.")
    }
    missing_source = sorted(target_non_image - set(non_image_state))
    unexpected_source = sorted(set(non_image_state) - target_non_image)
    if missing_source or unexpected_source:
        raise RuntimeError(
            "Task checkpoint non-image state does not match SAM2: "
            f"missing={missing_source[:10]}, unexpected={unexpected_source[:10]}"
        )
    incompatible = model.load_state_dict(non_image_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_non_image = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("image_encoder.")
    ]
    if unexpected or missing_non_image:
        raise RuntimeError(
            "Failed to load task checkpoint non-image state: "
            f"missing={missing_non_image[:10]}, unexpected={unexpected[:10]}"
        )
    return {
        "task_stage": checkpoint.get("args", {}).get("task_stage"),
        "trainable_mode": checkpoint.get("args", {}).get("trainable_mode"),
        "non_image_tensors_loaded": len(non_image_state),
    }
