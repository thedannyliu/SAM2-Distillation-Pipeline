"""SAM3.1 vision-trunk teacher loading for Stage 1 distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import nn


def unwrap_checkpoint_state(checkpoint: object) -> Mapping[str, torch.Tensor]:
    """Return the tensor state dictionary from a common PyTorch checkpoint."""
    state = checkpoint
    if isinstance(state, Mapping):
        for key in ("model", "model_state", "state_dict"):
            nested = state.get(key)
            if isinstance(nested, Mapping):
                state = nested
                break
    if not isinstance(state, Mapping):
        raise TypeError("SAM3.1 checkpoint does not contain a state dictionary")
    return {str(key): value for key, value in state.items() if torch.is_tensor(value)}


def extract_vision_trunk_state(
    checkpoint_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], str]:
    """Find and strictly extract one complete SAM3 ViT trunk from a checkpoint."""
    anchor = "patch_embed.proj.weight"
    if anchor not in target_state:
        raise KeyError(f"SAM3 vision trunk is missing expected key {anchor!r}")

    prefixes = {
        key[: -len(anchor)]
        for key, value in checkpoint_state.items()
        if key.endswith(anchor) and tuple(value.shape) == tuple(target_state[anchor].shape)
    }
    if not prefixes:
        examples = sorted(key for key in checkpoint_state if "patch_embed" in key)[:10]
        raise RuntimeError(
            "Could not find a SAM3 vision trunk in the checkpoint. "
            f"patch_embed examples: {examples}"
        )

    candidates: list[tuple[int, int, str, dict[str, torch.Tensor]]] = []
    for prefix in prefixes:
        extracted: dict[str, torch.Tensor] = {}
        for key, target in target_state.items():
            value = checkpoint_state.get(prefix + key)
            if value is not None and tuple(value.shape) == tuple(target.shape):
                extracted[key] = value
        detector_priority = int("detector." in prefix)
        candidates.append((len(extracted), detector_priority, prefix, extracted))

    matched, _, prefix, extracted = max(candidates, key=lambda item: (item[0], item[1]))
    missing = sorted(set(target_state) - set(extracted))
    if missing:
        raise RuntimeError(
            f"SAM3.1 vision trunk prefix {prefix!r} matched {matched}/{len(target_state)} "
            f"tensors; first missing keys: {missing[:10]}"
        )
    return extracted, prefix


class SAM31VisionTeacher(nn.Module):
    """Frozen official SAM3.1 ViT trunk returning ``[B, 1024, 72, 72]``."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        super().__init__()
        try:
            from sam3.model_builder import _create_vit_backbone
        except ImportError as exc:
            raise ImportError(
                "SAM3 source is required. Add /user-volume/repo/facebookresearch-sam3 "
                "to PYTHONPATH."
            ) from exc

        self.trunk = _create_vit_backbone(
            compile_mode=None,
            use_fa3=False,
            use_rope_real=False,
        )
        checkpoint = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=True
        )
        checkpoint_state = unwrap_checkpoint_state(checkpoint)
        trunk_state, self.checkpoint_prefix = extract_vision_trunk_state(
            checkpoint_state, self.trunk.state_dict()
        )
        self.trunk.load_state_dict(trunk_state, strict=True)
        del checkpoint, checkpoint_state, trunk_state

        self.trunk.eval()
        for parameter in self.trunk.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        self.trunk.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.trunk(images)
        if len(features) != 1 or tuple(features[0].shape[1:]) != (1024, 72, 72):
            shapes = [tuple(feature.shape) for feature in features]
            raise RuntimeError(f"Unexpected SAM3.1 trunk output shapes: {shapes}")
        return features[0]
