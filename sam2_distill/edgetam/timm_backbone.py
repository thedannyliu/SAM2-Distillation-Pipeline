"""Timm backbone wrapper for EdgeTAM training configs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn


class TimmBackbone(nn.Module):
    """Small timm feature extractor compatible with SAM2 ImageEncoder.

    EdgeTAM's upstream wrapper hardcodes ``pretrained=True``. This repo-owned
    wrapper keeps the same output contract while allowing offline smoke runs
    with random initialization or company runs with an explicit local checkpoint.
    """

    def __init__(
        self,
        name: str,
        features: Tuple[str, ...],
        pretrained: bool = False,
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("sam2_distill.edgetam.timm_backbone requires timm.") from exc

        out_indices = tuple(int(feature.removeprefix("layer")) for feature in features)
        kwargs = {
            "pretrained": pretrained,
            "in_chans": 3,
            "features_only": True,
            "out_indices": out_indices,
        }
        self.body = timm.create_model(name, **kwargs)
        self.checkpoint_load_summary = None
        if checkpoint_path:
            self.checkpoint_load_summary = self._load_tinyvit_checkpoint(checkpoint_path)
        self.channel_list = list(self.body.feature_info.channels())[::-1]

    def _load_tinyvit_checkpoint(self, checkpoint_path: str) -> dict[str, object]:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError("Loading .safetensors TinyViT checkpoints requires safetensors.") from exc

            checkpoint = load_file(str(path), device="cpu")
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model"):
                nested = checkpoint.get(key)
                if isinstance(nested, dict):
                    checkpoint = nested
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported checkpoint format: {path}")

        target_state = self.body.state_dict()
        filtered = {}
        for raw_key, value in checkpoint.items():
            if not torch.is_tensor(value):
                continue
            key = str(raw_key)
            for prefix in ("module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            candidates = [
                key,
                re.sub(r"^stages\.(\d+)\.", r"stages_\1.", key),
            ]
            for candidate in candidates:
                target = target_state.get(candidate)
                if target is not None and tuple(target.shape) == tuple(value.shape):
                    filtered[candidate] = value
                    break

        if not filtered:
            raise RuntimeError(
                f"No compatible TinyViT checkpoint tensors from {path} matched "
                f"{type(self.body).__name__}."
            )

        missing, unexpected = self.body.load_state_dict(filtered, strict=False)
        feature_keys = set(target_state)
        loaded_feature_keys = set(filtered)
        unloaded_feature_keys = sorted(feature_keys - loaded_feature_keys)
        return {
            "path": str(path),
            "loaded_tensors": len(filtered),
            "target_tensors": len(target_state),
            "missing_tensors": len(missing),
            "unexpected_tensors": len(unexpected),
            "unloaded_feature_tensors": len(unloaded_feature_keys),
            "unloaded_feature_tensor_examples": unloaded_feature_keys[:10],
        }

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return list(self.body(x))
