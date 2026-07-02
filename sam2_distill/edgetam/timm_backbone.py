"""Timm backbone wrapper for EdgeTAM training configs."""

from __future__ import annotations

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
        if checkpoint_path:
            path = Path(checkpoint_path)
            if not path.exists():
                raise FileNotFoundError(path)
            kwargs["checkpoint_path"] = str(path)
        self.body = timm.create_model(name, **kwargs)
        self.channel_list = list(self.body.feature_info.channels())[::-1]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return list(self.body(x))
