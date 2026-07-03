"""TinyViT feature adapter that emits SAM2-compatible Stage 1 targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FeatureTarget:
    name: str
    stride: int
    channels: int
    size: int


SAM2_STAGE1_TARGETS = (
    FeatureTarget("high_res_s0", stride=4, channels=32, size=256),
    FeatureTarget("high_res_s1", stride=8, channels=64, size=128),
    FeatureTarget("image_embed", stride=16, channels=256, size=64),
)


class TinyViTSAM2Adapter(nn.Module):
    """Wrap a timm TinyViT backbone with 1x1 projection heads for SAM2 features."""

    def __init__(
        self,
        model_name: str = "tiny_vit_21m_512.dist_in22k_ft_in1k",
        checkpoint_path: str | None = None,
        input_size: int = 1024,
    ) -> None:
        super().__init__()
        from sam2_distill.edgetam.timm_backbone import TimmBackbone

        self.backbone = TimmBackbone(
            name=model_name,
            features=("layer0", "layer1", "layer2", "layer3"),
            pretrained=False,
            checkpoint_path=checkpoint_path,
        )
        self.input_size = input_size

        info = self.backbone.body.feature_info
        reductions = list(info.reduction())
        channels = list(info.channels())
        self.target_to_feature_idx = {
            target.name: min(range(len(reductions)), key=lambda i: abs(reductions[i] - target.stride))
            for target in SAM2_STAGE1_TARGETS
        }

        self.projections = nn.ModuleDict()
        for target in SAM2_STAGE1_TARGETS:
            in_channels = channels[self.target_to_feature_idx[target.name]]
            self.projections[target.name] = nn.Conv2d(in_channels, target.channels, kernel_size=1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        outputs: dict[str, torch.Tensor] = {}
        for target in SAM2_STAGE1_TARGETS:
            feature = features[self.target_to_feature_idx[target.name]]
            projected = self.projections[target.name](feature)
            if projected.shape[-2:] != (target.size, target.size):
                projected = F.interpolate(
                    projected,
                    size=(target.size, target.size),
                    mode="bilinear",
                    align_corners=False,
                )
            outputs[target.name] = projected
        return outputs
