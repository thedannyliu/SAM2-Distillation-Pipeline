"""TinyViT adapter that emits the raw SAM3 vision-trunk feature contract."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from sam2_distill.models.tinyvit_adapter import ResidualDepthwiseAdapter


TINYVIT_FINAL_FEATURE_CHANNELS = {
    "tiny_vit_21m_512.dist_in22k_ft_in1k": 576,
    "tiny_vit_11m_224.dist_in22k_ft_in1k": 448,
    "tiny_vit_5m_224.dist_in22k_ft_in1k": 320,
}


def validate_sam3_tinyvit_channels(model_name: str, actual_channels: int) -> None:
    """Guard the final-stage TinyViT feature used by the SAM3 projection."""
    expected = TINYVIT_FINAL_FEATURE_CHANNELS.get(model_name)
    if expected is None:
        raise ValueError(f"Unsupported TinyViT model for SAM3 Stage 1: {model_name}")
    if actual_channels != expected:
        raise ValueError(
            f"{model_name} final feature has {actual_channels} channels; expected {expected}"
        )


class TinyViTSAM3Adapter(nn.Module):
    """TinyViT final feature plus projection and an optional residual adapter."""

    def __init__(
        self,
        model_name: str = "tiny_vit_21m_512.dist_in22k_ft_in1k",
        checkpoint_path: str | None = None,
        adapter_mode: str = "residual_dwconv",
        freeze_backbone_bn: bool = True,
    ) -> None:
        super().__init__()
        if adapter_mode not in ("projection", "residual_dwconv"):
            raise ValueError("adapter_mode must be projection or residual_dwconv")

        from sam2_distill.edgetam.timm_backbone import TimmBackbone

        self.model_name = model_name
        self.adapter_mode = adapter_mode
        self.freeze_backbone_bn = freeze_backbone_bn
        self.backbone = TimmBackbone(
            name=model_name,
            features=("layer0", "layer1", "layer2", "layer3"),
            pretrained=False,
            checkpoint_path=checkpoint_path,
        )
        in_channels = int(self.backbone.body.feature_info.channels()[-1])
        validate_sam3_tinyvit_channels(model_name, in_channels)
        self.projection = nn.Conv2d(in_channels, 1024, kernel_size=1)
        self.adapter = (
            ResidualDepthwiseAdapter(1024)
            if adapter_mode == "residual_dwconv"
            else nn.Identity()
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_backbone_bn:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature = self.backbone(images)[-1]
        projected = self.projection(feature)
        if self.adapter_mode == "residual_dwconv":
            projected = projected + self.adapter(projected)
        if projected.shape[-2:] != (72, 72):
            projected = F.interpolate(
                projected,
                size=(72, 72),
                mode="bilinear",
                align_corners=False,
            )
        return projected
