"""RepViT feature adapter for SAM2 Stage 1 distillation."""

from __future__ import annotations

from torch import nn

from sam2_distill.models.tinyvit_adapter import TinyViTSAM2Adapter


class RepViTSAM2Adapter(TinyViTSAM2Adapter):
    """Use the shared timm projection interface with a RepViT backbone."""

    def __init__(self, model_name: str, checkpoint_path: str | None, input_size: int = 1024) -> None:
        if not model_name.startswith("repvit_"):
            raise ValueError(f"Expected a RepViT model name, got {model_name}")
        super().__init__(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            input_size=input_size,
            adapter_mode="projection",
        )
        channels = list(self.backbone.body.feature_info.channels())
        final_feature_idx = len(channels) - 1
        self.target_to_feature_idx["image_embed"] = final_feature_idx
        self.projections["image_embed"] = nn.Conv2d(
            channels[final_feature_idx], 256, kernel_size=1
        )
