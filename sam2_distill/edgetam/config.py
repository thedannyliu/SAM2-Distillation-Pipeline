"""Config helpers for the EdgeTAM TinyViT student."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TinyViTEdgeTAMConfig:
    model_name: str = "tiny_vit_21m_512.dist_in22k_ft_in1k"
    image_size: int = 1024
    features: tuple[str, ...] = ("layer0", "layer1", "layer2", "layer3")
    memory_attention_layers: int = 2
    num_maskmem: int = 7
    max_obj_ptrs_in_encoder: int = 16
    num_global_latents: int = 256
    num_2d_latents: int = 256
    d_model: int = 256
    mem_dim: int = 64


def feature_indices(features: tuple[str, ...]) -> tuple[int, ...]:
    indices = []
    for feature in features:
        if not feature.startswith("layer"):
            raise ValueError(f"feature must look like 'layerN', got {feature!r}")
        indices.append(int(feature.removeprefix("layer")))
    return tuple(indices)


def probe_timm_backbone(
    model_name: str,
    features: tuple[str, ...],
    force_probe: bool = False,
) -> dict[str, Any]:
    if not force_probe and model_name in KNOWN_TIMM_FEATURE_METADATA:
        indices = feature_indices(features)
        meta = KNOWN_TIMM_FEATURE_METADATA[model_name]
        reductions = [meta["reductions"][idx] for idx in indices]
        channels = [meta["channels"][idx] for idx in indices]
        return {
            "model_name": model_name,
            "features": list(features),
            "reductions": reductions,
            "channels": channels,
            "backbone_channel_list": list(reversed(channels)),
            "source": "known_timm_metadata",
        }

    try:
        import timm
    except ImportError as exc:
        raise ImportError("TinyViT EdgeTAM config probing requires timm.") from exc

    model = timm.create_model(
        model_name,
        pretrained=False,
        in_chans=3,
        features_only=True,
        out_indices=feature_indices(features),
    )
    reductions = list(model.feature_info.reduction())
    channels = list(model.feature_info.channels())
    return {
        "model_name": model_name,
        "features": list(features),
        "reductions": reductions,
        "channels": channels,
        "backbone_channel_list": list(reversed(channels)),
        "source": "timm_probe",
    }


def render_edgetam_tinyvit_yaml(
    cfg: TinyViTEdgeTAMConfig,
    backbone_channel_list: list[int],
) -> str:
    features_yaml = "\n".join(f"      - {name}" for name in cfg.features)
    channels_yaml = "[" + ", ".join(str(x) for x in backbone_channel_list) + "]"
    return f"""# @package _global_

# EdgeTAM TinyViT-21M student config generated from timm feature_info.
model:
  _target_: sam2.modeling.sam2_base.SAM2Base
  image_encoder:
    _target_: sam2.modeling.backbones.image_encoder.ImageEncoder
    scalp: 1
    trunk:
      _target_: sam2_distill.edgetam.timm_backbone.TimmBackbone
      name: {cfg.model_name}
      pretrained: false
      features:
{features_yaml}
    neck:
      _target_: sam2.modeling.backbones.image_encoder.FpnNeck
      position_encoding:
        _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
        num_pos_feats: {cfg.d_model}
        normalize: true
        scale: null
        temperature: 10000
      d_model: {cfg.d_model}
      backbone_channel_list: {channels_yaml}
      fpn_top_down_levels: [2, 3]
      fpn_interp_model: nearest

  memory_attention:
    _target_: sam2.modeling.memory_attention.MemoryAttention
    d_model: {cfg.d_model}
    pos_enc_at_input: true
    layer:
      _target_: sam2.modeling.memory_attention.MemoryAttentionLayer
      activation: relu
      dim_feedforward: 2048
      dropout: 0.1
      pos_enc_at_attn: false
      self_attention:
        _target_: sam2.modeling.sam.transformer.RoPEAttention
        rope_theta: 10000.0
        feat_sizes: [32, 32]
        embedding_dim: {cfg.d_model}
        num_heads: 1
        downsample_rate: 1
        dropout: 0.1
      d_model: {cfg.d_model}
      pos_enc_at_cross_attn_keys: true
      pos_enc_at_cross_attn_queries: false
      cross_attention:
        _target_: sam2.modeling.sam.transformer.RoPEAttentionv2
        rope_theta: 10000.0
        q_sizes: [64, 64]
        k_sizes: [16, 16]
        embedding_dim: {cfg.d_model}
        num_heads: 1
        downsample_rate: 1
        dropout: 0.1
        kv_in_dim: {cfg.mem_dim}
    num_layers: {cfg.memory_attention_layers}

  memory_encoder:
    _target_: sam2.modeling.memory_encoder.MemoryEncoder
    out_dim: {cfg.mem_dim}
    position_encoding:
      _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
      num_pos_feats: {cfg.mem_dim}
      normalize: true
      scale: null
      temperature: 10000
    mask_downsampler:
      _target_: sam2.modeling.memory_encoder.MaskDownSampler
      kernel_size: 3
      stride: 2
      padding: 1
    fuser:
      _target_: sam2.modeling.memory_encoder.Fuser
      layer:
        _target_: sam2.modeling.memory_encoder.CXBlock
        dim: {cfg.d_model}
        kernel_size: 7
        padding: 3
        layer_scale_init_value: 1e-6
        use_dwconv: true
      num_layers: 2

  spatial_perceiver:
    _target_: sam2.modeling.perceiver.PerceiverResampler
    depth: 2
    dim: {cfg.mem_dim}
    dim_head: {cfg.mem_dim}
    heads: 1
    ff_mult: 4
    hidden_dropout_p: 0.0
    attention_dropout_p: 0.0
    pos_enc_at_key_value: true
    concat_kv_latents: false
    num_latents: {cfg.num_global_latents}
    num_latents_2d: {cfg.num_2d_latents}
    position_encoding:
      _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
      num_pos_feats: {cfg.mem_dim}
      normalize: true
      scale: null
      temperature: 10000
    use_self_attn: true

  num_maskmem: {cfg.num_maskmem}
  image_size: {cfg.image_size}
  sigmoid_scale_for_mem_enc: 20.0
  sigmoid_bias_for_mem_enc: -10.0
  use_mask_input_as_output_without_sam: true
  directly_add_no_mem_embed: true
  use_high_res_features_in_sam: true
  multimask_output_in_sam: true
  iou_prediction_use_sigmoid: true
  use_obj_ptrs_in_encoder: true
  add_tpos_enc_to_obj_ptrs: false
  only_obj_ptrs_in_the_past_for_eval: true
  max_obj_ptrs_in_encoder: {cfg.max_obj_ptrs_in_encoder}
  pred_obj_scores: true
  pred_obj_scores_mlp: true
  fixed_no_obj_ptr: true
  multimask_output_for_tracking: true
  use_multimask_token_for_obj_ptr: true
  multimask_min_pt_num: 0
  multimask_max_pt_num: 1
  use_mlp_for_obj_ptr_proj: true
  compile_image_encoder: false
"""


def write_edgetam_tinyvit_yaml(
    out: Path,
    cfg: TinyViTEdgeTAMConfig = TinyViTEdgeTAMConfig(),
    force_probe: bool = False,
) -> dict[str, Any]:
    probe = probe_timm_backbone(cfg.model_name, cfg.features, force_probe=force_probe)
    text = render_edgetam_tinyvit_yaml(
        cfg,
        backbone_channel_list=probe["backbone_channel_list"],
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return probe
KNOWN_TIMM_FEATURE_METADATA = {
    "tiny_vit_21m_512.dist_in22k_ft_in1k": {
        "reductions": [4, 8, 16, 32],
        "channels": [96, 192, 384, 576],
    },
    "tiny_vit_21m_512.in1k": {
        "reductions": [4, 8, 16, 32],
        "channels": [96, 192, 384, 576],
    },
}
