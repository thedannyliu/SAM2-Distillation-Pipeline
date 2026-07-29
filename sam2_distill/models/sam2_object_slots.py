"""Learned fixed-capacity object slots for SAM2 propagation."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


NO_OBJ_SCORE = -1024.0


def _slot_codes(slot_count: int, channels: int) -> torch.Tensor:
    slots = torch.arange(1, slot_count + 1, dtype=torch.float32)[:, None]
    dims = torch.arange(channels, dtype=torch.float32)[None, :] + 0.5
    codes = torch.cos(2 * torch.pi * slots * dims / channels)
    return codes / codes.square().mean(dim=1, keepdim=True).sqrt()


class SharedSlotMemoryAttention(nn.Module):
    """Superpose per-object memories before projecting attention K/V.

    The public interface matches SAM2 ``MemoryAttention``. Below
    ``min_objects`` it executes the original per-object computation. At or
    above the threshold, synchronized object memories are encoded into fixed
    slots and each bucket performs one memory-attention call.
    """

    returns_shared_slots = True

    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,
        slot_count: int = 8,
        min_objects: int = 4,
        memory_dim: int = 64,
    ) -> None:
        super().__init__()
        if slot_count < 1:
            raise ValueError("slot_count must be positive")
        if min_objects < 1:
            raise ValueError("min_objects must be positive")
        self.d_model = d_model
        self.layers = nn.ModuleList(
            copy.deepcopy(layer) for _ in range(num_layers)
        )
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first
        self.slot_count = slot_count
        self.min_objects = min_objects
        self.memory_dim = memory_dim
        self.slot_memory_scale = nn.Parameter(
            _slot_codes(slot_count, memory_dim)
        )

    def _run_attention(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        curr_pos: torch.Tensor | None,
        memory_pos: torch.Tensor | None,
        num_obj_ptr_tokens: int,
        num_spatial_mem: int,
    ) -> torch.Tensor:
        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos
        if self.batch_first:
            output = output.transpose(0, 1)
            memory = memory.transpose(0, 1)
            if curr_pos is not None:
                curr_pos = curr_pos.transpose(0, 1)
            if memory_pos is not None:
                memory_pos = memory_pos.transpose(0, 1)
        for layer in self.layers:
            layer_kwargs = {
                "num_k_exclude_rope": num_obj_ptr_tokens,
            }
            if type(
                getattr(layer, "cross_attn_image", None)
            ).__name__ == "RoPEAttentionv2":
                layer_kwargs["rope_k_repeat"] = num_spatial_mem
            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **layer_kwargs,
            )
        output = self.norm(output)
        return output.transpose(0, 1) if self.batch_first else output

    def _bucket_tensor(
        self,
        value: torch.Tensor,
        *,
        encode_slots: bool,
    ) -> torch.Tensor:
        sequence, object_count, channels = value.shape
        bucket_count = math.ceil(object_count / self.slot_count)
        padded_count = bucket_count * self.slot_count
        if padded_count != object_count:
            value = F.pad(value, (0, 0, 0, padded_count - object_count))
        bucketed = value.view(
            sequence, bucket_count, self.slot_count, channels
        )
        valid = value.new_ones(padded_count)
        valid[object_count:] = 0
        valid = valid.view(1, bucket_count, self.slot_count, 1)
        if encode_slots:
            if channels != self.memory_dim:
                raise ValueError(
                    f"memory channels {channels} != configured {self.memory_dim}"
                )
            scale = self.slot_memory_scale.view(
                1, 1, self.slot_count, channels
            )
            numerator = (bucketed * scale * valid).sum(dim=2)
            denominator = valid.sum(dim=2).clamp_min(1).sqrt()
        else:
            numerator = (bucketed * valid).sum(dim=2)
            denominator = valid.sum(dim=2).clamp_min(1)
        return numerator / denominator

    def forward(
        self,
        curr: torch.Tensor | list[torch.Tensor],
        memory: torch.Tensor,
        curr_pos: torch.Tensor | list[torch.Tensor] | None = None,
        memory_pos: torch.Tensor | None = None,
        num_obj_ptr_tokens: int = 0,
        num_spatial_mem: int = -1,
    ) -> torch.Tensor:
        if isinstance(curr, list):
            if not isinstance(curr_pos, list) or len(curr) != 1 or len(curr_pos) != 1:
                raise ValueError("SAM2 memory attention expects one feature level")
            curr, curr_pos = curr[0], curr_pos[0]
        object_count = curr.shape[1]
        if memory.shape[1] != object_count:
            raise ValueError("current features and memory must share object batch")
        if object_count < self.min_objects:
            return self._run_attention(
                curr,
                memory,
                curr_pos,
                memory_pos,
                num_obj_ptr_tokens,
                num_spatial_mem,
            )

        bucket_count = math.ceil(object_count / self.slot_count)
        curr_bucketed = curr[:, :: self.slot_count, :]
        curr_bucketed = curr_bucketed[:, :bucket_count]
        curr_pos_bucketed = (
            curr_pos[:, :: self.slot_count, :][:, :bucket_count]
            if curr_pos is not None
            else None
        )
        memory_bucketed = self._bucket_tensor(memory, encode_slots=True)
        memory_pos_bucketed = (
            self._bucket_tensor(memory_pos, encode_slots=False)
            if memory_pos is not None
            else None
        )
        output = self._run_attention(
            curr_bucketed,
            memory_bucketed,
            curr_pos_bucketed,
            memory_pos_bucketed,
            num_obj_ptr_tokens,
            num_spatial_mem,
        )
        return output.repeat_interleave(self.slot_count, dim=1)[
            :, :object_count
        ]


class LearnedObjectSlotDecoder(nn.Module):
    """Decode several object masks from one bucket-level spatial feature."""

    def __init__(
        self,
        slot_count: int,
        hidden_dim: int = 256,
        min_objects: int = 4,
    ) -> None:
        super().__init__()
        if slot_count < 1:
            raise ValueError("slot_count must be positive")
        if min_objects < 1:
            raise ValueError("min_objects must be positive")
        self.slot_count = slot_count
        self.hidden_dim = hidden_dim
        self.min_objects = min_objects
        self.slot_spatial_scale = nn.Parameter(
            _slot_codes(slot_count, hidden_dim)
        )
        self.slot_token_embed = nn.Parameter(
            torch.zeros(slot_count, hidden_dim)
        )
        nn.init.trunc_normal_(self.slot_token_embed, std=0.02)

    def _fuse_spatial(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        object_count, channels, height, width = features.shape
        bucket_count = math.ceil(object_count / self.slot_count)
        padded_count = bucket_count * self.slot_count
        if padded_count != object_count:
            features = F.pad(
                features,
                (0, 0, 0, 0, 0, 0, 0, padded_count - object_count),
            )
        bucketed = features.view(
            bucket_count, self.slot_count, channels, height, width
        )
        valid = features.new_ones(padded_count)
        valid[object_count:] = 0
        valid = valid.view(bucket_count, self.slot_count, 1, 1, 1)
        count = valid.sum(dim=1).clamp_min(1)
        mean = (bucketed * valid).sum(dim=1) / count
        centered = (bucketed - mean[:, None]) * valid
        scale = self.slot_spatial_scale.view(
            1, self.slot_count, channels, 1, 1
        )
        fused = mean + (centered * scale).sum(dim=1) / count.sqrt()
        return fused, valid[:, :, 0, 0, 0].bool()

    def _bucket_high_res(
        self, features: list[torch.Tensor] | None
    ) -> list[torch.Tensor] | None:
        if features is None:
            return None
        return [feature[:: self.slot_count] for feature in features]

    def forward(
        self,
        decoder: nn.Module,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        high_res_features: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        object_count = image_embeddings.shape[0]
        shared_image, valid_slots = self._fuse_spatial(image_embeddings)
        bucket_count = shared_image.shape[0]
        slot_embed = self.slot_token_embed
        token_groups = []
        if decoder.pred_obj_scores:
            token_groups.append(decoder.obj_score_token.weight + slot_embed)
        token_groups.append(decoder.iou_token.weight + slot_embed)
        token_groups.append(decoder.mask_tokens.weight[0:1] + slot_embed)
        tokens = torch.cat(token_groups, dim=0).unsqueeze(0)
        tokens = tokens.expand(bucket_count, -1, -1)
        pos_src = image_pe.expand(bucket_count, -1, -1, -1)
        token_output, spatial_output = decoder.transformer(
            shared_image, pos_src, tokens
        )

        cursor = 0
        if decoder.pred_obj_scores:
            object_score_tokens = token_output[
                :, cursor : cursor + self.slot_count
            ]
            cursor += self.slot_count
        iou_tokens = token_output[:, cursor : cursor + self.slot_count]
        cursor += self.slot_count
        mask_tokens = token_output[:, cursor : cursor + self.slot_count]

        batch, channels, height, width = shared_image.shape
        spatial_output = spatial_output.transpose(1, 2).view(
            batch, channels, height, width
        )
        bucket_high_res = self._bucket_high_res(high_res_features)
        if not decoder.use_high_res_features:
            upscaled = decoder.output_upscaling(spatial_output)
        else:
            feat_s0, feat_s1 = bucket_high_res
            dc1, ln1, act1, dc2, act2 = decoder.output_upscaling
            upscaled = act1(ln1(dc1(spatial_output) + feat_s1))
            upscaled = act2(dc2(upscaled) + feat_s0)

        hyper = decoder.output_hypernetworks_mlps[0](mask_tokens)
        _, mask_channels, mask_height, mask_width = upscaled.shape
        masks = torch.bmm(
            hyper, upscaled.view(batch, mask_channels, mask_height * mask_width)
        ).view(batch, self.slot_count, mask_height, mask_width)
        ious = decoder.iou_prediction_head(iou_tokens)[..., 0]
        if decoder.pred_obj_scores:
            object_scores = decoder.pred_obj_score_head(object_score_tokens)
        else:
            object_scores = ious.new_full((*ious.shape, 1), 10.0)

        valid_masks = masks[valid_slots][:object_count, None]
        valid_ious = ious[valid_slots][:object_count, None]
        valid_tokens = mask_tokens[valid_slots][:object_count, None]
        valid_scores = object_scores[valid_slots][:object_count]
        return valid_masks, valid_ious, valid_tokens, valid_scores


class ObjectSlotModelMixin:
    """SAM2 head override used by training and video-predictor classes."""

    object_slot_decoder: LearnedObjectSlotDecoder | None

    def _object_slot_training_anchor(self) -> torch.Tensor | None:
        if not self.training or self.object_slot_decoder is None:
            return None
        terms = [
            parameter.sum() * 0
            for parameter in self.parameters()
            if parameter.requires_grad
        ]
        return sum(terms) if terms else None

    def _init_object_slots(
        self,
        object_slot_count: int,
        object_slot_min_objects: int,
    ) -> None:
        self.object_slot_decoder = (
            LearnedObjectSlotDecoder(
                slot_count=object_slot_count,
                hidden_dim=self.hidden_dim,
                min_objects=object_slot_min_objects,
            )
            if object_slot_count > 0
            else None
        )

    def _forward_sam_heads(
        self,
        backbone_features: torch.Tensor,
        point_inputs: dict[str, torch.Tensor] | None = None,
        mask_inputs: torch.Tensor | None = None,
        high_res_features: list[torch.Tensor] | None = None,
        multimask_output: bool = False,
    ):
        slots = self.object_slot_decoder
        if (
            slots is None
            or (
                not self.training
                and backbone_features.shape[0] < slots.min_objects
            )
            or point_inputs is not None
            or mask_inputs is not None
            or multimask_output
        ):
            outputs = super()._forward_sam_heads(
                backbone_features,
                point_inputs,
                mask_inputs,
                high_res_features,
                multimask_output,
            )
            training_anchor = self._object_slot_training_anchor()
            if training_anchor is not None:
                outputs = tuple(
                    output + training_anchor
                    if isinstance(output, torch.Tensor)
                    else output
                    for output in outputs
                )
            return outputs

        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = slots(
            self.sam_mask_decoder,
            backbone_features,
            self.sam_prompt_encoder.get_dense_pe(),
            high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        low_res_masks = low_res_multimasks
        training_anchor = self._object_slot_training_anchor()
        if training_anchor is not None:
            low_res_masks = low_res_masks + training_anchor
        high_res_masks = high_res_multimasks
        sam_output_token = sam_output_tokens[:, 0]
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            lambda_is_obj_appearing = (
                object_score_logits.sigmoid()
                if self.soft_no_obj_ptr
                else is_obj_appearing.float()
            )
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = (
                obj_ptr
                + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
            )
        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )
