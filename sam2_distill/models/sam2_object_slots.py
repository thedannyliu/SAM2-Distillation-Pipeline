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


class _LowRankMemoryProjection(nn.Module):
    def __init__(self, memory_dim: int, rank: int, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(memory_dim)
        self.down = nn.Linear(memory_dim, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.zeros_(self.up.weight)

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.down(self.norm(value)))

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(value)


class LowRankObjectMemoryResidual(nn.Module):
    """Restore cheap object-specific state after bucket-shared attention.

    Spatial memory tokens are aligned with the current feature grid and
    averaged over memory frames. Object-pointer tokens are pooled separately.
    Each enabled path uses a low-rank projection into the SAM hidden
    dimension, avoiding another per-object memory-attention stack.
    """

    def __init__(
        self,
        d_model: int,
        memory_dim: int,
        spatial_rank: int,
        pointer_rank: int,
        temporal_pool: str = "mean",
        temporal_decay: float = 0.5,
    ) -> None:
        super().__init__()
        if spatial_rank < 0 or pointer_rank < 0:
            raise ValueError("Object residual ranks must be non-negative")
        if spatial_rank == 0 and pointer_rank == 0:
            raise ValueError("At least one object residual path is required")
        if temporal_pool not in {"mean", "latest", "recency"}:
            raise ValueError(
                "temporal_pool must be mean, latest, or recency"
            )
        if not 0 < temporal_decay <= 1:
            raise ValueError("temporal_decay must be in (0, 1]")
        self.d_model = d_model
        self.memory_dim = memory_dim
        self.spatial_rank = spatial_rank
        self.pointer_rank = pointer_rank
        self.temporal_pool = temporal_pool
        self.temporal_decay = temporal_decay
        self.spatial_path = self._make_path(spatial_rank)
        self.pointer_path = self._make_path(pointer_rank)

    def _make_path(
        self, rank: int
    ) -> _LowRankMemoryProjection | None:
        if rank == 0:
            return None
        return _LowRankMemoryProjection(
            memory_dim=self.memory_dim,
            rank=rank,
            d_model=self.d_model,
        )

    @staticmethod
    def _align_spatial_memory(
        spatial_memory: torch.Tensor,
        query_tokens: int,
        temporal_pool: str,
        temporal_decay: float,
    ) -> torch.Tensor:
        spatial_tokens, object_count, channels = spatial_memory.shape
        if spatial_tokens % query_tokens == 0:
            frames = spatial_memory.reshape(
                -1, query_tokens, object_count, channels
            )
            if temporal_pool == "latest":
                return frames[-1]
            if temporal_pool == "recency":
                powers = torch.arange(
                    frames.shape[0] - 1,
                    -1,
                    -1,
                    device=frames.device,
                    dtype=frames.dtype,
                )
                weights = temporal_decay**powers
                weights = weights / weights.sum()
                return (frames * weights[:, None, None, None]).sum(dim=0)
            return frames.mean(dim=0)
        return spatial_memory.mean(dim=0, keepdim=True).expand(
            query_tokens, -1, -1
        )

    def forward(
        self,
        memory: torch.Tensor,
        *,
        memory_pos: torch.Tensor | None,
        query_tokens: int,
        num_obj_ptr_tokens: int,
    ) -> torch.Tensor:
        if num_obj_ptr_tokens < 0 or num_obj_ptr_tokens > memory.shape[0]:
            raise ValueError("Invalid object-pointer token count")
        if memory_pos is not None and memory_pos.shape != memory.shape:
            raise ValueError("Memory and positional encoding shapes differ")
        positioned_memory = (
            memory if memory_pos is None else memory + memory_pos
        )
        object_count = memory.shape[1]
        residual = memory.new_zeros(
            query_tokens, object_count, self.d_model
        )
        spatial_end = memory.shape[0] - num_obj_ptr_tokens
        if self.spatial_path is not None and spatial_end > 0:
            spatial_memory = positioned_memory[:spatial_end]
            if (
                self.temporal_pool == "latest"
                and spatial_end % query_tokens == 0
            ):
                spatial_memory = spatial_memory[-query_tokens:]
            spatial = self.spatial_path.encode(
                spatial_memory
            )
            spatial = self._align_spatial_memory(
                spatial,
                query_tokens,
                self.temporal_pool,
                self.temporal_decay,
            )
            residual = residual + self.spatial_path.decode(spatial)
        if self.pointer_path is not None and num_obj_ptr_tokens > 0:
            pointers = self.pointer_path.encode(
                positioned_memory[spatial_end:]
            ).mean(dim=0)
            residual = residual + self.pointer_path.decode(pointers)[None]
        return residual


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
        object_residual_rank: int = 0,
        object_pointer_residual_rank: int = 0,
        object_residual_temporal_pool: str = "mean",
        object_residual_temporal_decay: float = 0.5,
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
        self.object_residual = (
            LowRankObjectMemoryResidual(
                d_model=d_model,
                memory_dim=memory_dim,
                spatial_rank=object_residual_rank,
                pointer_rank=object_pointer_residual_rank,
                temporal_pool=object_residual_temporal_pool,
                temporal_decay=object_residual_temporal_decay,
            )
            if object_residual_rank > 0
            or object_pointer_residual_rank > 0
            else None
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
        output = output.repeat_interleave(self.slot_count, dim=1)[
            :, :object_count
        ]
        if self.object_residual is not None:
            output = output + self.object_residual(
                memory,
                memory_pos=memory_pos,
                query_tokens=curr.shape[0],
                num_obj_ptr_tokens=num_obj_ptr_tokens,
            )
        return output


def _multiplex_mask_downsampler(
    mask_downsampler: nn.Module,
    slot_count: int,
) -> nn.Module:
    """Clone a SAM2 mask downsampler with one input channel per slot."""

    multiplex = copy.deepcopy(mask_downsampler)
    encoder = getattr(multiplex, "encoder", None)
    if not isinstance(encoder, nn.Sequential) or not encoder:
        raise TypeError("SAM2 mask downsampler must expose encoder[0]")
    first = encoder[0]
    if not isinstance(first, nn.Conv2d) or first.in_channels != 1:
        raise TypeError(
            "SAM2 mask downsampler must start with a 1-channel Conv2d"
        )
    replacement = nn.Conv2d(
        in_channels=slot_count,
        out_channels=first.out_channels,
        kernel_size=first.kernel_size,
        stride=first.stride,
        padding=first.padding,
        dilation=first.dilation,
        groups=first.groups,
        bias=first.bias is not None,
        padding_mode=first.padding_mode,
    ).to(device=first.weight.device, dtype=first.weight.dtype)
    with torch.no_grad():
        replacement.weight.copy_(first.weight.repeat(1, slot_count, 1, 1))
        if first.bias is not None:
            replacement.bias.copy_(first.bias)
    encoder[0] = replacement
    return multiplex


class SlotPreservingMemoryEncoder(nn.Module):
    """Encode an object bucket as a multi-channel mask memory.

    This follows SAM 3.1's central multiplex invariant: each object keeps a
    stable mask channel while the bucket produces one dense memory feature.
    The bucket feature is repeated only to preserve SAM2's public tensor
    interface; ``SlotPreservingMemoryAttention`` removes that repetition
    before attention.
    """

    def __init__(
        self,
        out_dim: int,
        mask_downsampler: nn.Module,
        fuser: nn.Module,
        position_encoding: nn.Module,
        in_dim: int = 256,
        slot_count: int = 8,
        min_objects: int = 4,
    ) -> None:
        super().__init__()
        if slot_count < 1 or min_objects < 1:
            raise ValueError("slot_count and min_objects must be positive")
        self.slot_count = slot_count
        self.min_objects = min_objects
        self.mask_downsampler = mask_downsampler
        self.multiplex_mask_downsampler = _multiplex_mask_downsampler(
            mask_downsampler, slot_count
        )
        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = (
            nn.Identity()
            if out_dim == in_dim
            else nn.Conv2d(in_dim, out_dim, kernel_size=1)
        )

    def _encode(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        mask_downsampler: nn.Module,
        skip_mask_sigmoid: bool,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if not skip_mask_sigmoid:
            masks = masks.sigmoid()
        masks = mask_downsampler(masks)
        pix_feat = pix_feat.to(masks.device)
        features = self.pix_feat_proj(pix_feat) + masks
        features = self.out_proj(self.fuser(features))
        position = self.position_encoding(features).to(features.dtype)
        return {
            "vision_features": features,
            "vision_pos_enc": [position],
        }

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        object_count = masks.shape[0]
        if pix_feat.shape[0] != object_count or masks.shape[1] != 1:
            raise ValueError("SAM2 multiplex memory expects [N,1,H,W] masks")
        if not self.training and object_count < self.min_objects:
            return self._encode(
                pix_feat,
                masks,
                self.mask_downsampler,
                skip_mask_sigmoid,
            )

        if not skip_mask_sigmoid:
            masks = masks.sigmoid()
            skip_mask_sigmoid = True

        bucket_count = math.ceil(object_count / self.slot_count)
        padded_count = bucket_count * self.slot_count
        if padded_count != object_count:
            padding = masks.new_zeros(
                padded_count - object_count,
                *masks.shape[1:],
            )
            masks = torch.cat([masks, padding], dim=0)
        multiplex_masks = masks.view(
            bucket_count,
            self.slot_count,
            masks.shape[-2],
            masks.shape[-1],
        )
        bucket_pix_feat = pix_feat[:: self.slot_count][:bucket_count]
        encoded = self._encode(
            bucket_pix_feat,
            multiplex_masks,
            self.multiplex_mask_downsampler,
            skip_mask_sigmoid,
        )
        encoded["vision_features"] = encoded[
            "vision_features"
        ].repeat_interleave(self.slot_count, dim=0)[:object_count]
        encoded["vision_pos_enc"] = [
            value.repeat_interleave(self.slot_count, dim=0)[:object_count]
            for value in encoded["vision_pos_enc"]
        ]
        return encoded


class SlotPreservingMemoryAttention(SharedSlotMemoryAttention):
    """Attend once per bucket without superposing object identities.

    Dense memory is already shared and slot-preserving after the multi-channel
    memory encoder. Object pointers remain private tokens and are packed along
    the memory sequence with learned slot positional embeddings.
    """

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
        super().__init__(
            d_model=d_model,
            pos_enc_at_input=pos_enc_at_input,
            layer=layer,
            num_layers=num_layers,
            batch_first=batch_first,
            slot_count=slot_count,
            min_objects=min_objects,
            memory_dim=memory_dim,
        )
        del self.slot_memory_scale
        self.slot_pointer_pos = nn.Parameter(
            0.02 * _slot_codes(slot_count, memory_dim)
        )

    def _pack_pointer_tokens(self, value: torch.Tensor) -> torch.Tensor:
        sequence, object_count, channels = value.shape
        bucket_count = math.ceil(object_count / self.slot_count)
        padded_count = bucket_count * self.slot_count
        if padded_count != object_count:
            value = F.pad(value, (0, 0, 0, padded_count - object_count))
        return (
            value.view(sequence, bucket_count, self.slot_count, channels)
            .permute(0, 2, 1, 3)
            .reshape(sequence * self.slot_count, bucket_count, channels)
        )

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
            if (
                not isinstance(curr_pos, list)
                or len(curr) != 1
                or len(curr_pos) != 1
            ):
                raise ValueError(
                    "SAM2 memory attention expects one feature level"
                )
            curr, curr_pos = curr[0], curr_pos[0]
        object_count = curr.shape[1]
        if memory.shape[1] != object_count:
            raise ValueError("current features and memory must share object batch")
        if not self.training and object_count < self.min_objects:
            return self._run_attention(
                curr,
                memory,
                curr_pos,
                memory_pos,
                num_obj_ptr_tokens,
                num_spatial_mem,
            )
        if not 0 <= num_obj_ptr_tokens <= memory.shape[0]:
            raise ValueError("Invalid object-pointer token count")

        bucket_count = math.ceil(object_count / self.slot_count)
        spatial_end = memory.shape[0] - num_obj_ptr_tokens
        bucket_memory = self._bucket_tensor(
            memory[:spatial_end], encode_slots=False
        )
        bucket_memory_pos = (
            self._bucket_tensor(
                memory_pos[:spatial_end], encode_slots=False
            )
            if memory_pos is not None
            else None
        )
        packed_pointer_count = 0
        if num_obj_ptr_tokens:
            pointers = self._pack_pointer_tokens(memory[spatial_end:])
            packed_pointer_count = pointers.shape[0]
            bucket_memory = torch.cat([bucket_memory, pointers], dim=0)
            if memory_pos is not None:
                pointer_pos = self._pack_pointer_tokens(
                    memory_pos[spatial_end:]
                )
                slot_pos = self.slot_pointer_pos.view(
                    1, self.slot_count, 1, self.memory_dim
                ).expand(
                    num_obj_ptr_tokens,
                    self.slot_count,
                    bucket_count,
                    self.memory_dim,
                )
                slot_pos = slot_pos.reshape(
                    packed_pointer_count, bucket_count, self.memory_dim
                )
                pointer_pos = pointer_pos + slot_pos
                bucket_memory_pos = torch.cat(
                    [bucket_memory_pos, pointer_pos], dim=0
                )

        bucket_curr = curr[:, :: self.slot_count, :][:, :bucket_count]
        bucket_curr_pos = (
            curr_pos[:, :: self.slot_count, :][:, :bucket_count]
            if curr_pos is not None
            else None
        )
        output = self._run_attention(
            bucket_curr,
            bucket_memory,
            bucket_curr_pos,
            bucket_memory_pos,
            packed_pointer_count,
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
