"""SAM2.1-L training model with real image-encoder reuse.

This module is imported only in the company SAM2 container, where the official
``sam2`` and ``training`` packages are on ``PYTHONPATH``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames

from sam2_distill.edgetam.train_model import EdgeTAMTrainWithTeacher
from sam2_distill.two_clock.experiments import get_experiment
from sam2_distill.two_clock.temporal import (
    FeatureAgeConditioner,
    TwoClockMemoryConditioner,
)


METADATA_WIDTH = 7
META_RAW_FRAME = 2
META_GT_VALID = 3
META_REFRESH = 4
META_SOURCE_RAW = 5
META_AGE = 6


def initialize_two_clock_sam21l(model, checkpoint_path: str):
    """Load every official tensor strictly while retaining new zero-init modules."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"unsupported SAM2 checkpoint: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_prefixes = ("age_conditioner.", "memory_time_conditioner.")
    invalid_missing = [key for key in missing if not key.startswith(allowed_prefixes)]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "official initializer mismatch: "
            f"missing={invalid_missing[:20]} unexpected={unexpected[:20]}"
        )
    return model


class TwoClockSAM2Train(EdgeTAMTrainWithTeacher):
    """Official SAM2.1-L student with cached raw features and two-clock reads."""

    def __init__(
        self,
        *args,
        experiment: str,
        max_feature_age: int = 5,
        box_jitter_fraction: float = 0.05,
        **kwargs,
    ) -> None:
        if kwargs.get("trainable_module_mode") is not None:
            raise ValueError("TwoClockSAM2Train owns its trainable-module policy")
        kwargs["trainable_module_mode"] = None
        kwargs["freeze_batchnorm"] = True
        kwargs["expose_obj_ptr_for_distillation"] = True
        super().__init__(*args, **kwargs)
        self.experiment = get_experiment(experiment)
        self.max_feature_age = max_feature_age
        self.box_jitter_fraction = box_jitter_fraction
        self.age_conditioner = (
            FeatureAgeConditioner(max_age=max_feature_age)
            if self.experiment.age_conditioning
            else None
        )
        self.memory_time_conditioner = (
            TwoClockMemoryConditioner(
                max_freshness=max_feature_age,
                use_freshness=self.experiment.freshness_conditioning,
            )
            if self.experiment.recency_conditioning
            else None
        )
        for parameter in self.sam_prompt_encoder.parameters():
            parameter.requires_grad = False
        self._freeze_batchnorm_modules()
        self._frozen_eval_modules = [self.sam_prompt_encoder]
        self._two_clock_context = None
        self.encoder_calls_last_forward = 0

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        backbone_out = super().prepare_prompt_inputs(
            backbone_out, input, start_frame_idx=start_frame_idx
        )
        prompt_kind = 0
        point_input = backbone_out["point_inputs_per_frame"].get(start_frame_idx)
        if point_input is not None:
            labels = point_input["point_labels"]
            is_box = labels.shape[-1] == 2 and torch.all(labels[..., 0] == 2) and torch.all(
                labels[..., 1] == 3
            )
            prompt_kind = 2 if bool(is_box) else 1
            if prompt_kind == 2 and self.box_jitter_fraction > 0:
                point_input["point_coords"] = self._jitter_boxes(
                    point_input["point_coords"], input.img_batch.shape[-2:]
                )
        backbone_out["two_clock_prompt_kind"] = prompt_kind
        return backbone_out

    def _jitter_boxes(
        self, boxes: torch.Tensor, image_hw: tuple[int, int]
    ) -> torch.Tensor:
        boxes = boxes.clone()
        height, width = image_hw
        lower = torch.minimum(boxes[:, 0], boxes[:, 1])
        upper = torch.maximum(boxes[:, 0], boxes[:, 1])
        size = (upper - lower).clamp_min(1.0)
        noise = torch.as_tensor(
            self.rng.uniform(-self.box_jitter_fraction, self.box_jitter_fraction, boxes.shape),
            device=boxes.device,
            dtype=boxes.dtype,
        )
        jittered = boxes + noise * size[:, None, :]
        x = jittered[..., 0].clamp(0, width - 1)
        y = jittered[..., 1].clamp(0, height - 1)
        first = torch.stack(
            (torch.minimum(x[:, 0], x[:, 1]), torch.minimum(y[:, 0], y[:, 1])),
            dim=-1,
        )
        second = torch.stack(
            (torch.maximum(x[:, 0], x[:, 1]), torch.maximum(y[:, 0], y[:, 1])),
            dim=-1,
        )
        return torch.stack((first, second), dim=1)

    def forward(self, input):
        if input.num_videos != 1:
            raise ValueError(
                "two-clock v1 requires per-GPU video batch 1 because safe spatial "
                "memory has a ragged refresh trajectory"
            )
        student_backbone = self._forward_refresh_images(input)
        student_backbone = self.prepare_prompt_inputs(student_backbone, input)
        student_outputs = self.forward_tracking(student_backbone, input)

        teacher = self.teacher_model
        if teacher is None:
            raise RuntimeError("two-clock training requires an online SAM2.1-L teacher")
        device = input.flat_img_batch.device
        if next(teacher.parameters()).device != device:
            teacher.to(device)
        teacher.eval()
        with torch.no_grad():
            teacher_backbone = teacher.forward_image(input.flat_img_batch)
            self._copy_prompt_plan(student_backbone, teacher_backbone)
            teacher_outputs = teacher.forward_tracking(teacher_backbone, input)
        self._attach_teacher_targets(student_outputs, teacher_outputs)
        return student_outputs

    def _schedule_from_input(self, input) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        identifiers = input.metadata.unique_objects_identifier
        if identifiers.shape[-1] != METADATA_WIDTH:
            raise ValueError(
                f"two-clock metadata width must be {METADATA_WIDTH}, got {identifiers.shape[-1]}"
            )
        num_frames, num_objects = identifiers.shape[:2]
        num_videos = input.num_videos
        refresh = torch.empty(num_frames, num_videos, dtype=torch.bool, device=identifiers.device)
        source_stage = torch.empty(num_frames, num_videos, dtype=torch.long, device=identifiers.device)
        age = torch.empty(num_frames, num_videos, dtype=torch.long, device=identifiers.device)
        mapping = input.obj_to_frame_idx
        for stage in range(num_frames):
            for video in range(num_videos):
                selected = mapping[stage, :, 1] == video
                if not selected.any():
                    raise ValueError(f"video {video} has no objects at stage {stage}")
                rows = identifiers[stage, selected]
                if self.experiment.name in {"O0", "O2"}:
                    values = (True, stage, 0)
                else:
                    raw = rows[:, META_RAW_FRAME]
                    source_raw = rows[:, META_SOURCE_RAW]
                    ages = rows[:, META_AGE]
                    refresh_values = rows[:, META_REFRESH].bool()
                    for tensor, name in (
                        (raw, "raw frame"),
                        (source_raw, "source frame"),
                        (ages, "age"),
                        (refresh_values, "refresh"),
                    ):
                        if not torch.all(tensor == tensor[0]):
                            raise ValueError(f"per-video {name} differs across objects")
                    values = (bool(refresh_values[0]), stage - int(ages[0]), int(ages[0]))
                refresh[stage, video], source_stage[stage, video], age[stage, video] = values
        if not torch.all(refresh[0]):
            raise ValueError("the prompted anchor must refresh")
        if age.min() < 0 or age.max() > self.max_feature_age:
            raise ValueError("feature age is outside the registered range")
        if not torch.equal(source_stage, torch.arange(num_frames, device=age.device)[:, None] - age):
            raise ValueError("source stage and age are inconsistent")
        return refresh, source_stage, age

    def _forward_refresh_images(self, input) -> dict:
        refresh, source_stage, age = self._schedule_from_input(input)
        num_frames, num_videos = refresh.shape
        flat_source = (
            torch.arange(num_videos, device=refresh.device)[None, :] * num_frames
            + source_stage
        )
        refresh_flat = torch.nonzero(refresh.transpose(0, 1).flatten(), as_tuple=False).flatten()
        source_flat = flat_source.transpose(0, 1).flatten()
        if not torch.equal(torch.unique(source_flat), refresh_flat):
            raise ValueError("source map does not exactly reference refresh frames")
        encoded = self.forward_image(input.flat_img_batch[refresh_flat])
        position = {int(frame): index for index, frame in enumerate(refresh_flat.tolist())}
        gather = torch.tensor(
            [position[int(frame)] for frame in source_flat.tolist()],
            device=source_flat.device,
            dtype=torch.long,
        )
        gathered = {}
        for key, value in encoded.items():
            if torch.is_tensor(value):
                gathered[key] = value.index_select(0, gather)
            elif isinstance(value, list):
                gathered[key] = [item.index_select(0, gather) for item in value]
            else:
                gathered[key] = value
        gathered["two_clock_refresh"] = refresh
        gathered["two_clock_source_stage"] = source_stage
        gathered["two_clock_age"] = age
        self.encoder_calls_last_forward = int(refresh.sum().item())
        return gathered

    def forward_tracking(self, backbone_out, input, return_dict=False):
        _, raw_vision_feats, vision_pos, feat_sizes = self._prepare_backbone_features(
            backbone_out
        )
        num_frames = backbone_out["num_frames"]
        init_cond = backbone_out["init_cond_frames"]
        order = init_cond + backbone_out["frames_not_in_init_cond"]
        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        identifiers = input.metadata.unique_objects_identifier
        prompt_kind = int(backbone_out["two_clock_prompt_kind"])
        for stage in order:
            img_ids = input.flat_obj_to_img_idx[stage]
            raw_feats = [feature[:, img_ids] for feature in raw_vision_feats]
            current_pos = [position[:, img_ids] for position in vision_pos]
            ages = identifiers[stage, :, META_AGE].long()
            if self.experiment.name in {"O0", "O2"}:
                ages = torch.zeros_like(ages)
            read_feats = self._condition_read_features(raw_feats, ages)
            self._two_clock_context = {
                "stage": stage,
                "age": ages,
                "raw": identifiers[stage, :, META_RAW_FRAME].long(),
                "source": identifiers[stage, :, META_SOURCE_RAW].long(),
            }
            current_out = self._two_clock_track_step(
                frame_idx=stage,
                is_init_cond_frame=stage in init_cond,
                read_vision_feats=read_feats,
                raw_vision_feats=raw_feats,
                current_vision_pos_embeds=current_pos,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage),
                output_dict=output_dict,
                num_frames=num_frames,
                run_mem_encoder=True,
            )
            current_out.update(
                {
                    "two_clock_gt_valid": identifiers[stage, :, META_GT_VALID].bool(),
                    "two_clock_refresh": identifiers[stage, :, META_REFRESH].bool()
                    if self.experiment.name not in {"O0", "O2"}
                    else torch.ones_like(identifiers[stage, :, META_REFRESH], dtype=torch.bool),
                    "two_clock_age": ages,
                    "two_clock_raw_frame": identifiers[stage, :, META_RAW_FRAME].long(),
                    "two_clock_source_raw": identifiers[stage, :, META_SOURCE_RAW].long()
                    if self.experiment.name not in {"O0", "O2"}
                    else identifiers[stage, :, META_RAW_FRAME].long(),
                    "two_clock_prompt_kind": prompt_kind,
                    "two_clock_is_anchor": stage in init_cond,
                    "two_clock_encoder_calls": self.encoder_calls_last_forward,
                }
            )
            target = "cond_frame_outputs" if stage in init_cond else "non_cond_frame_outputs"
            output_dict[target][stage] = current_out
        self._two_clock_context = None
        if return_dict:
            return output_dict
        combined = {**output_dict["cond_frame_outputs"], **output_dict["non_cond_frame_outputs"]}
        return [combined[stage] for stage in range(num_frames)]

    def _condition_read_features(
        self, raw_features: list[torch.Tensor], age: torch.Tensor
    ) -> list[torch.Tensor]:
        if self.age_conditioner is None:
            return list(raw_features)
        names = ("s0", "s1", "deep")[-len(raw_features) :]
        return [
            self.age_conditioner.condition_seq(name, feature, age)
            for name, feature in zip(names, raw_features)
        ]

    def _two_clock_track_step(
        self,
        *,
        frame_idx,
        is_init_cond_frame,
        read_vision_feats,
        raw_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        run_mem_encoder=True,
    ) -> dict:
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        high_res_features = [
            feature.permute(1, 2, 0).reshape(feature.size(1), feature.size(2), *size)
            for feature, size in zip(read_vision_feats[:-1], feat_sizes[:-1])
        ]
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            pix_feat = read_vision_feats[-1].permute(1, 2, 0).reshape(
                -1, self.hidden_dim, *feat_sizes[-1]
            )
            sam_outputs = self._use_mask_as_output(pix_feat, high_res_features, mask_inputs)
        else:
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx,
                is_init_cond_frame,
                read_vision_feats[-1:],
                current_vision_pos_embeds[-1:],
                feat_sizes[-1:],
                output_dict,
                num_frames,
            )
            if self.age_conditioner is not None:
                pix_feat = self.age_conditioner.condition_bchw(
                    "post", pix_feat, self._two_clock_context["age"]
                )
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=self._use_multimask(is_init_cond_frame, point_inputs),
            )
        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs
        current_out.update(
            {
                "distill_F16": self._seq_feature_to_bchw(read_vision_feats[-1], feat_sizes[-1]),
                "distill_F_M": pix_feat,
                "distill_is_init_cond_frame": bool(is_init_cond_frame),
                "multistep_pred_masks": low_res_masks,
                "multistep_pred_masks_high_res": high_res_masks,
                "multistep_pred_multimasks": [low_res_multimasks],
                "multistep_pred_multimasks_high_res": [high_res_multimasks],
                "multistep_pred_ious": [ious],
                "multistep_point_inputs": [point_inputs],
                "multistep_object_score_logits": [object_score_logits],
                "pred_masks": low_res_masks,
                "pred_masks_high_res": high_res_masks,
                "obj_ptr": obj_ptr,
                "object_score_logits": object_score_logits,
                "two_clock_prediction_raw": self._two_clock_context["raw"],
                "two_clock_observation_raw": self._two_clock_context["source"],
            }
        )
        refresh = self._two_clock_context["age"].eq(0)
        if not torch.all(refresh == refresh[0]):
            raise ValueError("safe memory requires one shared per-video refresh action")
        write_spatial = run_mem_encoder and (
            not self.experiment.suppress_stale_spatial_writes or bool(refresh[0])
        )
        self._encode_memory_in_output(
            raw_vision_feats,
            feat_sizes,
            point_inputs,
            write_spatial,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        current_out["two_clock_spatial_write"] = write_spatial
        return current_out

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        if track_in_reverse:
            raise ValueError("two-clock v1 is forward causal only")
        batch = current_vision_feats[-1].size(1)
        channels = self.hidden_dim
        height, width = feat_sizes[-1]
        device = current_vision_feats[-1].device
        if self.num_maskmem == 0:
            return current_vision_feats[-1].permute(1, 2, 0).reshape(
                batch, channels, height, width
            )
        if is_init_cond_frame:
            if self.directly_add_no_mem_embed:
                feature = current_vision_feats[-1] + self.no_mem_embed
                return feature.permute(1, 2, 0).reshape(batch, channels, height, width)
            memory = [self.no_mem_embed.expand(1, batch, self.mem_dim)]
            memory_pos = [self.no_mem_pos_enc.expand(1, batch, self.mem_dim)]
            num_pointer_tokens = 0
        else:
            memory, memory_pos = [], []
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond, unselected_cond = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            spatial_entries = [(0, output) for output in selected_cond.values()]
            available = [
                (stage, output)
                for stage, output in output_dict["non_cond_frame_outputs"].items()
                if stage < frame_idx and output.get("maskmem_features") is not None
            ]
            available = sorted(available)[-(self.num_maskmem - 1) :]
            first_tpos = self.num_maskmem - len(available)
            spatial_entries.extend(
                (first_tpos + index, output)
                for index, (_, output) in enumerate(available)
            )
            for t_pos, previous in spatial_entries:
                features = previous["maskmem_features"].to(device, non_blocking=True)
                memory.append(features.flatten(2).permute(2, 0, 1))
                position = previous["maskmem_pos_enc"][-1].to(device)
                slot_position = self.maskmem_tpos_enc[
                    self.num_maskmem - t_pos - 1
                ].reshape(1, self.mem_dim, 1, 1)
                position = position + slot_position
                if self.memory_time_conditioner is not None:
                    recency = (
                        self._two_clock_context["raw"]
                        - previous["two_clock_prediction_raw"]
                    )
                    freshness = previous["two_clock_prediction_raw"] - previous[
                        "two_clock_observation_raw"
                    ]
                    position = position + self.memory_time_conditioner.spatial_delta(
                        recency, freshness
                    )[:, :, None, None]
                memory_pos.append(position.flatten(2).permute(2, 0, 1))
            pointer_items = [(stage, output) for stage, output in selected_cond.items()]
            max_pointer_frames = min(num_frames, self.max_obj_ptrs_in_encoder)
            for difference in range(1, max_pointer_frames):
                stage = frame_idx - difference
                if stage < 0:
                    break
                output = output_dict["non_cond_frame_outputs"].get(
                    stage, unselected_cond.get(stage)
                )
                if output is not None:
                    pointer_items.append((stage, output))
            if len(pointer_items) > max_pointer_frames:
                raise AssertionError("object-pointer selection exceeds official capacity")
            if pointer_items:
                obj_ptrs = torch.stack([output["obj_ptr"] for _, output in pointer_items])
                stage_delta = torch.tensor(
                    [frame_idx - stage for stage, _ in pointer_items], device=device
                )
                if self.add_tpos_enc_to_obj_ptrs:
                    tpos_dim = channels if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                    pointer_pos = get_1d_sine_pe(
                        stage_delta / max(max_pointer_frames - 1, 1), dim=tpos_dim
                    )
                    pointer_pos = self.obj_ptr_tpos_proj(pointer_pos)
                    pointer_pos = pointer_pos[:, None].expand(-1, batch, self.mem_dim)
                else:
                    pointer_pos = obj_ptrs.new_zeros(
                        len(pointer_items), batch, self.mem_dim
                    )
                pointer_delta = None
                if self.memory_time_conditioner is not None:
                    recency = torch.stack(
                        [
                            self._two_clock_context["raw"] - output["two_clock_prediction_raw"]
                            for _, output in pointer_items
                        ]
                    )
                    freshness = torch.stack(
                        [
                            output["two_clock_prediction_raw"]
                            - output["two_clock_observation_raw"]
                            for _, output in pointer_items
                        ]
                    )
                    pointer_delta = self.memory_time_conditioner.pointer_delta(
                        recency.flatten(), freshness.flatten()
                    ).reshape(len(pointer_items), batch, channels)
                if self.mem_dim < channels:
                    tokens_per_pointer = channels // self.mem_dim
                    obj_ptrs = obj_ptrs.reshape(
                        -1, batch, tokens_per_pointer, self.mem_dim
                    ).permute(0, 2, 1, 3).flatten(0, 1)
                    pointer_pos = pointer_pos.repeat_interleave(tokens_per_pointer, dim=0)
                    if pointer_delta is not None:
                        pointer_delta = pointer_delta.reshape(
                            -1, batch, tokens_per_pointer, self.mem_dim
                        ).permute(0, 2, 1, 3).flatten(0, 1)
                if pointer_delta is not None:
                    pointer_pos = pointer_pos + pointer_delta
                memory.append(obj_ptrs)
                memory_pos.append(pointer_pos)
                num_pointer_tokens = obj_ptrs.shape[0]
            else:
                num_pointer_tokens = 0
        fused = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=torch.cat(memory),
            memory_pos=torch.cat(memory_pos),
            num_obj_ptr_tokens=num_pointer_tokens,
        )
        return fused.permute(1, 2, 0).reshape(batch, channels, height, width)

    @staticmethod
    def _attach_teacher_targets(student_outputs, teacher_outputs) -> None:
        if len(student_outputs) != len(teacher_outputs):
            raise ValueError("teacher/student frame count differs")
        for student, teacher in zip(student_outputs, teacher_outputs):
            mapping = {
                "teacher_distill_F_M": "distill_F_M",
                "teacher_pred_masks": "pred_masks",
                "teacher_obj_ptr": "obj_ptr",
            }
            for target, source in mapping.items():
                if source not in teacher:
                    raise KeyError(f"teacher output is missing {source}")
                student[target] = teacher[source].detach()
            score_outputs = teacher.get("multistep_object_score_logits")
            if not score_outputs:
                raise KeyError("teacher output is missing object-score logits")
            student["teacher_object_score_logits"] = score_outputs[-1].detach()
            student["teacher_prompt_match"] = True
