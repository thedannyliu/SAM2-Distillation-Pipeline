"""Training model extensions for EdgeTAM distillation.

This module is imported by Hydra configs only after the SAM2 training package is
available on PYTHONPATH.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.utils.checkpoint

from sam2_distill.edgetam.compat import patch_edgetam_perceiver_view
from sam2_distill.edgetam.teacher_features import (
    TeacherFeatureCache,
    attach_synthetic_teacher_features,
    attach_teacher_features,
    extract_teacher_model_state,
    flatten_tracking_outputs,
)
from sam2_distill.models.sam2_object_slots import ObjectSlotModelMixin
from training.model.sam2 import SAM2Train


patch_edgetam_perceiver_view()


class EdgeTAMTrain(ObjectSlotModelMixin, SAM2Train):
    """SAM2Train variant that exposes features needed for EdgeTAM distillation."""

    def __init__(
        self,
        *args,
        image_encoder_forward_batch_size: int | None = None,
        image_encoder_activation_checkpoint: bool = False,
        trainable_module_mode: str | None = None,
        freeze_batchnorm: bool = False,
        object_slot_count: int = 0,
        object_slot_min_objects: int = 4,
        expose_obj_ptr_for_distillation: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.image_encoder_forward_batch_size = image_encoder_forward_batch_size
        self.image_encoder_activation_checkpoint = image_encoder_activation_checkpoint
        self.trainable_module_mode = trainable_module_mode
        self.freeze_batchnorm = freeze_batchnorm
        self.expose_obj_ptr_for_distillation = (
            expose_obj_ptr_for_distillation
        )
        self._frozen_eval_modules: list[torch.nn.Module] = []
        self.trainable_parameter_summary = None
        self._init_object_slots(
            object_slot_count=object_slot_count,
            object_slot_min_objects=object_slot_min_objects,
        )
        if trainable_module_mode is not None:
            self._apply_trainable_module_mode(trainable_module_mode)
        if self.freeze_batchnorm:
            self._freeze_batchnorm_modules()
        if trainable_module_mode is not None:
            self.trainable_parameter_summary = self._parameter_summary()

    def _apply_trainable_module_mode(self, mode: str) -> dict[str, int]:
        if mode not in {
            "image_neck_only",
            "image_encoder_only",
            "mask_decoder_only",
            "mask_decoder_memory",
            "memory_only",
            "memory_perceiver_full",
            "image_encoder_memory_perceiver",
            "image_encoder_mask_decoder",
            "image_encoder_mask_decoder_memory",
            "object_slot_decoder",
            "object_slot_shared_kv",
            "object_slot_multiplex",
        }:
            raise ValueError(
                "trainable_module_mode must be one of: image_neck_only, "
                "image_encoder_only, mask_decoder_only, "
                "mask_decoder_memory, memory_only, memory_perceiver_full, "
                "image_encoder_memory_perceiver, "
                "image_encoder_mask_decoder, "
                "image_encoder_mask_decoder_memory, object_slot_decoder, "
                "object_slot_shared_kv, object_slot_multiplex"
            )

        for param in self.parameters():
            param.requires_grad = False

        if mode == "image_neck_only":
            modules = [self.image_encoder.neck]
        elif mode == "image_encoder_only":
            modules = [self.image_encoder]
        elif mode in {
            "object_slot_decoder",
            "object_slot_shared_kv",
            "object_slot_multiplex",
        }:
            if self.object_slot_decoder is None:
                raise ValueError(f"{mode} requires object_slot_count > 0")
            modules = [self.object_slot_decoder]
            if mode == "object_slot_shared_kv":
                slot_memory_scale = getattr(
                    self.memory_attention, "slot_memory_scale", None
                )
                if not isinstance(slot_memory_scale, torch.nn.Parameter):
                    raise ValueError(
                        "object_slot_shared_kv requires "
                        "SharedSlotMemoryAttention"
                    )
                slot_memory_scale.requires_grad = True
                object_residual = getattr(
                    self.memory_attention, "object_residual", None
                )
                if isinstance(object_residual, torch.nn.Module):
                    modules.append(object_residual)
            elif mode == "object_slot_multiplex":
                if not hasattr(
                    self.memory_encoder, "multiplex_mask_downsampler"
                ):
                    raise ValueError(
                        "object_slot_multiplex requires "
                        "SlotPreservingMemoryEncoder"
                    )
                if not hasattr(self.memory_attention, "slot_pointer_pos"):
                    raise ValueError(
                        "object_slot_multiplex requires "
                        "SlotPreservingMemoryAttention"
                    )
                modules.extend(
                    [
                        self.memory_encoder,
                        self.memory_attention,
                        self.sam_mask_decoder,
                    ]
                )
                for name in ("obj_ptr_proj", "obj_ptr_tpos_proj"):
                    module = getattr(self, name, None)
                    if isinstance(module, torch.nn.Module):
                        modules.append(module)
                for name in (
                    "maskmem_tpos_enc",
                    "no_mem_embed",
                    "no_mem_pos_enc",
                    "no_obj_ptr",
                    "no_obj_embed_spatial",
                ):
                    parameter = getattr(self, name, None)
                    if isinstance(parameter, torch.nn.Parameter):
                        parameter.requires_grad = True
        elif mode == "mask_decoder_only":
            modules = [self.sam_mask_decoder]
        elif mode in {
            "mask_decoder_memory",
            "memory_only",
            "memory_perceiver_full",
            "image_encoder_memory_perceiver",
        }:
            modules = [self.memory_attention, self.memory_encoder]
            if mode == "mask_decoder_memory":
                modules.append(self.sam_mask_decoder)
            if mode in {
                "memory_perceiver_full",
                "image_encoder_memory_perceiver",
            }:
                spatial_perceiver = getattr(self, "spatial_perceiver", None)
                if not isinstance(spatial_perceiver, torch.nn.Module):
                    raise ValueError(
                        f"{mode} requires model.spatial_perceiver"
                    )
                modules.append(spatial_perceiver)
            if mode == "image_encoder_memory_perceiver":
                modules.append(self.image_encoder)
            for name in ("obj_ptr_proj", "obj_ptr_tpos_proj"):
                module = getattr(self, name, None)
                if isinstance(module, torch.nn.Module):
                    modules.append(module)
            for name in (
                "maskmem_tpos_enc",
                "no_mem_embed",
                "no_mem_pos_enc",
                "no_obj_ptr",
                "no_obj_embed_spatial",
            ):
                parameter = getattr(self, name, None)
                if isinstance(parameter, torch.nn.Parameter):
                    parameter.requires_grad = True
        elif mode == "image_encoder_mask_decoder":
            modules = [self.image_encoder, self.sam_mask_decoder]
        else:
            for param in self.parameters():
                param.requires_grad = True
            modules = []
            for param in self.sam_prompt_encoder.parameters():
                param.requires_grad = False

        for module in modules:
            for param in module.parameters():
                param.requires_grad = True

        frozen_candidates = [
            self.image_encoder,
            self.sam_prompt_encoder,
            self.sam_mask_decoder,
            self.memory_attention,
            self.memory_encoder,
            getattr(self, "spatial_perceiver", None),
            self.object_slot_decoder,
        ]
        self._frozen_eval_modules = [
            module
            for module in frozen_candidates
            if module is not None
            and not any(param.requires_grad for param in module.parameters())
        ]

        return self._parameter_summary()

    def _parameter_summary(self) -> dict[str, int]:
        trainable = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )
        total = sum(param.numel() for param in self.parameters())
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "frozen_parameters": int(total - trainable),
        }

    def _freeze_batchnorm_modules(self) -> None:
        for module in self.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
                for param in module.parameters(recurse=False):
                    param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_batchnorm:
            self._freeze_batchnorm_modules()
        if mode:
            for module in self._frozen_eval_modules:
                module.eval()
        return self

    def forward_image(self, img_batch: torch.Tensor):
        if (
            self.image_encoder_forward_batch_size is None
            or self.image_encoder_forward_batch_size <= 0
            or img_batch.size(0) <= self.image_encoder_forward_batch_size
        ):
            return self._forward_image_impl(img_batch)

        chunks = []
        for start in range(0, img_batch.size(0), self.image_encoder_forward_batch_size):
            chunk = img_batch[start : start + self.image_encoder_forward_batch_size]
            if self.image_encoder_activation_checkpoint and torch.is_grad_enabled():
                chunk_tuple = torch.utils.checkpoint.checkpoint(
                    self._forward_image_as_tuple,
                    chunk,
                    use_reentrant=False,
                )
                chunks.append(self._tuple_to_backbone_output(chunk_tuple))
            else:
                chunks.append(self._forward_image_impl(chunk))
        return self._concat_backbone_outputs(chunks)

    def forward_tracking(self, backbone_out, input, return_dict=False):
        if not self.expose_obj_ptr_for_distillation:
            return super().forward_tracking(
                backbone_out,
                input,
                return_dict=return_dict,
            )
        output_dict = super().forward_tracking(
            backbone_out,
            input,
            return_dict=True,
        )
        if return_dict:
            return output_dict
        return flatten_tracking_outputs(
            output_dict,
            input.num_frames,
            keep_obj_ptr=True,
        )

    def _forward_image_impl(self, img_batch: torch.Tensor) -> dict:
        if getattr(
            self.image_encoder,
            "outputs_preprojected_sam_features",
            False,
        ):
            return self.image_encoder(img_batch)
        return super().forward_image(img_batch)

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
        frames_to_add_correction_pt=None,
        gt_masks=None,
    ):
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
        self._prepare_multiplex_training_layout(
            current_vision_feats[-1].shape[1]
        )
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
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

        current_out["distill_F16"] = self._seq_feature_to_bchw(
            current_vision_feats[-1], feat_sizes[-1]
        )
        current_out["distill_F_M"] = pix_feat
        current_out["distill_is_init_cond_frame"] = bool(is_init_cond_frame)
        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        # Upstream SAM2 leaves ``sam_outputs`` uninitialized when the
        # iterative-correction loop has zero iterations. Exact-box controls
        # intentionally request zero correction clicks, so keep the initial
        # SAM prediction instead of entering that unsupported path.
        if (
            self.num_correction_pt_per_frame > 0
            and frame_idx in frames_to_add_correction_pt
        ):
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame,
                point_inputs,
                gt_masks,
                high_res_features,
                pix_feat,
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                object_score_logits,
                current_out,
            )
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = final_sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out

    @staticmethod
    def _seq_feature_to_bchw(feature: torch.Tensor, feat_size: tuple[int, int]) -> torch.Tensor:
        batch = feature.size(1)
        channels = feature.size(2)
        return feature.permute(1, 2, 0).reshape(batch, channels, *feat_size)

    def _forward_image_as_tuple(self, img_batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self._forward_image_impl(img_batch)
        return (
            out["vision_features"],
            *out["vision_pos_enc"],
            *out["backbone_fpn"],
        )

    @staticmethod
    def _tuple_to_backbone_output(values: tuple[torch.Tensor, ...]) -> dict:
        num_levels = (len(values) - 1) // 2
        if num_levels < 1 or len(values) != 1 + 2 * num_levels:
            raise ValueError(f"unexpected checkpointed backbone tuple length: {len(values)}")
        return {
            "vision_features": values[0],
            "vision_pos_enc": list(values[1 : 1 + num_levels]),
            "backbone_fpn": list(values[1 + num_levels :]),
        }

    @staticmethod
    def _concat_backbone_outputs(chunks: list[dict]) -> dict:
        if not chunks:
            raise ValueError("cannot concatenate empty backbone outputs")
        out = {}
        for key, value in chunks[0].items():
            if isinstance(value, torch.Tensor):
                out[key] = torch.cat([chunk[key] for chunk in chunks], dim=0)
            elif isinstance(value, list):
                out[key] = [
                    torch.cat([chunk[key][idx] for chunk in chunks], dim=0)
                    for idx in range(len(value))
                ]
            else:
                out[key] = value
        return out


class EdgeTAMTrainWithTeacher(EdgeTAMTrain):
    """EdgeTAM training model that attaches frozen teacher features to outputs.

    The upstream SAM2 trainer calls ``loss(model(batch), batch.masks)`` and does
    not pass the batch into the loss. Teacher distillation targets therefore
    need to be present in the per-frame output dictionaries before the loss is
    evaluated.
    """

    def __init__(
        self,
        *args,
        teacher_model: torch.nn.Module | None = None,
        teacher_feature_cache_path: str | None = None,
        teacher_model_config: str | None = None,
        teacher_checkpoint: str | None = None,
        synthetic_teacher: bool = False,
        synthetic_teacher_offset: float = 0.01,
        freeze_teacher: bool = True,
        pair_teacher_student_prompts: bool = False,
        **kwargs,
    ):
        teacher_prompt_kwargs = {
            key: kwargs[key]
            for key in (
                "prob_to_use_pt_input_for_train",
                "prob_to_use_box_input_for_train",
                "prob_to_sample_from_gt_for_train",
                "num_frames_to_correct_for_train",
                "rand_frames_to_correct_for_train",
                "add_all_frames_to_correct_as_cond",
                "num_init_cond_frames_for_train",
                "rand_init_cond_frames_for_train",
                "num_correction_pt_per_frame",
                "use_act_ckpt_iterative_pt_sampling",
                "prob_to_use_pt_input_for_eval",
                "prob_to_use_box_input_for_eval",
                "num_frames_to_correct_for_eval",
                "num_init_cond_frames_for_eval",
                "forward_backbone_per_frame_for_eval",
                "expose_obj_ptr_for_distillation",
            )
            if key in kwargs
        }
        super().__init__(*args, **kwargs)
        if teacher_model is None and teacher_model_config is not None:
            if teacher_checkpoint is None:
                raise ValueError(
                    "teacher_checkpoint is required with teacher_model_config"
                )
            teacher_model = self._build_teacher_model(
                teacher_model_config,
                teacher_checkpoint,
                teacher_prompt_kwargs,
            )
        # Bypass nn.Module.__setattr__: the frozen online teacher must move and
        # run on each rank, but must not enter DDP, optimizer, or checkpoints.
        object.__setattr__(self, "_teacher_model", teacher_model)
        self.teacher_feature_cache = (
            TeacherFeatureCache(teacher_feature_cache_path)
            if teacher_feature_cache_path is not None
            else None
        )
        self.synthetic_teacher = synthetic_teacher
        self.synthetic_teacher_offset = synthetic_teacher_offset
        self.pair_teacher_student_prompts = pair_teacher_student_prompts
        if pair_teacher_student_prompts and self.num_correction_pt_per_frame != 0:
            raise ValueError(
                "paired prompt diagnostics require "
                "num_correction_pt_per_frame=0; model-dependent correction "
                "clicks need an explicit external correction plan"
            )

        if self._teacher_model is not None and freeze_teacher:
            self._teacher_model.eval()
            for param in self._teacher_model.parameters():
                param.requires_grad = False

        teacher_sources = [
            self._teacher_model is not None,
            self.teacher_feature_cache is not None,
            self.synthetic_teacher,
        ]
        if sum(teacher_sources) != 1:
            raise ValueError(
                "EdgeTAMTrainWithTeacher requires exactly one online teacher "
                "(teacher_model or teacher_model_config), "
                "teacher_feature_cache_path, or synthetic_teacher=True"
            )

    @staticmethod
    def _build_teacher_model(
        config_path: str,
        checkpoint_path: str,
        prompt_kwargs: dict,
    ) -> torch.nn.Module:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        config = OmegaConf.load(Path(config_path))
        model_config = config.model if "model" in config else config.trainer.model
        model_config._target_ = "sam2_distill.edgetam.train_model.EdgeTAMTrain"
        for key, value in prompt_kwargs.items():
            model_config[key] = value
        model_config.image_encoder_forward_batch_size = 1
        model_config.image_encoder_activation_checkpoint = False
        model_config.trainable_module_mode = None
        model_config.freeze_batchnorm = True
        teacher = instantiate(model_config, _recursive_=True)
        checkpoint = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=True
        )
        state = extract_teacher_model_state(checkpoint)
        teacher.load_state_dict(state, strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        return teacher

    @property
    def teacher_model(self):
        return self._teacher_model

    def forward(self, input):
        if self.pair_teacher_student_prompts:
            return self._forward_with_paired_prompts(input)

        student_outputs = super().forward(input)

        if self._teacher_model is not None:
            input_device = input.flat_img_batch.device
            teacher_parameter = next(self._teacher_model.parameters())
            if teacher_parameter.device != input_device:
                self._teacher_model.to(input_device)
            self._teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = self._teacher_model(input)
            attach_teacher_features(student_outputs, teacher_outputs)
        elif self.teacher_feature_cache is not None:
            self.teacher_feature_cache.attach(student_outputs)
        else:
            attach_synthetic_teacher_features(
                student_outputs,
                offset=self.synthetic_teacher_offset,
            )

        return student_outputs

    def _forward_with_paired_prompts(self, input):
        if self._teacher_model is None:
            raise ValueError("paired prompts require an online teacher model")
        if not self.training:
            raise RuntimeError("paired prompt diagnostics are training-only")

        input_device = input.flat_img_batch.device
        teacher_parameter = next(self._teacher_model.parameters())
        if teacher_parameter.device != input_device:
            self._teacher_model.to(input_device)
        self._teacher_model.eval()
        # SAM2Train.prepare_prompt_inputs switches policy using the root
        # module's training flag. Keep all teacher children in eval mode while
        # selecting the same training-time control path as the student.
        self._teacher_model.training = True

        try:
            student_backbone = self.forward_image(input.flat_img_batch)
            student_backbone = self.prepare_prompt_inputs(student_backbone, input)
            student_outputs = self.forward_tracking(student_backbone, input)

            with torch.no_grad():
                teacher_backbone = self._teacher_model.forward_image(
                    input.flat_img_batch
                )
                self._copy_prompt_plan(student_backbone, teacher_backbone)
                teacher_outputs = self._teacher_model.forward_tracking(
                    teacher_backbone,
                    input,
                )
        finally:
            self._teacher_model.training = False

        self._assert_paired_prompt_outputs(student_outputs, teacher_outputs)
        attach_teacher_features(student_outputs, teacher_outputs)
        for output in student_outputs:
            output["teacher_prompt_match"] = True
        return student_outputs

    @classmethod
    def _copy_prompt_plan(cls, source: dict, destination: dict) -> None:
        for key in (
            "gt_masks_per_frame",
            "num_frames",
            "use_pt_input",
            "init_cond_frames",
            "frames_not_in_init_cond",
            "mask_inputs_per_frame",
            "point_inputs_per_frame",
            "frames_to_add_correction_pt",
        ):
            if key not in source:
                raise KeyError(f"student prompt plan is missing {key!r}")
            destination[key] = cls._clone_prompt_value(source[key])

    @classmethod
    def _clone_prompt_value(cls, value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {
                key: cls._clone_prompt_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._clone_prompt_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._clone_prompt_value(item) for item in value)
        return value

    @classmethod
    def _prompt_values_equal(cls, left, right) -> bool:
        if torch.is_tensor(left) or torch.is_tensor(right):
            return (
                torch.is_tensor(left)
                and torch.is_tensor(right)
                and torch.equal(left, right)
            )
        if isinstance(left, dict) or isinstance(right, dict):
            return (
                isinstance(left, dict)
                and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(
                    cls._prompt_values_equal(left[key], right[key])
                    for key in left
                )
            )
        if isinstance(left, (list, tuple)) or isinstance(
            right, (list, tuple)
        ):
            return (
                isinstance(left, (list, tuple))
                and isinstance(right, (list, tuple))
                and len(left) == len(right)
                and all(
                    cls._prompt_values_equal(left_item, right_item)
                    for left_item, right_item in zip(left, right)
                )
            )
        return left == right

    @classmethod
    def _assert_paired_prompt_outputs(
        cls,
        student_outputs: list[dict],
        teacher_outputs: list[dict],
    ) -> None:
        if len(student_outputs) != len(teacher_outputs):
            raise RuntimeError(
                "paired prompt frame-count mismatch: "
                f"student={len(student_outputs)} teacher={len(teacher_outputs)}"
            )
        for frame_idx, (student, teacher) in enumerate(
            zip(student_outputs, teacher_outputs)
        ):
            for key in (
                "distill_is_init_cond_frame",
                "multistep_point_inputs",
            ):
                if key not in student or key not in teacher:
                    raise KeyError(
                        f"paired prompt output frame {frame_idx} is missing {key!r}"
                    )
                if not cls._prompt_values_equal(student[key], teacher[key]):
                    raise RuntimeError(
                        f"teacher/student prompt mismatch at frame {frame_idx}: {key}"
                    )
