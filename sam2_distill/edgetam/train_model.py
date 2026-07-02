"""Training model extensions for EdgeTAM distillation.

This module is imported by Hydra configs only after the SAM2 training package is
available on PYTHONPATH.
"""

from __future__ import annotations

import torch
import torch.utils.checkpoint

from sam2_distill.edgetam.teacher_features import (
    attach_synthetic_teacher_features,
    attach_teacher_features,
)
from training.model.sam2 import SAM2Train


class EdgeTAMTrain(SAM2Train):
    """SAM2Train variant that exposes features needed for EdgeTAM distillation."""

    def __init__(
        self,
        *args,
        image_encoder_forward_batch_size: int | None = None,
        image_encoder_activation_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.image_encoder_forward_batch_size = image_encoder_forward_batch_size
        self.image_encoder_activation_checkpoint = image_encoder_activation_checkpoint

    def forward_image(self, img_batch: torch.Tensor):
        if (
            self.image_encoder_forward_batch_size is None
            or self.image_encoder_forward_batch_size <= 0
            or img_batch.size(0) <= self.image_encoder_forward_batch_size
        ):
            return super().forward_image(img_batch)

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
                chunks.append(super().forward_image(chunk))
        return self._concat_backbone_outputs(chunks)

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
        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        if frame_idx in frames_to_add_correction_pt:
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
        out = super().forward_image(img_batch)
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
        synthetic_teacher: bool = False,
        synthetic_teacher_offset: float = 0.01,
        freeze_teacher: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.synthetic_teacher = synthetic_teacher
        self.synthetic_teacher_offset = synthetic_teacher_offset

        if self.teacher_model is not None and freeze_teacher:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        if self.teacher_model is None and not self.synthetic_teacher:
            raise ValueError("EdgeTAMTrainWithTeacher requires teacher_model or synthetic_teacher=True")

    def forward(self, input):
        student_outputs = super().forward(input)

        if self.teacher_model is not None:
            was_training = self.teacher_model.training
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = self.teacher_model(input)
            if was_training:
                self.teacher_model.train()
            attach_teacher_features(student_outputs, teacher_outputs)
        else:
            attach_synthetic_teacher_features(
                student_outputs,
                offset=self.synthetic_teacher_offset,
            )

        return student_outputs
