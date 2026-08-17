"""Streaming fixed-refresh predictor for two-clock SAM2.1-L checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from sam2.sam2_video_predictor import SAM2VideoPredictor

from sam2_distill.two_clock.experiments import get_experiment
from sam2_distill.two_clock.model import TwoClockSAM2Train
from sam2_distill.two_clock.schedule import (
    balanced_refresh_phase,
    fixed_refresh_source,
)
from sam2_distill.two_clock.temporal import (
    FeatureAgeConditioner,
    TwoClockMemoryConditioner,
)


class TwoClockVideoPredictor(SAM2VideoPredictor):
    """Official predictor API with real R1--R6 image-encoder skipping."""

    _condition_read_features = TwoClockSAM2Train._condition_read_features
    _two_clock_track_step = TwoClockSAM2Train._two_clock_track_step
    _prepare_memory_conditioned_features = (
        TwoClockSAM2Train._prepare_memory_conditioned_features
    )
    _seq_feature_to_bchw = staticmethod(TwoClockSAM2Train._seq_feature_to_bchw)

    def __init__(
        self,
        *args,
        experiment: str,
        max_feature_age: int = 5,
        fixed_refresh_interval: int = 1,
        refresh_phase_mode: str = "anchor",
        refresh_phase_seed: int = 250107256,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if fixed_refresh_interval < 1 or fixed_refresh_interval > max_feature_age + 1:
            raise ValueError("fixed refresh interval must be in [1, max_age + 1]")
        self.experiment = get_experiment(experiment)
        self.max_feature_age = max_feature_age
        self.fixed_refresh_interval = fixed_refresh_interval
        if refresh_phase_mode not in {"anchor", "balanced"}:
            raise ValueError("refresh phase mode must be anchor or balanced")
        self.refresh_phase_mode = refresh_phase_mode
        self.refresh_phase_seed = refresh_phase_seed
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
        self._two_clock_context = None
        self._two_clock_raw_features = None

    @torch.inference_mode()
    def init_state(self, *args, **kwargs):
        self._record_current_stream()
        state = super().init_state(*args, **kwargs)
        video_path = kwargs.get("video_path")
        if video_path is None and args:
            video_path = args[0]
        video_id = Path(str(video_path)).name if video_path is not None else ""
        phase = 0
        if self.refresh_phase_mode == "balanced":
            if not video_id:
                raise ValueError("balanced refresh phases require a video path")
            phase = balanced_refresh_phase(
                video_id,
                self.fixed_refresh_interval,
                seed=self.refresh_phase_seed,
            )
        state["two_clock_encoder_calls"] = 1
        state["two_clock_frame_requests"] = 0
        state["two_clock_refresh_interval"] = self.fixed_refresh_interval
        state["two_clock_refresh_phase"] = phase
        state["two_clock_anchor_frame"] = 0
        self._two_clock_last_state = state
        return state

    def reset_two_clock_telemetry(self) -> None:
        self._two_clock_stream_records = []
        self._two_clock_last_state = None

    def _record_current_stream(self) -> None:
        state = getattr(self, "_two_clock_last_state", None)
        if state is None or not state.get("two_clock_prompt_registered", False):
            return
        if not hasattr(self, "_two_clock_stream_records"):
            self._two_clock_stream_records = []
        self._two_clock_stream_records.append(
            {
                "encoder_calls": int(state["two_clock_encoder_calls"]),
                "frame_requests": int(state["two_clock_frame_requests"]),
                "tracking_frames": int(state["num_frames"])
                - int(state["two_clock_anchor_frame"]),
            }
        )
        state["two_clock_prompt_registered"] = False

    def two_clock_telemetry(self) -> dict[str, int | float]:
        records = list(getattr(self, "_two_clock_stream_records", []))
        state = getattr(self, "_two_clock_last_state", None)
        if state is not None and state.get("two_clock_prompt_registered", False):
            records.append(
                {
                    "encoder_calls": int(state["two_clock_encoder_calls"]),
                    "frame_requests": int(state["two_clock_frame_requests"]),
                    "tracking_frames": int(state["num_frames"])
                    - int(state["two_clock_anchor_frame"]),
                }
            )
        encoder_calls = sum(record["encoder_calls"] for record in records)
        tracking_frames = sum(record["tracking_frames"] for record in records)
        return {
            "streams": len(records),
            "encoder_calls": encoder_calls,
            "frame_requests": sum(record["frame_requests"] for record in records),
            "tracking_frames": tracking_frames,
            "refresh_rate": encoder_calls / max(tracking_frames, 1),
        }

    @torch.inference_mode()
    def reset_state(self, inference_state):
        self._record_current_stream()
        super().reset_state(inference_state)
        inference_state["cached_features"].clear()
        inference_state["two_clock_encoder_calls"] = 0
        inference_state["two_clock_frame_requests"] = 0
        inference_state["two_clock_anchor_frame"] = 0
        inference_state["two_clock_refresh_phase"] = 0
        inference_state["two_clock_prompt_registered"] = False

    def _register_prompt_anchor(self, inference_state, frame_idx: int) -> None:
        existing = inference_state.get("two_clock_prompt_registered", False)
        anchor = int(inference_state.get("two_clock_anchor_frame", 0))
        if existing and anchor != frame_idx:
            raise ValueError(
                "one two-clock inference state requires a shared prompt frame; "
                "use separate per-object inference for later-appearing objects"
            )
        if not existing and frame_idx != anchor:
            inference_state["cached_features"].clear()
            inference_state["two_clock_encoder_calls"] = 0
        inference_state["two_clock_anchor_frame"] = int(frame_idx)
        inference_state["two_clock_prompt_registered"] = True

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        self._register_prompt_anchor(inference_state, frame_idx)
        return super().add_new_mask(inference_state, frame_idx, obj_id, mask)

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, **kwargs):
        self._register_prompt_anchor(inference_state, frame_idx)
        return super().add_new_points_or_box(
            inference_state, frame_idx, obj_id, **kwargs
        )

    def _source_frame(self, inference_state, frame_idx: int) -> int:
        anchor = int(inference_state.get("two_clock_anchor_frame", 0))
        phase = int(inference_state.get("two_clock_refresh_phase", 0))
        return fixed_refresh_source(
            frame_idx,
            anchor=anchor,
            interval=self.fixed_refresh_interval,
            phase=phase,
        )

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        source_frame = self._source_frame(inference_state, frame_idx)
        image, backbone_out = inference_state["cached_features"].get(
            source_frame, (None, None)
        )
        if backbone_out is None:
            device = inference_state["device"]
            image = (
                inference_state["images"][source_frame]
                .to(device)
                .float()
                .unsqueeze(0)
            )
            backbone_out = self.forward_image(image)
            inference_state["cached_features"] = {
                source_frame: (image, backbone_out)
            }
            inference_state["two_clock_encoder_calls"] = (
                inference_state.get("two_clock_encoder_calls", 0) + 1
            )
        inference_state["two_clock_frame_requests"] = (
            inference_state.get("two_clock_frame_requests", 0) + 1
        )

        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded = {
            "backbone_fpn": [
                feature.expand(batch_size, -1, -1, -1)
                for feature in backbone_out["backbone_fpn"]
            ],
            "vision_pos_enc": [
                position.expand(batch_size, -1, -1, -1)
                for position in backbone_out["vision_pos_enc"]
            ],
        }
        backbone, raw_features, positions, sizes = self._prepare_backbone_features(
            expanded
        )
        age_value = frame_idx - source_frame
        age = torch.full(
            (batch_size,), age_value, device=raw_features[-1].device, dtype=torch.long
        )
        read_features = self._condition_read_features(raw_features, age)
        self._two_clock_context = {
            "stage": frame_idx,
            "age": age,
            "raw": torch.full_like(age, frame_idx),
            "source": torch.full_like(age, source_frame),
        }
        self._two_clock_raw_features = raw_features
        return expanded_image, backbone, read_features, positions, sizes

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
    ):
        if track_in_reverse:
            raise ValueError("two-clock v1 evaluates forward causal tracking only")
        if prev_sam_mask_logits is not None:
            raise ValueError("two-clock v1 evaluation does not use correction clicks")
        if self._two_clock_raw_features is None:
            raise RuntimeError("raw cached features are unavailable")
        return self._two_clock_track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            read_vision_feats=current_vision_feats,
            raw_vision_feats=self._two_clock_raw_features,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=num_frames,
            run_mem_encoder=run_mem_encoder,
        )

    def _run_single_frame_inference(self, *args, **kwargs):
        compact, masks = super()._run_single_frame_inference(*args, **kwargs)
        if self._two_clock_context is None:
            raise RuntimeError("two-clock context was not populated")
        compact.update(
            {
                "two_clock_prediction_raw": self._two_clock_context["raw"].clone(),
                "two_clock_observation_raw": self._two_clock_context[
                    "source"
                ].clone(),
                "two_clock_spatial_write": compact["maskmem_features"] is not None,
            }
        )
        return compact, masks


def build_two_clock_video_predictor(
    *,
    resolved_config: str | Path,
    checkpoint_path: str | Path,
    experiment: str,
    refresh_interval: int,
    refresh_phase_mode: str = "anchor",
    refresh_phase_seed: int = 250107256,
    device: str | torch.device,
) -> tuple[TwoClockVideoPredictor, dict]:
    """Build and strictly load a training or official SAM2.1-L checkpoint."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(resolved_config))
    model = config.model if "model" in config else config.trainer.model
    model._target_ = "sam2_distill.two_clock.predictor.TwoClockVideoPredictor"
    model.experiment = experiment
    model.fixed_refresh_interval = refresh_interval
    model.refresh_phase_mode = refresh_phase_mode
    model.refresh_phase_seed = refresh_phase_seed
    for key in (
        "teacher_model_config",
        "teacher_checkpoint",
        "pair_teacher_student_prompts",
        "box_jitter_fraction",
        "freeze_batchnorm",
        "image_encoder_activation_checkpoint",
        "image_encoder_forward_batch_size",
        "trainable_module_mode",
        "expose_obj_ptr_for_distillation",
        "prob_to_use_pt_input_for_train",
        "prob_to_use_box_input_for_train",
        "prob_to_sample_from_gt_for_train",
        "num_frames_to_correct_for_train",
        "rand_frames_to_correct_for_train",
        "num_init_cond_frames_for_train",
        "rand_init_cond_frames_for_train",
        "num_correction_pt_per_frame",
        "use_act_ckpt_iterative_pt_sampling",
        "prob_to_use_pt_input_for_eval",
        "prob_to_use_box_input_for_eval",
        "num_frames_to_correct_for_eval",
        "num_init_cond_frames_for_eval",
        "forward_backbone_per_frame_for_eval",
    ):
        model.pop(key, None)
    predictor = instantiate(model, _recursive_=True)
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True, mmap=True
    )
    state = checkpoint.get("model", checkpoint)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    predictor.load_state_dict(state, strict=True)
    predictor = predictor.to(device).eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    return predictor, {
        "experiment": experiment,
        "refresh_interval": refresh_interval,
        "refresh_phase_mode": refresh_phase_mode,
        "refresh_phase_seed": refresh_phase_seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch")
        if isinstance(checkpoint, dict)
        else None,
        "strict_load": True,
    }
