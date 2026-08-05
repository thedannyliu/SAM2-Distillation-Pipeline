"""SAM2 video predictor with the learned object-slot propagation head."""

from typing import Any

import torch
import torch.nn.functional as F

from sam2.sam2_video_predictor import SAM2VideoPredictor

from sam2_distill.models.sam2_object_slots import (
    NO_OBJ_SCORE,
    ObjectSlotModelMixin,
    PersistentMultiplexLayout,
)


class SAM2ObjectSlotVideoPredictor(
    ObjectSlotModelMixin,
    SAM2VideoPredictor,
):
    def __init__(
        self,
        *args,
        object_slot_count: int = 0,
        object_slot_min_objects: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._init_object_slots(
            object_slot_count=object_slot_count,
            object_slot_min_objects=object_slot_min_objects,
        )
        self._multiplex_runtime_layout: PersistentMultiplexLayout | None = None

    def _new_multiplex_layout(self) -> PersistentMultiplexLayout | None:
        if self.object_slot_decoder is None:
            return None
        return PersistentMultiplexLayout(
            self.object_slot_decoder.slot_count
        )

    def _sync_multiplex_state(self, inference_state) -> None:
        if self.object_slot_decoder is None:
            return
        layout = inference_state.get("sam2_tv_multiplex_layout")
        if not isinstance(layout, PersistentMultiplexLayout):
            layout = self._new_multiplex_layout()
            inference_state["sam2_tv_multiplex_layout"] = layout
        layout.update(list(inference_state["obj_ids"]))
        self._multiplex_runtime_layout = layout
        self._set_multiplex_layout(layout)

    def init_state(self, *args, **kwargs):
        inference_state = super().init_state(*args, **kwargs)
        inference_state["sam2_tv_multiplex_layout"] = (
            self._new_multiplex_layout()
        )
        inference_state["sam2_tv_conditioning_memory_cache"] = {}
        self._sync_multiplex_state(inference_state)
        return inference_state

    def _obj_id_to_idx(self, inference_state, obj_id):
        obj_idx = super()._obj_id_to_idx(inference_state, obj_id)
        self._sync_multiplex_state(inference_state)
        return obj_idx

    def reset_state(self, inference_state):
        result = super().reset_state(inference_state)
        inference_state["sam2_tv_multiplex_layout"] = (
            self._new_multiplex_layout()
        )
        inference_state["sam2_tv_conditioning_memory_cache"] = {}
        self._sync_multiplex_state(inference_state)
        return result

    def remove_object(self, inference_state, obj_id, *args, **kwargs):
        result = super().remove_object(
            inference_state, obj_id, *args, **kwargs
        )
        self._sync_multiplex_state(inference_state)
        return result

    def _activate_multiplex_object_ids(self, object_ids: list[Any]) -> None:
        if self._multiplex_runtime_layout is None:
            return
        self._set_multiplex_layout(
            self._multiplex_runtime_layout.select(object_ids)
        )

    def _multiplex_bucket_indices(
        self,
        object_ids: list[Any],
    ) -> list[list[int]]:
        if self._multiplex_runtime_layout is None:
            return []
        return self._multiplex_runtime_layout.select(
            object_ids
        ).bucket_indices()

    def _multiplex_missing_output(
        self,
        template: dict[str, Any],
    ) -> dict[str, Any]:
        output = {}
        for key, value in template.items():
            if key == "maskmem_pos_enc":
                output[key] = [item.clone() for item in value]
            elif key == "maskmem_features":
                output[key] = value.clone()
            elif key == "pred_masks":
                output[key] = torch.full_like(value, NO_OBJ_SCORE)
            elif key == "obj_ptr":
                no_object = getattr(self, "no_obj_ptr", None)
                if (
                    isinstance(no_object, torch.Tensor)
                    and no_object.numel() == value.shape[-1]
                ):
                    output[key] = no_object.to(
                        device=value.device,
                        dtype=value.dtype,
                    ).reshape(1, -1).expand_as(value)
                else:
                    output[key] = torch.zeros_like(value)
            elif key == "object_score_logits":
                output[key] = torch.full_like(value, NO_OBJ_SCORE)
            else:
                raise KeyError(f"Unsupported compact output key: {key}")
        return output

    def _reencode_multiplex_conditioning_memory(
        self,
        inference_state,
    ) -> None:
        object_ids = list(inference_state["obj_ids"])
        if len(object_ids) < self.object_slot_decoder.min_objects:
            return
        output_dicts = inference_state["output_dict_per_obj"]
        reencoded = inference_state.setdefault(
            "sam2_tv_conditioning_memory_cache", {}
        )
        conditioning_frames = sorted(
            set().union(
                *(
                    set(output["cond_frame_outputs"])
                    for output in output_dicts.values()
                )
            )
        )
        for frame_idx in conditioning_frames:
            for object_indices in self._multiplex_bucket_indices(object_ids):
                if not any(
                    frame_idx
                    in output_dicts[index]["cond_frame_outputs"]
                    for index in object_indices
                ):
                    continue
                frame_outputs = []
                conditioning = []
                for object_idx in object_indices:
                    object_output = output_dicts[object_idx]
                    current_out = object_output["cond_frame_outputs"].get(
                        frame_idx
                    )
                    is_conditioning = current_out is not None
                    if current_out is None:
                        current_out = object_output[
                            "non_cond_frame_outputs"
                        ].get(frame_idx)
                    frame_outputs.append(current_out)
                    conditioning.append(is_conditioning)

                active_ids = tuple(
                    object_ids[index] for index in object_indices
                )
                cache_key = (frame_idx, active_ids)
                signature = (
                    tuple(
                        id(output["pred_masks"])
                        if output is not None
                        else None
                        for output in frame_outputs
                    ),
                    tuple(conditioning),
                )
                if reencoded.get(cache_key) == signature:
                    continue

                template = next(
                    output for output in frame_outputs if output is not None
                )
                pred_masks = []
                object_scores = []
                for output in frame_outputs:
                    if output is None:
                        pred_masks.append(
                            torch.full_like(
                                template["pred_masks"], NO_OBJ_SCORE
                            )
                        )
                        object_scores.append(
                            torch.full_like(
                                template["object_score_logits"],
                                NO_OBJ_SCORE,
                            )
                        )
                    else:
                        pred_masks.append(output["pred_masks"])
                        object_scores.append(output["object_score_logits"])

                self._activate_multiplex_object_ids(list(active_ids))
                device = inference_state["device"]
                high_res_masks = F.interpolate(
                    torch.cat(pred_masks).to(device, non_blocking=True),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                object_score_logits = torch.cat(object_scores).to(
                    device, non_blocking=True
                )
                self._multiplex_conditioning_override = torch.tensor(
                    conditioning,
                    dtype=high_res_masks.dtype,
                    device=device,
                )
                try:
                    maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        batch_size=len(object_indices),
                        high_res_masks=high_res_masks,
                        object_score_logits=object_score_logits,
                        is_mask_from_pts=True,
                    )
                finally:
                    del self._multiplex_conditioning_override

                for local_idx, output in enumerate(frame_outputs):
                    if output is None:
                        continue
                    output["maskmem_features"] = maskmem_features[
                        local_idx : local_idx + 1
                    ]
                    output["maskmem_pos_enc"] = [
                        value[local_idx : local_idx + 1]
                        for value in maskmem_pos_enc
                    ]
                reencoded[cache_key] = signature
    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state):
        result = super().propagate_in_video_preflight(inference_state)
        if self.object_slot_decoder is not None:
            self._sync_multiplex_state(inference_state)
            try:
                self._reencode_multiplex_conditioning_memory(inference_state)
            finally:
                self._activate_multiplex_object_ids(
                    list(inference_state["obj_ids"])
                )
        return result

    def _propagate_in_video_legacy(self, inference_state, **kwargs):
        yield from super().propagate_in_video(inference_state, **kwargs)

    @torch.inference_mode()
    def propagate_in_video(self, inference_state, **kwargs):
        self._sync_multiplex_state(inference_state)
        if self.object_slot_decoder is None:
            yield from self._propagate_in_video_legacy(
                inference_state, **kwargs
            )
            return
        from sam2_distill.models.sam2_object_buckets import (
            SAM2ObjectBucketAdapter,
        )

        adapter = SAM2ObjectBucketAdapter(
            self,
            bucket_size=self.object_slot_decoder.slot_count,
            min_bucket_objects=self.object_slot_decoder.min_objects,
        )
        yield from adapter.propagate_in_video(inference_state, **kwargs)
