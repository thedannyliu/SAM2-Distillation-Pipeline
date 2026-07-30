"""Batch synchronized SAM2 object tracks into capacity-bounded buckets."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch


_OUTPUT_KEYS = ("cond_frame_outputs", "non_cond_frame_outputs")


def _merge_values(values: list[Any], path: str) -> Any:
    first = values[0]
    if first is None:
        if any(value is not None for value in values[1:]):
            raise RuntimeError(f"Mixed None and tensor values at {path}")
        return None
    if isinstance(first, torch.Tensor):
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise RuntimeError(f"Mixed output value types at {path}")
        return torch.cat(values, dim=0)
    if isinstance(first, list):
        if not all(isinstance(value, list) and len(value) == len(first) for value in values):
            raise RuntimeError(f"Mismatched output lists at {path}")
        return [
            _merge_values([value[index] for value in values], f"{path}[{index}]")
            for index in range(len(first))
        ]
    if isinstance(first, tuple):
        if not all(
            isinstance(value, tuple) and len(value) == len(first) for value in values
        ):
            raise RuntimeError(f"Mismatched output tuples at {path}")
        return tuple(
            _merge_values([value[index] for value in values], f"{path}[{index}]")
            for index in range(len(first))
        )
    if not all(value == first for value in values[1:]):
        raise RuntimeError(f"Mismatched scalar output values at {path}")
    return first


def _slice_value(value: Any, index: int) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value[index : index + 1]
    if isinstance(value, list):
        return [_slice_value(item, index) for item in value]
    if isinstance(value, tuple):
        return tuple(_slice_value(item, index) for item in value)
    return value


def merge_object_output_dicts(output_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge synchronized per-object SAM2 histories along the batch dimension."""
    if not output_dicts:
        raise ValueError("At least one object output dictionary is required")
    merged: dict[str, Any] = {}
    for output_key in _OUTPUT_KEYS:
        frame_sets = [set(output[output_key]) for output in output_dicts]
        if any(frame_set != frame_sets[0] for frame_set in frame_sets[1:]):
            raise RuntimeError(
                "Bucketed SAM2 currently requires synchronized object histories; "
                f"{output_key} frame sets differ: {frame_sets}"
            )
        merged_frames = {}
        for frame_idx in sorted(frame_sets[0]):
            frame_outputs = [output[output_key][frame_idx] for output in output_dicts]
            keys = set(frame_outputs[0])
            if any(set(output) != keys for output in frame_outputs[1:]):
                raise RuntimeError(
                    f"Object output keys differ at {output_key}[{frame_idx}]"
                )
            merged_frames[frame_idx] = {
                key: _merge_values(
                    [output[key] for output in frame_outputs],
                    f"{output_key}[{frame_idx}].{key}",
                )
                for key in keys
            }
        merged[output_key] = merged_frames
    return merged


def split_bucket_output(output: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
    """Split one compact SAM2 frame output back into per-object views."""
    return [
        {key: _slice_value(value, index) for key, value in output.items()}
        for index in range(batch_size)
    ]


class SAM2ObjectBucketAdapter:
    """Run SAM2 propagation with persistent capacity-bounded object buckets.

    The wrapped predictor keeps the standard SAM2 inference-state contract. Objects
    must be prompted before propagation and have synchronized frame histories.
    """

    implementation_name = "persistent_history_v2"

    def __init__(
        self,
        predictor: Any,
        bucket_size: int,
        min_bucket_objects: int = 4,
    ) -> None:
        if bucket_size < 1:
            raise ValueError("bucket_size must be positive")
        if min_bucket_objects < 1:
            raise ValueError("min_bucket_objects must be positive")
        required = (
            "propagate_in_video",
            "propagate_in_video_preflight",
            "_run_single_frame_inference",
            "_get_orig_video_res_output",
        )
        missing = [name for name in required if not hasattr(predictor, name)]
        if missing:
            raise TypeError(f"SAM2 predictor lacks bucket API methods: {missing}")
        if getattr(predictor, "non_overlap_masks_for_mem_enc", False):
            raise ValueError(
                "Bucket execution requires non_overlap_masks_for_mem_enc=false "
                "to preserve independent per-object tracker semantics"
            )
        if getattr(predictor, "clear_non_cond_mem_around_input", False):
            raise ValueError(
                "Bucket execution does not yet support "
                "clear_non_cond_mem_around_input=true"
            )
        self.predictor = predictor
        self.bucket_size = bucket_size
        self.min_bucket_objects = min_bucket_objects
        self.execution_stats = {
            "bucket_sessions": 0,
            "legacy_small_sessions": 0,
            "legacy_unsynchronized_sessions": 0,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.predictor, name)

    def _bucket_indices(self, object_count: int) -> Iterator[list[int]]:
        for start in range(0, object_count, self.bucket_size):
            yield list(range(start, min(start + self.bucket_size, object_count)))

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state: dict[str, Any],
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ):
        object_ids = inference_state["obj_ids"]
        object_count = len(object_ids)
        if object_count == 0:
            raise RuntimeError("No objects are registered in the inference state")
        if object_count < self.min_bucket_objects:
            self.execution_stats["legacy_small_sessions"] += 1
            yield from self.predictor.propagate_in_video(
                inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=reverse,
            )
            return

        self.predictor.propagate_in_video_preflight(inference_state)
        output_dicts = inference_state["output_dict_per_obj"]
        for output_key in _OUTPUT_KEYS:
            frame_sets = [
                set(output[output_key]) for output in output_dicts.values()
            ]
            if any(
                frame_set != frame_sets[0]
                for frame_set in frame_sets[1:]
            ):
                self.execution_stats["legacy_unsynchronized_sessions"] += 1
                yield from self.predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=start_frame_idx,
                    max_frame_num_to_track=max_frame_num_to_track,
                    reverse=reverse,
                )
                return
        self.execution_stats["bucket_sessions"] += 1
        buckets = []
        for object_indices in self._bucket_indices(object_count):
            bucket_outputs = [output_dicts[index] for index in object_indices]
            buckets.append(
                (
                    object_indices,
                    bucket_outputs,
                    merge_object_output_dicts(bucket_outputs),
                )
            )
        if start_frame_idx is None:
            start_frame_idx = min(
                frame_idx
                for output_dict in output_dicts.values()
                for frame_idx in output_dict["cond_frame_outputs"]
            )

        num_frames = inference_state["num_frames"]
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            processing_order = (
                range(start_frame_idx, end_frame_idx - 1, -1)
                if start_frame_idx > 0
                else []
            )
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in processing_order:
            pred_masks_per_obj: list[torch.Tensor | None] = [None] * object_count
            for object_indices, bucket_outputs, bucket_history in buckets:
                if frame_idx in bucket_history["cond_frame_outputs"]:
                    current_out = bucket_history["cond_frame_outputs"][frame_idx]
                    pred_masks = current_out["pred_masks"].to(
                        inference_state["device"], non_blocking=True
                    )
                    for local_idx, object_idx in enumerate(object_indices):
                        pred_masks_per_obj[object_idx] = pred_masks[
                            local_idx : local_idx + 1
                        ]
                elif frame_idx in bucket_history["non_cond_frame_outputs"]:
                    current_out = bucket_history["non_cond_frame_outputs"][frame_idx]
                    pred_masks = current_out["pred_masks"].to(
                        inference_state["device"], non_blocking=True
                    )
                    for local_idx, object_idx in enumerate(object_indices):
                        pred_masks_per_obj[object_idx] = pred_masks[
                            local_idx : local_idx + 1
                        ]
                else:
                    current_out, pred_masks = (
                        self.predictor._run_single_frame_inference(
                            inference_state=inference_state,
                            output_dict=bucket_history,
                            frame_idx=frame_idx,
                            batch_size=len(object_indices),
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=reverse,
                            run_mem_encoder=True,
                        )
                    )
                    bucket_history["non_cond_frame_outputs"][frame_idx] = current_out
                    split_outputs = split_bucket_output(
                        current_out, len(object_indices)
                    )
                    for local_idx, object_idx in enumerate(object_indices):
                        bucket_outputs[local_idx]["non_cond_frame_outputs"][
                            frame_idx
                        ] = split_outputs[local_idx]
                        pred_masks_per_obj[object_idx] = pred_masks[
                            local_idx : local_idx + 1
                        ]

                frames_tracked_per_obj = inference_state.get(
                    "frames_tracked_per_obj"
                )
                if frames_tracked_per_obj is not None:
                    for object_idx in object_indices:
                        frames_tracked_per_obj[object_idx][frame_idx] = {
                            "reverse": reverse
                        }

            if any(mask is None for mask in pred_masks_per_obj):
                raise RuntimeError(f"Missing bucket output on frame {frame_idx}")
            all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            _, video_res_masks = self.predictor._get_orig_video_res_output(
                inference_state, all_pred_masks
            )
            yield frame_idx, object_ids, video_res_masks
