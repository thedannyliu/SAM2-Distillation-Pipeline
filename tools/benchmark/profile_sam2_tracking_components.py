#!/usr/bin/env python3
"""Profile semantic components in one official SAM2.1-L tracking stream."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--sam2-cfg", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--ann-root", required=True, type=Path)
    parser.add_argument("--video-list-file", required=True, type=Path)
    parser.add_argument("--video-name")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--warmup-frames", type=int, default=16)
    parser.add_argument("--measure-frames", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def frame_paths(root: Path, video: str) -> list[Path]:
    return sorted(
        (
            path
            for path in (root / video).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg"}
        ),
        key=lambda path: int(path.stem),
    )


def prompted_objects(
    ann_root: Path, video: str, first_stem: str
) -> list[tuple[int, Path]]:
    video_root = ann_root / video
    if not video_root.is_dir():
        return []
    records = []
    for object_root in sorted(path for path in video_root.iterdir() if path.is_dir()):
        path = object_root / f"{first_stem}.png"
        if path.is_file():
            records.append((int(object_root.name), path))
    return records


def select_video(args: argparse.Namespace) -> tuple[str, list[Path], list[tuple[int, Path]]]:
    requested = [args.video_name] if args.video_name else [
        Path(line.strip()).stem
        for line in args.video_list_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    minimum_frames = 1 + args.warmup_frames + args.measure_frames
    for video in requested:
        frames = frame_paths(args.image_root, video)
        if len(frames) < minimum_frames:
            continue
        objects = prompted_objects(args.ann_root, video, frames[0].stem)
        if objects:
            if args.max_objects > 0:
                objects = objects[: args.max_objects]
            return video, frames, objects
    raise RuntimeError(
        f"No selected video has {minimum_frames} frames and first-frame object prompts"
    )


ADDITIVE_COMPONENTS = (
    "image_encoder",
    "prompt_encoder",
    "memory_attention",
    "mask_decoder",
    "memory_encoder",
    "object_pointer_projection",
    "object_pointer_temporal_projection",
)

NESTED_COMPONENTS = {
    "image_trunk": "image_encoder",
    "image_neck": "image_encoder",
    "mask_decoder_transformer": "mask_decoder",
    "memory_mask_downsampler": "memory_encoder",
    "memory_pixel_projection": "memory_encoder",
    "memory_fuser": "memory_encoder",
    "memory_output_projection": "memory_encoder",
}


class ComponentEvents:
    def __init__(self, predictor) -> None:
        self.predictor = predictor
        self.original_get = predictor._get_image_feature
        self.active_key: tuple[int, int] | None = None
        self.events: dict[str, dict[tuple[int, int], list[tuple[Any, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.handles: list[Any] = []
        self.available_components: list[str] = []
        self.missing_components: list[str] = []

    @staticmethod
    def _events() -> tuple[Any, Any]:
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    @staticmethod
    def elapsed(events: list[tuple[Any, Any]]) -> float:
        return sum(float(start.elapsed_time(end)) for start, end in events)

    def _register(self, name: str, module: Any | None) -> None:
        if module is None:
            self.missing_components.append(name)
            return
        starts: list[tuple[tuple[int, int], Any]] = []

        def before(_module, _inputs):
            if self.active_key is None:
                return
            start, _ = self._events()
            start.record()
            starts.append((self.active_key, start))

        def after(_module, _inputs, output):
            if not starts:
                return output
            key, start = starts.pop()
            _, end = self._events()
            end.record()
            self.events[name][key].append((start, end))
            return output

        self.handles.append(module.register_forward_pre_hook(before))
        self.handles.append(module.register_forward_hook(after))
        self.available_components.append(name)

    def install(self) -> None:
        modules = {
            "image_encoder": self.predictor.image_encoder,
            "prompt_encoder": self.predictor.sam_prompt_encoder,
            "memory_attention": self.predictor.memory_attention,
            "mask_decoder": self.predictor.sam_mask_decoder,
            "memory_encoder": self.predictor.memory_encoder,
            "object_pointer_projection": self.predictor.obj_ptr_proj,
            "object_pointer_temporal_projection": getattr(
                self.predictor, "obj_ptr_tpos_proj", None
            ),
            "image_trunk": getattr(self.predictor.image_encoder, "trunk", None),
            "image_neck": getattr(self.predictor.image_encoder, "neck", None),
            "mask_decoder_transformer": getattr(
                self.predictor.sam_mask_decoder, "transformer", None
            ),
            "memory_mask_downsampler": getattr(
                self.predictor.memory_encoder, "mask_downsampler", None
            ),
            "memory_pixel_projection": getattr(
                self.predictor.memory_encoder, "pix_feat_proj", None
            ),
            "memory_fuser": getattr(self.predictor.memory_encoder, "fuser", None),
            "memory_output_projection": getattr(
                self.predictor.memory_encoder, "out_proj", None
            ),
        }
        for name, module in modules.items():
            self._register(name, module)

        def get_image_feature(*args, **kwargs):
            if self.active_key is None:
                return self.original_get(*args, **kwargs)
            start, end = self._events()
            start.record()
            try:
                return self.original_get(*args, **kwargs)
            finally:
                end.record()
                self.events["image_feature_path"][self.active_key].append((start, end))

        self.predictor._get_image_feature = get_image_feature

    def uninstall(self) -> None:
        self.predictor._get_image_feature = self.original_get
        for handle in self.handles:
            handle.remove()


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.uint8) > 0


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))


def summarize_rows(
    rows: list[dict[str, float]],
    additive_components: tuple[str, ...] = ADDITIVE_COMPONENTS,
    nested_components: dict[str, str] = NESTED_COMPONENTS,
) -> dict[str, Any]:
    full = [row["tracking_step_ms"] for row in rows]
    full_sum = sum(full)
    component_table = []
    additive_sum = 0.0
    for component in additive_components:
        values = [row[f"{component}_ms"] for row in rows]
        component_sum = sum(values)
        additive_sum += component_sum
        component_table.append(
            {
                "component": component,
                "mean_ms": statistics.mean(values),
                "median_ms": statistics.median(values),
                "percent_of_tracking": 100.0 * component_sum / full_sum,
            }
        )
    residual_sum = full_sum - additive_sum
    component_table.append(
        {
            "component": "framework_and_uninstrumented_residual",
            "mean_ms": residual_sum / len(rows),
            "median_ms": statistics.median(
                [row["residual_ms"] for row in rows]
            ),
            "percent_of_tracking": 100.0 * residual_sum / full_sum,
        }
    )
    nested_table = []
    for component, parent in nested_components.items():
        values = [row[f"{component}_ms"] for row in rows]
        parent_sum = sum(row[f"{parent}_ms"] for row in rows)
        component_sum = sum(values)
        nested_table.append(
            {
                "component": component,
                "parent": parent,
                "mean_ms": statistics.mean(values),
                "median_ms": statistics.median(values),
                "percent_of_tracking": 100.0 * component_sum / full_sum,
                "percent_of_parent": (
                    100.0 * component_sum / parent_sum if parent_sum else None
                ),
            }
        )
    feature = [row["image_feature_path_ms"] for row in rows]
    return {
        "measured_frames": len(rows),
        "tracking_step_mean_ms": statistics.mean(full),
        "tracking_step_median_ms": statistics.median(full),
        "tracking_step_p90_ms": percentile(full, 90),
        "component_breakdown": component_table,
        "component_percent_total": sum(
            row["percent_of_tracking"] for row in component_table
        ),
        "nested_component_breakdown": nested_table,
        "image_feature_path_mean_ms": statistics.mean(feature),
        "image_feature_path_median_ms": statistics.median(feature),
        "image_feature_path_percent_of_tracking": 100.0 * sum(feature) / full_sum,
        "frames_with_negative_residual": sum(row["residual_ms"] < 0 for row in rows),
    }


def main() -> None:
    args = parse_args()
    if args.warmup_frames < 0 or args.measure_frames < 1 or args.repetitions < 1:
        raise ValueError("warmup must be non-negative; measurement and repetitions must be positive")
    for path in (
        args.sam2_root,
        args.checkpoint,
        args.image_root,
        args.ann_root,
        args.video_list_file,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if str(args.sam2_root) not in sys.path:
        sys.path.insert(0, str(args.sam2_root))
    from sam2.build_sam import build_sam2_video_predictor

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Component timing requires one CUDA GPU")
    video, frames, objects = select_video(args)
    predictor = build_sam2_video_predictor(
        args.sam2_cfg,
        str(args.checkpoint),
        device=str(device),
        apply_postprocessing=False,
    )
    timer = ComponentEvents(predictor)
    timer.install()
    rows: list[dict[str, float]] = []
    try:
        for repetition in range(args.repetitions):
            state = predictor.init_state(
                str(args.image_root / video), offload_video_to_cpu=True
            )
            for object_id, mask_path in objects:
                predictor.add_new_mask(state, 0, object_id, load_mask(mask_path))
            iterator = iter(
                predictor.propagate_in_video(
                    state,
                    start_frame_idx=0,
                    max_frame_num_to_track=args.warmup_frames + args.measure_frames,
                )
            )
            seen_nonprompt = 0
            expected_frame_idx = 0
            while True:
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                timer.active_key = (repetition, expected_frame_idx)
                try:
                    frame_idx, object_ids, masks = next(iterator)
                except StopIteration:
                    timer.active_key = None
                    break
                torch.cuda.synchronize(device)
                timer.active_key = None
                tracking_ms = (time.perf_counter() - started) * 1000.0
                frame_idx = int(frame_idx)
                if frame_idx != expected_frame_idx:
                    raise RuntimeError(
                        f"Expected sequential frame {expected_frame_idx}, got {frame_idx}"
                    )
                expected_frame_idx += 1
                if frame_idx == 0:
                    continue
                seen_nonprompt += 1
                if seen_nonprompt <= args.warmup_frames:
                    continue
                key = (repetition, frame_idx)
                row = {
                    "repetition": repetition,
                    "frame_idx": frame_idx,
                    "object_count": len(object_ids),
                    "tracking_step_ms": tracking_ms,
                    "mask_outputs": int(masks.shape[0]),
                }
                for component in (*ADDITIVE_COMPONENTS, *NESTED_COMPONENTS):
                    row[f"{component}_ms"] = timer.elapsed(
                        timer.events[component].get(key, [])
                    )
                feature_events = timer.events["image_feature_path"].get(key, [])
                if not feature_events or row["image_encoder_ms"] == 0:
                    raise RuntimeError(f"Missing essential timing events for frame {frame_idx}")
                row["image_feature_path_ms"] = timer.elapsed(feature_events)
                row["residual_ms"] = tracking_ms - sum(
                    row[f"{component}_ms"] for component in ADDITIVE_COMPONENTS
                )
                rows.append(row)
                if seen_nonprompt >= args.warmup_frames + args.measure_frames:
                    break
            predictor.reset_state(state)
    finally:
        timer.uninstall()

    expected = args.repetitions * args.measure_frames
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} measured frames, got {len(rows)}")
    aggregate = summarize_rows(rows)
    summary = {
        "status": "pass",
        "measurement_contract": (
            "single-stream synchronized tracking step; excludes model load, video init, "
            "prompt insertion, JPEG decode/preload, and PNG serialization"
        ),
        "percentage_contract": (
            "top-level semantic components plus framework_and_uninstrumented_residual "
            "sum to 100%; nested components are drill-down measurements and are not additive"
        ),
        "video": video,
        "video_frames": len(frames),
        "object_count": len(objects),
        "warmup_frames_per_repeat": args.warmup_frames,
        "measure_frames_per_repeat": args.measure_frames,
        "repetitions": args.repetitions,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "checkpoint": str(args.checkpoint),
        "available_instrumented_components": timer.available_components,
        "missing_optional_components": timer.missing_components,
        **aggregate,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "component_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("component", "mean_ms", "median_ms", "percent_of_tracking"),
        )
        writer.writeheader()
        writer.writerows(aggregate["component_breakdown"])
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
