#!/usr/bin/env python3
"""Measure the image-encoder share of one SAM2 tracking stream."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
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


class ComponentEvents:
    def __init__(self, predictor) -> None:
        self.predictor = predictor
        self.original_get = predictor._get_image_feature
        self.original_encoder = predictor.image_encoder.forward
        self.active_key: tuple[int, int] | None = None
        self.repeat = 0
        self.feature_events: dict[tuple[int, int], list[tuple[Any, Any]]] = {}
        self.encoder_events: dict[tuple[int, int], list[tuple[Any, Any]]] = {}

    @staticmethod
    def _events() -> tuple[Any, Any]:
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    @staticmethod
    def elapsed(events: list[tuple[Any, Any]]) -> float:
        return sum(float(start.elapsed_time(end)) for start, end in events)

    def install(self) -> None:
        def image_encoder_forward(*args, **kwargs):
            start, end = self._events()
            start.record()
            output = self.original_encoder(*args, **kwargs)
            end.record()
            if self.active_key is not None:
                self.encoder_events.setdefault(self.active_key, []).append((start, end))
            return output

        def get_image_feature(inference_state, frame_idx: int, batch_size: int):
            key = (self.repeat, int(frame_idx))
            self.active_key = key
            start, end = self._events()
            start.record()
            try:
                return self.original_get(inference_state, frame_idx, batch_size)
            finally:
                end.record()
                self.feature_events.setdefault(key, []).append((start, end))
                self.active_key = None

        self.predictor.image_encoder.forward = image_encoder_forward
        self.predictor._get_image_feature = get_image_feature

    def uninstall(self) -> None:
        self.predictor._get_image_feature = self.original_get
        self.predictor.image_encoder.forward = self.original_encoder


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.uint8) > 0


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    full = [row["tracking_step_ms"] for row in rows]
    encoder = [row["image_encoder_ms"] for row in rows]
    feature = [row["image_feature_path_ms"] for row in rows]
    full_sum = sum(full)
    return {
        "measured_frames": len(rows),
        "tracking_step_mean_ms": statistics.mean(full),
        "tracking_step_median_ms": statistics.median(full),
        "tracking_step_p90_ms": percentile(full, 90),
        "image_encoder_mean_ms": statistics.mean(encoder),
        "image_encoder_median_ms": statistics.median(encoder),
        "image_feature_path_mean_ms": statistics.mean(feature),
        "image_feature_path_median_ms": statistics.median(feature),
        "image_encoder_share_of_tracking": sum(encoder) / full_sum,
        "image_feature_path_share_of_tracking": sum(feature) / full_sum,
        "non_image_tracking_mean_ms": (full_sum - sum(encoder)) / len(rows),
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
            timer.repeat = repetition
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
            while True:
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                try:
                    frame_idx, object_ids, masks = next(iterator)
                except StopIteration:
                    break
                torch.cuda.synchronize(device)
                tracking_ms = (time.perf_counter() - started) * 1000.0
                frame_idx = int(frame_idx)
                if frame_idx == 0:
                    continue
                seen_nonprompt += 1
                if seen_nonprompt <= args.warmup_frames:
                    continue
                key = (repetition, frame_idx)
                encoder_events = timer.encoder_events.get(key, [])
                feature_events = timer.feature_events.get(key, [])
                if not encoder_events or not feature_events:
                    raise RuntimeError(f"Missing component timing events for frame {frame_idx}")
                rows.append(
                    {
                        "repetition": repetition,
                        "frame_idx": frame_idx,
                        "object_count": len(object_ids),
                        "tracking_step_ms": tracking_ms,
                        "image_encoder_ms": timer.elapsed(encoder_events),
                        "image_feature_path_ms": timer.elapsed(feature_events),
                        "mask_outputs": int(masks.shape[0]),
                    }
                )
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
        **aggregate,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
