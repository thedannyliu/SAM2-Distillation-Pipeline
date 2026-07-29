#!/usr/bin/env python3
"""Measure SAM2 shared-session latency as the number of prompted objects grows."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.data.audit_vos_object_density import shared_frame_prompts


TARGET_RELATIVE_LATENCY = {2: 1.10, 4: 1.25, 8: 1.60, 16: 2.20}
MIN_BINARY_MASK_AGREEMENT = 0.9999
MIN_PER_MASK_IOU = 0.999


def parse_object_counts(value: str) -> list[int]:
    counts = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not counts or counts[0] != 1 or any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError(
            "object counts must be positive and include the one-object baseline"
        )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-kind",
        choices=("edgetam-trainer", "sam2", "stage1-student"),
        default="stage1-student",
    )
    parser.add_argument("--prompt-kind", choices=("box", "point"), default="box")
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--sam2-cfg", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sam2-checkpoint", type=Path)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument(
        "--student-model-name",
        default="tiny_vit_21m_512.dist_in22k_ft_in1k",
    )
    parser.add_argument(
        "--student-family",
        choices=("tinyvit", "repvit"),
        default="tinyvit",
    )
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--ann-root", required=True, type=Path)
    parser.add_argument("--video-list-file", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--object-counts", type=parse_object_counts, default=[1, 2, 4, 8])
    parser.add_argument("--max-videos", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--warmup-videos", type=int, default=1)
    parser.add_argument("--seed", type=int, default=310107256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--execution-mode",
        choices=("legacy", "bucket"),
        default="legacy",
    )
    parser.add_argument("--bucket-size", type=int, default=4)
    parser.add_argument("--bucket-min-objects", type=int, default=4)
    parser.add_argument("--verify-bucket-frames", type=int, default=0)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def read_videos(path: Path, max_videos: int) -> list[str]:
    videos = [
        Path(line.strip()).stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return videos[:max_videos] if max_videos > 0 else videos


def add_prompts(
    predictor,
    evaluator,
    args: argparse.Namespace,
    state: dict[str, Any],
    object_count: int,
    prompt_frame: int,
    prompt_records: list[tuple[str, Path]],
) -> None:
    for object_id, mask_path in prompt_records[:object_count]:
        mask = evaluator.load_mask(mask_path)
        if args.prompt_kind == "box":
            predictor.add_new_points_or_box(
                state,
                frame_idx=prompt_frame,
                obj_id=object_id,
                box=evaluator.mask_bbox(mask),
            )
        else:
            points, labels = evaluator.mask_point(mask)
            predictor.add_new_points_or_box(
                state,
                frame_idx=prompt_frame,
                obj_id=object_id,
                points=points,
                labels=labels,
            )


def measure_video(
    predictor,
    evaluator,
    args: argparse.Namespace,
    device: torch.device,
    video: str,
    object_count: int,
    prompt_frame: int,
    prompt_records: list[tuple[str, Path]],
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    total_start = time.perf_counter()

    init_start = time.perf_counter()
    state = predictor.init_state(video_path=str(args.image_root / video))
    synchronize(device)
    init_sec = time.perf_counter() - init_start

    prompt_start = time.perf_counter()
    add_prompts(
        predictor,
        evaluator,
        args,
        state,
        object_count,
        prompt_frame,
        prompt_records,
    )
    synchronize(device)
    prompt_sec = time.perf_counter() - prompt_start

    propagation_start = time.perf_counter()
    propagated_frames = 0
    object_mask_outputs = 0
    for _, object_ids, video_res_masks in predictor.propagate_in_video(state):
        propagated_frames += 1
        object_mask_outputs += len(object_ids)
        _ = video_res_masks.shape
    synchronize(device)
    propagation_sec = time.perf_counter() - propagation_start
    total_sec = time.perf_counter() - total_start
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )

    if hasattr(predictor, "reset_state"):
        predictor.reset_state(state)
    del state
    gc.collect()
    return {
        "video": video,
        "object_count": object_count,
        "prompt_frame": prompt_frame,
        "propagated_frames": propagated_frames,
        "object_mask_outputs": object_mask_outputs,
        "init_sec": init_sec,
        "prompt_sec": prompt_sec,
        "propagation_sec": propagation_sec,
        "total_sec": total_sec,
        "propagation_ms_per_frame": 1000.0
        * propagation_sec
        / max(propagated_frames, 1),
        "propagation_fps": propagated_frames / max(propagation_sec, 1e-12),
        "end_to_end_fps": propagated_frames / max(total_sec, 1e-12),
        "object_masks_per_sec": object_mask_outputs / max(propagation_sec, 1e-12),
        "peak_memory_mb": peak_memory_mb,
    }


def binary_mask_metrics(
    reference_masks: torch.Tensor,
    candidate_masks: torch.Tensor,
) -> dict[str, Any]:
    if reference_masks.shape != candidate_masks.shape:
        raise ValueError(
            f"mask shapes differ: {reference_masks.shape} != {candidate_masks.shape}"
        )
    reference_flat = reference_masks.reshape(reference_masks.shape[0], -1)
    candidate_flat = candidate_masks.reshape(candidate_masks.shape[0], -1)
    mismatch = torch.count_nonzero(reference_flat != candidate_flat, dim=1)
    intersection = torch.count_nonzero(reference_flat & candidate_flat, dim=1)
    union = torch.count_nonzero(reference_flat | candidate_flat, dim=1)
    iou = torch.where(
        union > 0,
        intersection.float() / union.float(),
        torch.ones_like(union, dtype=torch.float32),
    )
    return {
        "mismatched_pixels": int(mismatch.sum()),
        "total_pixels": reference_masks.numel(),
        "mask_ious": [float(value) for value in iou],
        "mismatch_fractions": [
            float(value) for value in mismatch.float() / reference_flat.shape[1]
        ],
    }


def verify_bucket_equivalence(
    base_predictor,
    bucket_predictor,
    evaluator,
    args: argparse.Namespace,
    video: str,
    object_count: int,
    prompt_frame: int,
    prompt_records: list[tuple[str, Path]],
) -> dict[str, Any]:
    def run(predictor) -> list[tuple[int, list[Any], torch.Tensor]]:
        state = predictor.init_state(video_path=str(args.image_root / video))
        add_prompts(
            predictor,
            evaluator,
            args,
            state,
            object_count,
            prompt_frame,
            prompt_records,
        )
        outputs = []
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            outputs.append(
                (
                    int(frame_idx),
                    list(object_ids),
                    (masks.detach().cpu() > 0),
                )
            )
            if len(outputs) >= args.verify_bucket_frames:
                break
        if hasattr(predictor, "reset_state"):
            predictor.reset_state(state)
        del state
        gc.collect()
        return outputs

    legacy_outputs = run(base_predictor)
    bucket_outputs = run(bucket_predictor)
    mismatched_pixels = 0
    total_pixels = 0
    mask_ious = []
    mismatch_fractions = []
    metadata_match = len(legacy_outputs) == len(bucket_outputs)
    for legacy, bucket in zip(legacy_outputs, bucket_outputs):
        legacy_frame, legacy_ids, legacy_masks = legacy
        bucket_frame, bucket_ids, bucket_masks = bucket
        metadata_match = metadata_match and (
            legacy_frame == bucket_frame
            and legacy_ids == bucket_ids
            and legacy_masks.shape == bucket_masks.shape
        )
        if legacy_masks.shape == bucket_masks.shape:
            metrics = binary_mask_metrics(legacy_masks, bucket_masks)
            mismatched_pixels += metrics["mismatched_pixels"]
            total_pixels += metrics["total_pixels"]
            mask_ious.extend(metrics["mask_ious"])
            mismatch_fractions.extend(metrics["mismatch_fractions"])
    binary_mask_agreement = (
        1.0 - mismatched_pixels / total_pixels if total_pixels else 0.0
    )
    min_mask_iou = min(mask_ious, default=0.0)
    tolerance_pass = (
        binary_mask_agreement >= MIN_BINARY_MASK_AGREEMENT
        and min_mask_iou >= MIN_PER_MASK_IOU
    )
    return {
        "video": video,
        "object_count": object_count,
        "frames_compared": min(len(legacy_outputs), len(bucket_outputs)),
        "metadata_match": metadata_match,
        "total_pixels": total_pixels,
        "mismatched_pixels": mismatched_pixels,
        "binary_mask_agreement": binary_mask_agreement,
        "min_binary_mask_agreement": MIN_BINARY_MASK_AGREEMENT,
        "min_mask_iou": min_mask_iou,
        "mean_mask_iou": statistics.mean(mask_ious) if mask_ious else 0.0,
        "required_min_mask_iou": MIN_PER_MASK_IOU,
        "max_mismatch_fraction_per_mask": max(mismatch_fractions, default=0.0),
        "exact_match": mismatched_pixels == 0,
        "pass": metadata_match and total_pixels > 0 and tolerance_pass,
    }


def aggregate_rows(rows: list[dict[str, Any]], object_counts: list[int]) -> list[dict[str, Any]]:
    aggregate = []
    baseline_ms = {
        (str(row["video"]), int(row.get("repetition", 0))): float(
            row["propagation_ms_per_frame"]
        )
        for row in rows
        if int(row["object_count"]) == 1
    }
    for object_count in object_counts:
        selected = [
            row for row in rows if int(row["object_count"]) == object_count
        ]
        frame_ms = [float(row["propagation_ms_per_frame"]) for row in selected]
        propagation_fps = [float(row["propagation_fps"]) for row in selected]
        relative_latencies = [
            float(row["propagation_ms_per_frame"])
            / baseline_ms[(str(row["video"]), int(row.get("repetition", 0)))]
            for row in selected
        ]
        relative_latency = statistics.median(relative_latencies)
        target = TARGET_RELATIVE_LATENCY.get(object_count)
        aggregate.append(
            {
                "object_count": object_count,
                "samples": len(selected),
                "videos": len({str(row["video"]) for row in selected}),
                "median_propagation_ms_per_frame": statistics.median(frame_ms),
                "p90_propagation_ms_per_frame": float(np.percentile(frame_ms, 90)),
                "median_propagation_fps": statistics.median(propagation_fps),
                "median_end_to_end_fps": statistics.median(
                    float(row["end_to_end_fps"]) for row in selected
                ),
                "median_prompt_sec": statistics.median(
                    float(row["prompt_sec"]) for row in selected
                ),
                "median_peak_memory_mb": statistics.median(
                    float(row["peak_memory_mb"]) for row in selected
                ),
                "relative_latency_vs_1": relative_latency,
                "relative_fps_vs_1": 1.0 / relative_latency,
                "target_relative_latency": target if target is not None else "",
                "target_pass": (
                    int(relative_latency <= target) if target is not None else ""
                ),
            }
        )
    return aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def init_wandb(args: argparse.Namespace, config: dict[str, Any]):
    if args.wandb_mode == "disabled" or not args.wandb_project:
        return None
    import wandb

    wandb_dir = args.out_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run_file = wandb_dir / "wandb_run.json"
    run_id = None
    if run_file.is_file():
        run_id = json.loads(run_file.read_text(encoding="utf-8")).get("run_id")
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or args.out_dir.name,
        id=run_id,
        resume="allow" if run_id else None,
        dir=str(wandb_dir),
        mode=args.wandb_mode,
        config=config,
        job_type="multiobject-latency-benchmark",
    )
    run_file.write_text(
        json.dumps({"run_id": run.id, "url": run.url}, indent=2) + "\n",
        encoding="utf-8",
    )
    return run


def main() -> None:
    args = parse_args()
    required = [
        args.sam2_root,
        args.checkpoint,
        args.image_root,
        args.ann_root,
        args.video_list_file,
    ]
    if args.model_kind == "stage1-student":
        required.extend([args.sam2_checkpoint, args.student_checkpoint])
    for path in required:
        if path is None or not path.exists():
            raise FileNotFoundError(path)
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if args.warmup_videos < 0:
        raise SystemExit("--warmup-videos cannot be negative")
    if args.bucket_size < 1:
        raise SystemExit("--bucket-size must be positive")
    if args.bucket_min_objects < 1:
        raise SystemExit("--bucket-min-objects must be positive")
    if args.verify_bucket_frames < 0:
        raise SystemExit("--verify-bucket-frames cannot be negative")

    from tools.eval import run_sam2_vos_prompt_dataset as evaluator

    evaluator.add_import_roots(args.sam2_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    model_args = SimpleNamespace(
        model_kind=args.model_kind,
        sam2_root=args.sam2_root,
        sam2_cfg=args.sam2_cfg,
        checkpoint=args.checkpoint,
        sam2_checkpoint=args.sam2_checkpoint,
        student_checkpoint=args.student_checkpoint,
        student_model_name=args.student_model_name,
        student_family=args.student_family,
    )
    base_predictor, load_summary = evaluator.build_predictor(model_args, device)
    predictor = base_predictor
    if args.execution_mode == "bucket":
        from sam2_distill.models.sam2_object_buckets import SAM2ObjectBucketAdapter

        predictor = SAM2ObjectBucketAdapter(
            base_predictor,
            args.bucket_size,
            min_bucket_objects=args.bucket_min_objects,
        )
    videos = read_videos(args.video_list_file, args.max_videos)
    if not videos:
        raise RuntimeError(f"Empty benchmark cohort: {args.video_list_file}")
    max_objects = max(args.object_counts)
    prepared = {}
    for video in videos:
        shared = shared_frame_prompts(args.ann_root / video, max_objects)
        if shared is None:
            raise RuntimeError(
                f"{video} lacks {max_objects} non-empty object masks on one frame"
            )
        prepared[video] = shared

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "git_commit": git_commit(),
        "model_kind": args.model_kind,
        "checkpoint": str(args.checkpoint),
        "sam2_checkpoint": str(args.sam2_checkpoint)
        if args.sam2_checkpoint is not None
        else None,
        "object_counts": args.object_counts,
        "videos": videos,
        "repetitions": args.repetitions,
        "prompt_kind": args.prompt_kind,
        "execution_mode": args.execution_mode,
        "bucket_size": args.bucket_size if args.execution_mode == "bucket" else None,
        "bucket_min_objects": (
            args.bucket_min_objects if args.execution_mode == "bucket" else None
        ),
        "bucket_implementation": (
            predictor.implementation_name if args.execution_mode == "bucket" else None
        ),
        "verify_bucket_frames": args.verify_bucket_frames,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else str(device),
    }
    run = init_wandb(args, config)

    verification = None
    with torch.inference_mode(), evaluator.autocast_context(device):
        if args.execution_mode == "bucket" and args.verify_bucket_frames > 0:
            video = videos[0]
            frame_idx, prompts = prepared[video]
            verification = verify_bucket_equivalence(
                base_predictor,
                predictor,
                evaluator,
                args,
                video,
                max_objects,
                frame_idx,
                prompts,
            )
            print(json.dumps({"bucket_verification": verification}), flush=True)

        for video in videos[: args.warmup_videos]:
            frame_idx, prompts = prepared[video]
            measure_video(
                predictor,
                evaluator,
                args,
                device,
                video,
                max_objects,
                frame_idx,
                prompts,
            )

        schedule = [
            (repetition, object_count)
            for repetition in range(args.repetitions)
            for object_count in args.object_counts
        ]
        random.Random(args.seed).shuffle(schedule)
        rows = []
        for repetition, object_count in schedule:
            for video in videos:
                frame_idx, prompts = prepared[video]
                row = measure_video(
                    predictor,
                    evaluator,
                    args,
                    device,
                    video,
                    object_count,
                    frame_idx,
                    prompts,
                )
                row["repetition"] = repetition
                rows.append(row)
                print(json.dumps(row), flush=True)

    aggregate = aggregate_rows(rows, args.object_counts)
    write_csv(args.out_dir / "per_video.csv", rows)
    write_csv(args.out_dir / "aggregate.csv", aggregate)
    gate_rows = [
        row for row in aggregate if row["target_relative_latency"] != ""
    ]
    summary = {
        "status": "complete",
        **config,
        "load": load_summary,
        "warmup_videos": args.warmup_videos,
        "aggregate": aggregate,
        "bucket_verification": verification,
        "bucket_execution_stats": getattr(
            predictor, "execution_stats", None
        ),
        "latency_target_pass": bool(gate_rows)
        and all(int(row["target_pass"]) == 1 for row in gate_rows),
        "wandb_run_id": run.id if run is not None else None,
        "wandb_url": run.url if run is not None else None,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    if run is not None:
        for row in aggregate:
            run.log(
                {
                    "object_count": row["object_count"],
                    "latency/median_ms_per_frame": row[
                        "median_propagation_ms_per_frame"
                    ],
                    "latency/p90_ms_per_frame": row[
                        "p90_propagation_ms_per_frame"
                    ],
                    "throughput/median_fps": row["median_propagation_fps"],
                    "latency/relative_vs_1": row["relative_latency_vs_1"],
                    "memory/median_peak_mb": row["median_peak_memory_mb"],
                }
            )
        run.summary["latency_target_pass"] = summary["latency_target_pass"]
        if verification is not None:
            run.summary["bucket_equivalence_pass"] = verification["pass"]
            run.summary["bucket_binary_mask_agreement"] = verification[
                "binary_mask_agreement"
            ]
            run.summary["bucket_min_mask_iou"] = verification["min_mask_iou"]
            run.summary["bucket_exact_match"] = verification["exact_match"]
        run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
