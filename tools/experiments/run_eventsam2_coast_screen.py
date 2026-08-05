#!/usr/bin/env python3
"""Screen a bottleneck-skipping COAST action with optimistic full-mask resets."""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_ints(value: str) -> list[int]:
    values = sorted({int(field) for field in value.split(",") if field.strip()})
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--full-pred-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--video-list-file", type=Path)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--profile-videos", type=int, default=4)
    parser.add_argument("--sentinel-resolution", type=int, default=256)
    parser.add_argument("--refresh-intervals", type=parse_ints, default=parse_ints("1,2,4,8,16"))
    parser.add_argument("--safe-horizons", type=parse_ints, default=parse_ints("4,8,16"))
    parser.add_argument("--safe-drop-points", type=float, default=1.0)
    parser.add_argument("--full-latency-ms", type=float, required=True)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-id", default="eventsam2-coast-screen-v1")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def read_video_names(image_root: Path, list_file: Path | None, max_videos: int) -> list[str]:
    if list_file is None:
        names = sorted(path.name for path in image_root.iterdir() if path.is_dir())
    else:
        names = [
            line.strip()
            for line in list_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if max_videos > 0:
        names = names[:max_videos]
    missing = [name for name in names if not (image_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing selected image directories: {missing[:10]}")
    if not names:
        raise ValueError("No videos selected")
    return names


@lru_cache(maxsize=2048)
def read_mask(path_string: str) -> np.ndarray:
    mask = cv2.imread(path_string, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path_string)
    return mask > 0


def empty_mask(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=bool)


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum()) / float(union)


def boundary_f(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(np.uint8)
    target = target.astype(np.uint8)
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_boundary = cv2.morphologyEx(prediction, cv2.MORPH_GRADIENT, kernel) > 0
    target_boundary = cv2.morphologyEx(target, cv2.MORPH_GRADIENT, kernel) > 0
    radius = max(1, int(math.ceil(0.008 * math.hypot(*target.shape))))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    tolerance = ((xx * xx + yy * yy) <= radius * radius).astype(np.uint8)
    pred_dilated = cv2.dilate(pred_boundary.astype(np.uint8), tolerance) > 0
    target_dilated = cv2.dilate(target_boundary.astype(np.uint8), tolerance) > 0
    precision = float((pred_boundary & target_dilated).sum()) / max(int(pred_boundary.sum()), 1)
    recall = float((target_boundary & pred_dilated).sum()) / max(int(target_boundary.sum()), 1)
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def score(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    j = mask_iou(prediction, target)
    f = boundary_f(prediction, target)
    return j, f, (j + f) / 2.0


def compute_backward_flows(
    frame_paths: list[Path], resolution: int
) -> tuple[list[np.ndarray], list[float]]:
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    previous = None
    flows = []
    latencies = []
    for path in frame_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        current = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)
        if previous is not None:
            started = time.perf_counter()
            flows.append(dis.calc(current, previous, None).astype(np.float32))
            latencies.append((time.perf_counter() - started) * 1000.0)
        previous = current
    return flows, latencies


def warp_mask(mask: np.ndarray, backward_flow: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    flow = cv2.resize(backward_flow, (width, height), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= width / backward_flow.shape[1]
    flow[..., 1] *= height / backward_flow.shape[0]
    xx, yy = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    warped = cv2.remap(
        mask.astype(np.uint8),
        xx + flow[..., 0],
        yy + flow[..., 1],
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 0


def mask_path(root: Path, video: str, object_name: str, stem: str) -> Path:
    return root / video / object_name / f"{stem}.png"


def load_or_empty(path: Path, shape: tuple[int, int]) -> np.ndarray:
    return read_mask(str(path)) if path.is_file() else empty_mask(*shape)


def video_assets(args: argparse.Namespace, video: str):
    frames = sorted(
        path for path in (args.image_root / video).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not frames:
        raise ValueError(f"No frames for {video}")
    image = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(frames[0])
    gt_video = args.gt_root / video
    objects = sorted(path.name for path in gt_video.iterdir() if path.is_dir())
    if not objects:
        raise ValueError(f"No GT object directories for {video}")
    return frames, objects, image.shape


def append_scores(
    rows: list[dict[str, object]],
    *,
    args: argparse.Namespace,
    video: str,
    objects: list[str],
    stem: str,
    predictions: dict[str, np.ndarray],
    policy: str,
    interval: int,
    phase: int,
) -> None:
    for object_name in objects:
        gt_path = mask_path(args.gt_root, video, object_name, stem)
        if not gt_path.is_file():
            continue
        target = read_mask(str(gt_path))
        candidate = predictions.get(object_name, empty_mask(*target.shape))
        full = load_or_empty(
            mask_path(args.full_pred_root, video, object_name, stem), target.shape
        )
        j, f, jf = score(candidate, target)
        full_j, full_f, full_jf = score(full, target)
        rows.append(
            {
                "video": video,
                "frame_stem": stem,
                "object": object_name,
                "policy": policy,
                "refresh_interval": interval,
                "phase": phase,
                "j": j,
                "f": f,
                "j_and_f": jf,
                "full_j": full_j,
                "full_f": full_f,
                "full_j_and_f": full_jf,
                "regret_points": max(0.0, (full_jf - jf) * 100.0),
            }
        )


def run_fixed_video(args: argparse.Namespace, video: str):
    score_rows: list[dict[str, object]] = []
    system_rows: list[dict[str, object]] = []
    frames, objects, shape = video_assets(args, video)
    flows, flow_latencies = compute_backward_flows(frames, args.sentinel_resolution)
    for interval in args.refresh_intervals:
        phases = range(interval) if interval > 1 else range(1)
        for phase in phases:
            transient: dict[str, np.ndarray] = {}
            for frame_index, frame_path in enumerate(frames):
                refresh = frame_index == 0 or (frame_index - phase) % interval == 0
                started = time.perf_counter()
                if refresh:
                    transient = {
                        object_name: load_or_empty(
                            mask_path(
                                args.full_pred_root, video, object_name, frame_path.stem
                            ),
                            shape,
                        )
                        for object_name in objects
                    }
                    action = "refresh"
                    transition_ms = 0.0
                else:
                    transient = {
                        object_name: warp_mask(mask, flows[frame_index - 1])
                        for object_name, mask in transient.items()
                    }
                    action = "coast"
                    transition_ms = (time.perf_counter() - started) * 1000.0
                system_rows.append(
                    {
                        "video": video,
                        "policy": f"periodic_i{interval}_p{phase}",
                        "refresh_interval": interval,
                        "phase": phase,
                        "frame_index": frame_index,
                        "action": action,
                        "transition_ms": transition_ms,
                    }
                )
                append_scores(
                    score_rows,
                    args=args,
                    video=video,
                    objects=objects,
                    stem=frame_path.stem,
                    predictions=transient,
                    policy=f"periodic_i{interval}_p{phase}",
                    interval=interval,
                    phase=phase,
                )
    return score_rows, system_rows, flow_latencies


def run_fixed_policies(args: argparse.Namespace, videos: list[str]):
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(lambda video: run_fixed_video(args, video), videos))
    score_rows = [row for scores, _, _ in results for row in scores]
    system_rows = [row for _, systems, _ in results for row in systems]
    flow_latencies = [value for _, _, latencies in results for value in latencies]
    return pd.DataFrame(score_rows), pd.DataFrame(system_rows), flow_latencies


def run_safe_horizon_video(args: argparse.Namespace, video: str) -> list[dict[str, object]]:
    rows = []
    frames, objects, shape = video_assets(args, video)
    flows, _ = compute_backward_flows(frames, args.sentinel_resolution)
    annotated = {
        path.stem
        for object_name in objects
        for path in (args.gt_root / video / object_name).glob("*.png")
    }
    for anchor, anchor_path in enumerate(frames[:-1]):
        if anchor_path.stem not in annotated:
            continue
        initial = {
            object_name: load_or_empty(
                mask_path(args.full_pred_root, video, object_name, anchor_path.stem),
                shape,
            )
            for object_name in objects
        }
        for horizon in args.safe_horizons:
            transient = {key: value.copy() for key, value in initial.items()}
            regrets = []
            coast_frames = min(horizon, len(frames) - anchor - 1)
            if coast_frames < horizon:
                continue
            for offset in range(1, coast_frames + 1):
                frame_index = anchor + offset
                transient = {
                    object_name: warp_mask(mask, flows[frame_index - 1])
                    for object_name, mask in transient.items()
                }
                stem = frames[frame_index].stem
                for object_name in objects:
                    gt_path = mask_path(args.gt_root, video, object_name, stem)
                    if not gt_path.is_file():
                        continue
                    target = read_mask(str(gt_path))
                    candidate = transient.get(object_name, empty_mask(*target.shape))
                    full = load_or_empty(
                        mask_path(args.full_pred_root, video, object_name, stem), target.shape
                    )
                    regret = max(0.0, (score(full, target)[2] - score(candidate, target)[2]) * 100.0)
                    regrets.append(regret)
            if regrets:
                maximum = max(regrets)
                rows.append(
                    {
                        "video": video,
                        "anchor_frame": anchor,
                        "anchor_stem": anchor_path.stem,
                        "horizon": horizon,
                        "evaluated_coast_frames": coast_frames,
                        "max_regret_points": maximum,
                        "mean_regret_points": float(np.mean(regrets)),
                        "safe": maximum <= args.safe_drop_points,
                    }
                )
    return rows


def run_safe_horizon(args: argparse.Namespace, videos: list[str]) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(lambda video: run_safe_horizon_video(args, video), videos))
    return pd.DataFrame(row for rows in results for row in rows)


def profile_coast_compute(
    args: argparse.Namespace, video: str
) -> tuple[list[float], list[float]]:
    frames, objects, shape = video_assets(args, video)
    flows, flow_latencies = compute_backward_flows(frames, args.sentinel_resolution)
    transient = {
        object_name: load_or_empty(
            mask_path(args.full_pred_root, video, object_name, frames[0].stem), shape
        )
        for object_name in objects
    }
    transition_latencies = []
    for flow in flows:
        started = time.perf_counter()
        transient = {
            object_name: warp_mask(mask, flow)
            for object_name, mask in transient.items()
        }
        transition_latencies.append((time.perf_counter() - started) * 1000.0)
    return flow_latencies, transition_latencies


def summarize(
    args: argparse.Namespace,
    scores: pd.DataFrame,
    systems: pd.DataFrame,
    flow_latencies: list[float],
    transition_latencies: list[float],
    oracle: pd.DataFrame,
) -> dict[str, object]:
    phase_quality = (
        scores.groupby(["policy", "refresh_interval", "phase"], as_index=False)
        .agg(j=("j", "mean"), f=("f", "mean"), j_and_f=("j_and_f", "mean"), full_j_and_f=("full_j_and_f", "mean"))
    )
    phase_quality["drop_points"] = (
        phase_quality["full_j_and_f"] - phase_quality["j_and_f"]
    ) * 100.0
    policy_quality = (
        phase_quality.groupby("refresh_interval", as_index=False)
        .agg(
            j_and_f=("j_and_f", "mean"),
            mean_drop_points=("drop_points", "mean"),
            worst_phase_drop_points=("drop_points", "max"),
        )
    )
    phase_full_fraction = (
        systems.assign(is_refresh=systems["action"] == "refresh")
        .groupby(["refresh_interval", "phase"], as_index=False)["is_refresh"]
        .mean()
    )
    full_fraction_by_interval = phase_full_fraction.groupby("refresh_interval")[
        "is_refresh"
    ].mean()
    transition_ms = float(np.median(transition_latencies)) if transition_latencies else 0.0
    flow_ms = float(np.median(flow_latencies)) if flow_latencies else 0.0
    coast_ms = flow_ms + transition_ms
    fixed_table = []
    for row in policy_quality.to_dict("records"):
        interval = int(row["refresh_interval"])
        full_fraction = float(full_fraction_by_interval.loc[interval])
        e2e_ms = full_fraction * args.full_latency_ms + (1.0 - full_fraction) * coast_ms
        fixed_table.append(
            {
                **row,
                "full_fraction": full_fraction,
                "estimated_e2e_ms": e2e_ms,
                "estimated_speedup": args.full_latency_ms / e2e_ms,
            }
        )

    oracle_table = []
    if not oracle.empty:
        per_video = (
            oracle.groupby(["horizon", "video"], as_index=False)["safe"].mean()
        )
        for horizon, rows in oracle.groupby("horizon"):
            video_rows = per_video[per_video["horizon"] == horizon]
            ideal_speedup = (
                (horizon + 1) * args.full_latency_ms
                / (args.full_latency_ms + horizon * coast_ms)
            )
            oracle_table.append(
                {
                    "horizon": int(horizon),
                    "anchors": int(len(rows)),
                    "safe_anchor_fraction": float(rows["safe"].mean()),
                    "video_macro_safe_fraction": float(video_rows["safe"].mean()),
                    "video_p10_safe_fraction": float(video_rows["safe"].quantile(0.1)),
                    "ideal_periodic_speedup_if_safe": ideal_speedup,
                    "passes_coverage_gates": bool(
                        rows["safe"].mean() >= 0.60
                        and video_rows["safe"].quantile(0.1) >= 0.40
                    ),
                }
            )

    return {
        "experiment_kind": "optimistic oracle-reset COAST screen; not state-consistent deployment",
        "videos": int(scores["video"].nunique()),
        "safe_drop_points": args.safe_drop_points,
        "latency": {
            "profile_videos": min(args.profile_videos, int(scores["video"].nunique())),
            "full_reference_ms": args.full_latency_ms,
            "sentinel_flow_median_ms": flow_ms,
            "multi_object_transition_median_ms": transition_ms,
            "coast_compute_median_ms": coast_ms,
            "coast_fraction_of_full": coast_ms / args.full_latency_ms,
            "passes_coast_20pct_gate": coast_ms <= 0.20 * args.full_latency_ms,
        },
        "fixed_policy_table": fixed_table,
        "safe_horizon_table": oracle_table,
    }


def log_wandb(args: argparse.Namespace, summary: dict[str, object]) -> None:
    if not args.wandb_project or args.wandb_mode == "disabled":
        return
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        id=args.wandb_run_id,
        name=args.wandb_run_id,
        resume="allow",
        mode=args.wandb_mode,
        dir=str(args.out_dir / "wandb"),
        config={
            "sentinel_resolution": args.sentinel_resolution,
            "refresh_intervals": args.refresh_intervals,
            "safe_horizons": args.safe_horizons,
            "safe_drop_points": args.safe_drop_points,
            "full_latency_ms": args.full_latency_ms,
        },
    )
    run.summary.update(summary)
    run.finish()


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(1)
    for path in (args.image_root, args.gt_root, args.full_pred_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.full_latency_ms <= 0:
        raise ValueError("--full-latency-ms must be positive")
    if args.workers < 1 or args.profile_videos < 1:
        raise ValueError("--workers and --profile-videos must be positive")
    videos = read_video_names(args.image_root, args.video_list_file, args.max_videos)
    profile_flow_latencies = []
    profile_transition_latencies = []
    for video in videos[: args.profile_videos]:
        flow_values, transition_values = profile_coast_compute(args, video)
        profile_flow_latencies.extend(flow_values)
        profile_transition_latencies.extend(transition_values)
    scores, systems, _ = run_fixed_policies(args, videos)
    oracle = run_safe_horizon(args, videos)
    if scores.empty or oracle.empty:
        raise RuntimeError("No annotated COAST comparisons were produced")
    summary = summarize(
        args,
        scores,
        systems,
        profile_flow_latencies,
        profile_transition_latencies,
        oracle,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(args.out_dir / "fixed_policy_object_metrics.parquet", index=False)
    systems.to_parquet(args.out_dir / "fixed_policy_system_metrics.parquet", index=False)
    oracle.to_parquet(args.out_dir / "safe_horizon_oracle.parquet", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    log_wandb(args, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
