#!/usr/bin/env python3
"""Measure adjacent-frame drift in fresh SAM2.1-L image embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
METRIC_FIELDS = (
    "video",
    "video_length",
    "transition_index",
    "previous_frame",
    "current_frame",
    "time_seconds",
    "cosine_similarity",
    "cosine_distance",
    "mse",
    "symmetric_nmse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--sam2-cfg", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--video-list-file", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-videos", type=int, default=20)
    parser.add_argument("--seed", type=int, default=250107256)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "none"), default="bf16")
    return parser.parse_args()


def frame_paths(image_root: Path, video: str) -> list[Path]:
    video_root = image_root / video
    if not video_root.is_dir():
        return []

    def frame_key(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return sys.maxsize, path.name

    return sorted(
        (path for path in video_root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=frame_key,
    )


def read_video_names(path: Path) -> list[str]:
    return [
        Path(line.strip()).stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stratified_video_selection(
    candidates: list[tuple[str, int]], num_videos: int, seed: int
) -> list[dict[str, int | str]]:
    """Select videos evenly from four deterministic frame-count strata."""
    if num_videos <= 0:
        raise ValueError("num_videos must be positive")
    eligible = sorted(
        ((name, length) for name, length in candidates if length >= 2),
        key=lambda item: (item[1], item[0]),
    )
    if len(eligible) < num_videos:
        raise ValueError(f"requested {num_videos} videos, only {len(eligible)} have >=2 frames")

    strata = [list(group) for group in np.array_split(np.asarray(eligible, dtype=object), 4)]
    base, remainder = divmod(num_videos, 4)
    quotas = [base + (index < remainder) for index in range(4)]
    rng = random.Random(seed)
    selected: list[dict[str, int | str]] = []
    for stratum_index, (stratum, quota) in enumerate(zip(strata, quotas), start=1):
        if len(stratum) < quota:
            raise ValueError(f"length stratum {stratum_index} cannot supply {quota} videos")
        choices = rng.sample(range(len(stratum)), quota)
        for index in choices:
            name, length = stratum[index]
            selected.append(
                {
                    "video": str(name),
                    "frame_count": int(length),
                    "length_stratum": stratum_index,
                }
            )
    return sorted(
        selected,
        key=lambda row: (
            int(row["length_stratum"]),
            int(row["frame_count"]),
            str(row["video"]),
        ),
    )


def adjacent_embedding_metrics(
    current: torch.Tensor, previous: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cosine similarity, raw MSE, and symmetric energy-normalized MSE."""
    if current.shape != previous.shape:
        raise ValueError(f"feature shapes differ: {tuple(current.shape)} != {tuple(previous.shape)}")
    current_flat = current.float().flatten(1)
    previous_flat = previous.float().flatten(1)
    cosine = torch.nn.functional.cosine_similarity(current_flat, previous_flat, dim=1, eps=eps)
    squared_difference = (current_flat - previous_flat).square().mean(dim=1)
    energy = 0.5 * (
        current_flat.square().mean(dim=1) + previous_flat.square().mean(dim=1)
    )
    symmetric_nmse = squared_difference / energy.clamp_min(eps)
    return cosine, squared_difference, symmetric_nmse


def amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def extract_video_rows(
    predictor,
    video: str,
    paths: list[Path],
    batch_size: int,
    fps: float,
    device: torch.device,
    amp_dtype: str,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    previous_feature: torch.Tensor | None = None
    previous_paths: list[Path] = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [load_rgb(path) for path in batch_paths]
            with amp_context(device, amp_dtype):
                predictor.set_image_batch(images)
            features = predictor._features["image_embed"].detach().float()

            if previous_feature is not None:
                prior = torch.cat((previous_feature, features[:-1]), dim=0)
                current = features
                metric_paths = batch_paths
                prior_paths = [previous_paths[-1], *batch_paths[:-1]]
                current_ordinals = range(start, start + len(batch_paths))
            else:
                if len(features) < 2:
                    previous_feature = features[-1:].clone()
                    previous_paths = batch_paths
                    continue
                prior = features[:-1]
                current = features[1:]
                metric_paths = batch_paths[1:]
                prior_paths = batch_paths[:-1]
                current_ordinals = range(start + 1, start + len(batch_paths))

            cosine, mse, symmetric_nmse = adjacent_embedding_metrics(current, prior)
            for offset, (previous_path, current_path, current_ordinal) in enumerate(
                zip(prior_paths, metric_paths, current_ordinals)
            ):
                cosine_value = float(cosine[offset].cpu())
                rows.append(
                    {
                        "video": video,
                        "video_length": len(paths),
                        "transition_index": current_ordinal,
                        "previous_frame": previous_path.name,
                        "current_frame": current_path.name,
                        "time_seconds": current_ordinal / fps,
                        "cosine_similarity": cosine_value,
                        "cosine_distance": 1.0 - cosine_value,
                        "mse": float(mse[offset].cpu()),
                        "symmetric_nmse": float(symmetric_nmse[offset].cpu()),
                    }
                )
            previous_feature = features[-1:].clone()
            previous_paths = batch_paths
    return rows


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_rows(
    rows: list[dict[str, int | float | str]],
) -> list[dict[str, int | float | str]]:
    grouped: dict[str, list[dict[str, int | float | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["video"]), []).append(row)
    summaries = []
    for video, video_rows in grouped.items():
        cosine = [float(row["cosine_similarity"]) for row in video_rows]
        nmse = [float(row["symmetric_nmse"]) for row in video_rows]
        summaries.append(
            {
                "video": video,
                "frame_count": int(video_rows[0]["video_length"]),
                "transitions": len(video_rows),
                "cosine_mean": float(np.mean(cosine)),
                "cosine_median": float(np.median(cosine)),
                "cosine_p05": percentile(cosine, 5),
                "symmetric_nmse_mean": float(np.mean(nmse)),
                "symmetric_nmse_median": float(np.median(nmse)),
                "symmetric_nmse_p95": percentile(nmse, 95),
            }
        )
    return sorted(summaries, key=lambda row: str(row["video"]))


def write_csv(path: Path, rows: list[dict], fieldnames: tuple[str, ...] | list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(rows: list[dict], selection: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to render embedding-drift figures") from error

    by_video: dict[str, list[dict]] = {}
    for row in rows:
        by_video.setdefault(str(row["video"]), []).append(row)
    videos = [str(item["video"]) for item in selection]

    def small_multiples(
        metric: str, label: str, filename: str, log_scale: bool = False
    ) -> None:
        columns = min(4, len(videos))
        rows_count = math.ceil(len(videos) / columns)
        figure, axes = plt.subplots(
            rows_count, columns, figsize=(18, 3 * rows_count), sharex=False, sharey=True
        )
        flat_axes = np.asarray(axes, dtype=object).reshape(-1)
        for axis, video in zip(flat_axes, videos):
            video_rows = by_video[video]
            x = [int(row["transition_index"]) for row in video_rows]
            y = [max(float(row[metric]), 1e-12) if log_scale else float(row[metric]) for row in video_rows]
            axis.plot(x, y, linewidth=0.8, color="#1565c0")
            axis.set_title(f"{video} (n={len(video_rows) + 1})", fontsize=9)
            axis.grid(alpha=0.2, linewidth=0.5)
            if log_scale:
                axis.set_yscale("log")
        for axis in flat_axes[len(videos) :]:
            axis.set_visible(False)
        figure.supxlabel("Current-frame ordinal (comparison is current vs previous frame)")
        figure.supylabel(label)
        figure.suptitle("SAM2.1-L fresh image-embedding adjacent-frame drift", fontsize=14)
        figure.tight_layout(rect=(0.02, 0.02, 1, 0.97))
        figure.savefig(out_dir / filename, dpi=180)
        plt.close(figure)

    small_multiples("cosine_similarity", "Cosine similarity", "cosine_small_multiples.png")
    small_multiples(
        "mse", "Raw MSE (log scale)", "mse_small_multiples.png", True
    )
    small_multiples(
        "symmetric_nmse",
        "Symmetric normalized MSE (log scale)",
        "nmse_small_multiples.png",
        True,
    )

    max_transitions = max(len(by_video[video]) for video in videos)
    cosine_distance = np.full((len(videos), max_transitions), np.nan, dtype=np.float64)
    log_nmse = np.full_like(cosine_distance, np.nan)
    for row_index, video in enumerate(videos):
        values = by_video[video]
        cosine_distance[row_index, : len(values)] = [float(row["cosine_distance"]) for row in values]
        log_nmse[row_index, : len(values)] = np.log10(
            np.maximum([float(row["symmetric_nmse"]) for row in values], 1e-12)
        )
    figure, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    images = [
        axes[0].imshow(cosine_distance, aspect="auto", interpolation="nearest", cmap="magma"),
        axes[1].imshow(log_nmse, aspect="auto", interpolation="nearest", cmap="magma"),
    ]
    axes[0].set_title("Cosine distance (1 - cosine similarity)")
    axes[1].set_title("log10 symmetric normalized MSE")
    for axis, image in zip(axes, images):
        axis.set_yticks(range(len(videos)), labels=videos, fontsize=7)
        figure.colorbar(image, ax=axis, fraction=0.015, pad=0.01)
    axes[1].set_xlabel("Current-frame ordinal; blank area is beyond video length")
    figure.suptitle("Adjacent fresh image-embedding drift heatmap", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(out_dir / "embedding_drift_heatmap.png", dpi=180)
    plt.close(figure)

    x = np.log10(np.maximum([float(row["cosine_distance"]) for row in rows], 1e-12))
    y = np.log10(np.maximum([float(row["symmetric_nmse"]) for row in rows], 1e-12))
    figure, axis = plt.subplots(figsize=(8, 7))
    density = axis.hexbin(x, y, gridsize=55, mincnt=1, bins="log", cmap="viridis")
    figure.colorbar(density, ax=axis, label="log10 transition count")
    axis.set_xlabel("log10 cosine distance (1 - cosine similarity)")
    axis.set_ylabel("log10 symmetric normalized MSE")
    axis.set_title("Do cosine distance and normalized MSE carry distinct signals?")
    axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(out_dir / "cosine_distance_vs_nmse_hexbin.png", dpi=180)
    plt.close(figure)


def git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.batch_size <= 0:
        raise ValueError("fps and batch_size must be positive")
    for path in (args.sam2_root, args.checkpoint, args.image_root, args.video_list_file):
        if not path.exists():
            raise FileNotFoundError(path)
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(args.sam2_root))
    sys.path.insert(0, str(repo_root))

    names = read_video_names(args.video_list_file)
    candidates = [(name, len(frame_paths(args.image_root, name))) for name in names]
    selection = stratified_video_selection(candidates, args.num_videos, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device(args.device)
    model = build_sam2(args.sam2_cfg, str(args.checkpoint), device=str(device), mode="eval")
    model.eval()
    model.requires_grad_(False)
    predictor = SAM2ImagePredictor(model)

    started = time.perf_counter()
    rows: list[dict] = []
    for index, item in enumerate(selection, start=1):
        video = str(item["video"])
        paths = frame_paths(args.image_root, video)
        print(f"[{index:02d}/{len(selection):02d}] {video}: {len(paths)} frames", flush=True)
        rows.extend(
            extract_video_rows(
                predictor, video, paths, args.batch_size, args.fps, device, args.amp_dtype
            )
        )
    elapsed = time.perf_counter() - started

    summaries = summarize_rows(rows)
    write_csv(args.out_dir / "adjacent_embedding_metrics.csv", rows, METRIC_FIELDS)
    write_csv(args.out_dir / "per_video_summary.csv", summaries, list(summaries[0]))
    plot_outputs(rows, selection, args.out_dir)

    cosine = [float(row["cosine_similarity"]) for row in rows]
    nmse = [float(row["symmetric_nmse"]) for row in rows]
    summary = {
        "schema_version": 1,
        "status": "pass",
        "feature": {
            "name": "image_embed",
            "shape_per_frame": [256, 64, 64],
            "source": "SAM2ImagePredictor._features['image_embed']",
            "observation_policy": "fresh image encoder call for every frame",
            "metric_compute_dtype": "float32",
            "encoder_amp_dtype": args.amp_dtype,
        },
        "dataset": "SA-V val JPEGImages_24fps",
        "fps": args.fps,
        "seed": args.seed,
        "videos": len(selection),
        "frames": sum(int(item["frame_count"]) for item in selection),
        "transitions": len(rows),
        "elapsed_seconds": elapsed,
        "global_metrics": {
            "cosine_mean": float(np.mean(cosine)),
            "cosine_median": float(np.median(cosine)),
            "cosine_p05": percentile(cosine, 5),
            "symmetric_nmse_mean": float(np.mean(nmse)),
            "symmetric_nmse_median": float(np.median(nmse)),
            "symmetric_nmse_p95": percentile(nmse, 95),
        },
        "inputs": {
            "sam2_config": args.sam2_cfg,
            "checkpoint": str(args.checkpoint),
            "image_root": str(args.image_root),
            "video_list_file": str(args.video_list_file),
        },
        "git_commit": git_commit(repo_root),
        "outputs": {
            "transition_csv": str(args.out_dir / "adjacent_embedding_metrics.csv"),
            "video_summary_csv": str(args.out_dir / "per_video_summary.csv"),
            "selection": str(args.out_dir / "selection.json"),
            "figures": [
                "cosine_small_multiples.png",
                "mse_small_multiples.png",
                "nmse_small_multiples.png",
                "embedding_drift_heatmap.png",
                "cosine_distance_vs_nmse_hexbin.png",
            ],
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
