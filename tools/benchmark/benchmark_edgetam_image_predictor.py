#!/usr/bin/env python3
"""Benchmark official EdgeTAM image predictor latency on bounded smoke images."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, int(round((q / 100.0) * (len(sorted_values) - 1)))))
    return float(sorted_values[index])


def load_images(manifest: Path, limit: int) -> list[Path]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    image_paths = [Path(row["image_path"]) for row in rows[:limit]]
    if not image_paths:
        raise SystemExit(f"No image paths found in {manifest}")
    return image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edgetam-root", required=True, type=Path)
    parser.add_argument("--sam2-cfg", default="configs/edgetam.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.limit > 500:
        raise SystemExit("--limit must be in [1, 500] for smoke benchmarks")
    if args.iters < 1:
        raise SystemExit("--iters must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    sys.path.insert(0, str(args.edgetam_root))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device(args.device)
    model = build_sam2(args.sam2_cfg, str(args.checkpoint), device=str(device))
    predictor = SAM2ImagePredictor(model)
    image_paths = load_images(args.manifest, args.limit)
    images = []
    prompts = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            images.append(np.asarray(image))
            prompts.append(
                (
                    np.array([[width / 2.0, height / 2.0]], dtype=np.float32),
                    np.array([1], dtype=np.int32),
                )
            )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies = []
    total_iters = args.warmup + args.iters
    for iteration in range(total_iters):
        image_idx = iteration % len(images)
        point_coords, point_labels = prompts[image_idx]
        sync(device)
        start = time.perf_counter()
        predictor.set_image(images[image_idx])
        predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            normalize_coords=False,
        )
        sync(device)
        elapsed = time.perf_counter() - start
        if iteration >= args.warmup:
            latencies.append(elapsed)

    mean_sec = statistics.fmean(latencies)
    summary = {
        "status": "pass",
        "device": str(device),
        "num_images": len(images),
        "warmup": args.warmup,
        "iters": args.iters,
        "mean_seconds": mean_sec,
        "p50_seconds": float(statistics.median(latencies)),
        "p95_seconds": percentile(latencies, 95),
        "mean_fps": float(1.0 / mean_sec),
        "peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "latencies.jsonl").write_text(
        "".join(json.dumps({"seconds": value}) + "\n" for value in latencies),
        encoding="utf-8",
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
