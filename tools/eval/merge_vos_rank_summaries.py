#!/usr/bin/env python3
"""Merge torchrun rank summaries from generic VOS inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    paths = sorted(args.run_dir.glob("summary.rank*.json"))
    if not paths:
        single = args.run_dir / "summary.json"
        if not single.is_file():
            raise FileNotFoundError(f"No rank summaries under {args.run_dir}")
        paths = [single]
    ranks = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected = int(ranks[0]["world_size"])
    if len(ranks) != expected or sorted(row["rank"] for row in ranks) != list(range(expected)):
        raise RuntimeError(f"Expected {expected} distinct rank summaries, found {len(ranks)}")
    frames = sum(int(row["processed_frames"]) for row in ranks)
    videos = sum(len(row["video_names"]) for row in ranks)
    gpu_seconds = sum(float(row["elapsed_sec"]) for row in ranks)
    wall_seconds = max(float(row["elapsed_sec"]) for row in ranks)
    timed_frames = sum(int(row["model_timed_frames"]) for row in ranks)
    model_latency_sum_ms = sum(float(row["model_frame_latency_sum_ms"]) for row in ranks)
    encoder_calls = sum(int(row.get("two_clock_encoder_calls", 0)) for row in ranks)
    tracking_frames = sum(int(row.get("two_clock_tracking_frames", 0)) for row in ranks)
    phase_modes = {row.get("two_clock_refresh_phase_mode") for row in ranks}
    phase_seeds = {row.get("two_clock_refresh_phase_seed") for row in ranks}
    video_storage = {bool(row.get("offload_video_to_cpu", False)) for row in ranks}
    model_kinds = {row.get("model_kind") for row in ranks}
    build_experiments = {row.get("build", {}).get("experiment") for row in ranks}
    build_intervals = {row.get("build", {}).get("refresh_interval") for row in ranks}
    if any(
        len(values) != 1
        for values in (
            phase_modes,
            phase_seeds,
            video_storage,
            model_kinds,
            build_experiments,
            build_intervals,
        )
    ):
        raise RuntimeError("rank summaries disagree on evaluation protocol")
    summary = {
        "status": "pass",
        "model_kind": model_kinds.pop(),
        "build_experiment": build_experiments.pop(),
        "build_refresh_interval": build_intervals.pop(),
        "world_size": expected,
        "videos": videos,
        "processed_frames": frames,
        "gpu_seconds": gpu_seconds,
        "parallel_wall_seconds": wall_seconds,
        "single_stream_wall_ms_per_frame": gpu_seconds * 1000.0 / max(frames, 1),
        "parallel_throughput_ms_per_frame": wall_seconds * 1000.0 / max(frames, 1),
        "model_timed_frames": timed_frames,
        "single_stream_model_mean_ms": model_latency_sum_ms / max(timed_frames, 1),
        "rank_model_median_ms": [row["model_frame_median_ms"] for row in ranks],
        "rank_model_p95_ms": [row.get("model_frame_p95_ms") for row in ranks],
        "rank_summaries": [str(path) for path in paths],
        "two_clock_encoder_calls": encoder_calls,
        "two_clock_tracking_frames": tracking_frames,
        "two_clock_refresh_rate": encoder_calls / max(tracking_frames, 1),
        "two_clock_refresh_phase_mode": phase_modes.pop(),
        "two_clock_refresh_phase_seed": phase_seeds.pop(),
        "offload_video_to_cpu": video_storage.pop(),
    }
    out = args.out or args.run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
