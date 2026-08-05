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
        raise FileNotFoundError(f"No rank summaries under {args.run_dir}")
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
    summary = {
        "status": "pass",
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
        "rank_summaries": [str(path) for path in paths],
    }
    out = args.out or args.run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
