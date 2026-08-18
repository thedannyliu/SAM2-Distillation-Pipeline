#!/usr/bin/env python3
"""Summarize a fixed-phase two-clock evaluation with paired video statistics."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_MODELS = ("O0", "W0", "O1", "O2", "A", "B", "C", "D", "E")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _unit(root: Path, model: str, interval: int, phase: int) -> Path:
    return root / model / f"R{interval}" / f"phase_{phase}"


def _weighted_video_metrics(age_payload: dict) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in age_payload["per_video"]:
        groups[row["video"]].append(row)
    result = {}
    for video, rows in groups.items():
        count = sum(row["count"] for row in rows)
        result[video] = {
            metric: sum(row[metric] * row["count"] for row in rows) / count
            for metric in ("J", "F", "J&F")
        }
    return result


def _macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in ("J", "F", "J&F")
    }


def _bootstrap_delta(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    *,
    samples: int,
    seed: int,
) -> dict:
    videos = sorted(set(left) & set(right))
    differences = [left[name]["J&F"] - right[name]["J&F"] for name in videos]
    rng = random.Random(seed)
    means = [
        float(np.mean([differences[rng.randrange(len(differences))] for _ in videos]))
        for _ in range(samples)
    ]
    return {
        "videos": len(videos),
        "delta_J&F": float(np.mean(differences)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def summarize(
    root: Path,
    models: tuple[str, ...] = DEFAULT_MODELS,
    *,
    intervals: dict[str, int] | None = None,
    bootstrap_samples: int = 10_000,
    seed: int = 250107256,
) -> dict:
    intervals = intervals or {
        model: 1 if model in {"O0", "W0"} else 4 for model in models
    }
    if set(intervals) != set(models):
        raise ValueError("intervals must define exactly one interval per model")
    phase_rows = []
    video_metrics: dict[str, dict[str, dict[str, float]]] = {}
    age_rows = []
    model_rows = []
    for model in models:
        interval = intervals[model]
        phases = range(interval)
        phase_video = {}
        phase_timing = []
        for phase in phases:
            unit = _unit(root, model, interval, phase)
            sav = _load(unit / "sav_eval.json")
            age = _load(unit / "age_metrics.json")
            timing = _load(unit / "pred/summary.json")
            if sav.get("status") != "pass" or age.get("status") != "pass":
                raise RuntimeError(f"incomplete evaluation unit: {unit}")
            phase_video[phase] = _weighted_video_metrics(age)
            phase_rows.append(
                {
                    "model": model,
                    "interval": interval,
                    "phase": phase,
                    **sav["metrics"],
                }
            )
            phase_timing.append(timing)

        videos = sorted(set.intersection(*(set(rows) for rows in phase_video.values())))
        model_video = {
            video: _macro([phase_video[phase][video] for phase in phases])
            for video in videos
        }
        video_metrics[model] = model_video
        metrics = _macro(list(model_video.values()))

        by_age_video: dict[int, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for phase in phases:
            age = _load(_unit(root, model, interval, phase) / "age_metrics.json")
            for row in age["per_video"]:
                by_age_video[row["age"]][row["video"]].append(row)
        for feature_age, groups in sorted(by_age_video.items()):
            per_video = [_macro(rows) for rows in groups.values()]
            age_rows.append(
                {
                    "model": model,
                    "age": feature_age,
                    "videos": len(per_video),
                    **_macro(per_video),
                }
            )

        model_rows.append(
            {
                "model": model,
                "interval": interval,
                "phases": interval,
                "videos": len(model_video),
                **metrics,
                "refresh_pct": 100
                * float(np.mean([row["two_clock_refresh_rate"] for row in phase_timing])),
                "model_mean_ms": float(
                    np.mean([row["model_frame_mean_ms"] for row in phase_timing])
                ),
                "model_p95_max_ms": float(
                    max(row["model_frame_p95_ms"] for row in phase_timing)
                ),
                "wall_mean_ms": float(
                    np.mean([row["wall_ms_per_frame"] for row in phase_timing])
                ),
                "parallel_throughput_ms": float(
                    np.mean(
                        [
                            row.get(
                                "parallel_throughput_ms_per_frame",
                                row["wall_ms_per_frame"],
                            )
                            for row in phase_timing
                        ]
                    )
                ),
            }
        )

    paired = {}
    for model in models:
        paired[model] = {}
        for reference in ("O0", "O1"):
            if model == reference or reference not in video_metrics:
                continue
            paired[model][reference] = _bootstrap_delta(
                video_metrics[model],
                video_metrics[reference],
                samples=bootstrap_samples,
                seed=seed,
            )
    return {
        "status": "pass",
        "protocol": "eval30_fixed_phase_v1",
        "phase_neutral_primary": True,
        "models": model_rows,
        "by_phase_official": phase_rows,
        "by_age_paired_video_macro": age_rows,
        "paired_bootstrap": paired,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: dict) -> str:
    lines = [
        "# SAM2.1-L eval30 fixed-phase report",
        "",
        "Primary metric: per-video metrics averaged over all fixed phases, then macro-averaged over videos.",
        "",
        "| Model | R | Videos | J&F | J | F | Refresh % | Model mean ms | Model P95 max ms | Single-stream wall ms | 4-GPU throughput ms/frame |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["models"]:
        lines.append(
            f"| {row['model']} | {row['interval']} | {row['videos']} | "
            f"{row['J&F']:.2f} | {row['J']:.2f} | {row['F']:.2f} | "
            f"{row['refresh_pct']:.2f} | {row['model_mean_ms']:.2f} | "
            f"{row['model_p95_max_ms']:.2f} | {row['wall_mean_ms']:.2f} | "
            f"{row['parallel_throughput_ms']:.2f} |"
        )
    lines.extend(["", "## Paired J&F differences", ""])
    lines.extend(
        [
            "| Model | Reference | Delta | 95% CI | Videos |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model, references in payload["paired_bootstrap"].items():
        for reference, row in references.items():
            lines.append(
                f"| {model} | {reference} | {row['delta_J&F']:.2f} | "
                f"[{row['ci95_low']:.2f}, {row['ci95_high']:.2f}] | {row['videos']} |"
            )
    lines.extend(["", "## Paired feature-age metrics", ""])
    lines.extend(
        [
            "| Model | Age | Videos | J&F | J | F |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["by_age_paired_video_macro"]:
        lines.append(
            f"| {row['model']} | {row['age']} | {row['videos']} | "
            f"{row['J&F']:.2f} | {row['J']:.2f} | {row['F']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument(
        "--model-interval",
        action="append",
        default=[],
        metavar="MODEL=K",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=250107256)
    args = parser.parse_args()
    models = tuple(value for value in args.models.split(",") if value)
    if args.model_interval:
        intervals = {
            model: int(interval)
            for model, interval in (
                value.split("=", 1) for value in args.model_interval
            )
        }
    else:
        intervals = None
    payload = summarize(
        args.root,
        models=models,
        intervals=intervals,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.out_dir / "models.csv", payload["models"])
    _write_csv(args.out_dir / "by_phase.csv", payload["by_phase_official"])
    _write_csv(args.out_dir / "by_age.csv", payload["by_age_paired_video_macro"])
    report = _markdown(payload)
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
