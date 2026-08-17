#!/usr/bin/env python3
"""Report GT-only SA-V J/F by fixed-policy feature age."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--pred-root", required=True, type=Path)
    parser.add_argument("--refresh-interval", required=True, type=int)
    parser.add_argument(
        "--refresh-phase-mode", choices=("anchor", "balanced"), default="anchor"
    )
    parser.add_argument("--refresh-phase-seed", type=int, default=250107256)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--video-list-file", type=Path)
    parser.add_argument("--include-first-and-last", action="store_true")
    return parser.parse_args()


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image) > 0


def _single_frame_metrics(evaluator_class, prediction, target) -> tuple[float, float]:
    if not prediction.any() and not target.any():
        return 100.0, 100.0
    evaluator = evaluator_class()
    evaluator.feed_frame(prediction, target)
    iou, boundary = evaluator.conclude()
    keys = set(iou) | set(boundary)
    if len(keys) != 1:
        raise RuntimeError(f"expected one binary object, found {keys}")
    key = next(iter(keys))
    return float(iou[key]), float(boundary[key])


def main() -> None:
    args = parse_args()
    if not 1 <= args.refresh_interval <= 6:
        raise ValueError("v1 refresh interval must be R1--R6")
    for path in (args.sam2_root, args.gt_root, args.pred_root):
        if not path.exists():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(args.sam2_root / "sav_dataset"))
    from utils.sav_benchmark import Evaluator
    from sam2_distill.two_clock.schedule import (
        balanced_refresh_phase,
        fixed_refresh_source,
    )

    if args.video_list_file:
        video_names = [
            line.strip()
            for line in args.video_list_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        video_names = sorted(path.name for path in args.gt_root.iterdir() if path.is_dir())

    by_age: dict[int, list[tuple[float, float]]] = defaultdict(list)
    per_video: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for video_name in video_names:
        phase = 0
        if args.refresh_phase_mode == "balanced":
            phase = balanced_refresh_phase(
                video_name,
                args.refresh_interval,
                seed=args.refresh_phase_seed,
            )
        gt_video = args.gt_root / video_name
        pred_video = args.pred_root / video_name
        for object_dir in sorted(path for path in gt_video.iterdir() if path.is_dir()):
            frames = sorted(object_dir.glob("*.png"), key=lambda path: int(path.stem))
            if not frames:
                continue
            anchor = int(frames[0].stem)
            scored = frames if args.include_first_and_last else frames[1:-1]
            for gt_path in scored:
                pred_path = pred_video / object_dir.name / gt_path.name
                if not pred_path.is_file():
                    raise FileNotFoundError(pred_path)
                raw_frame = int(gt_path.stem)
                source = fixed_refresh_source(
                    raw_frame,
                    anchor=anchor,
                    interval=args.refresh_interval,
                    phase=phase,
                )
                age = raw_frame - source
                metrics = _single_frame_metrics(
                    Evaluator, _mask(pred_path), _mask(gt_path)
                )
                by_age[age].append(metrics)
                per_video[video_name][age].append(metrics)

    rows = []
    for age in range(args.refresh_interval):
        values = by_age.get(age, [])
        if values:
            j = float(np.mean([value[0] for value in values]))
            f = float(np.mean([value[1] for value in values]))
        else:
            j = f = float("nan")
        rows.append(
            {
                "age": age,
                "count": len(values),
                "J": j,
                "F": f,
                "J&F": (j + f) / 2,
            }
        )
    missing_ages = [row["age"] for row in rows if row["count"] == 0]
    if missing_ages:
        raise RuntimeError(
            "GT-only evaluation did not cover feature ages "
            f"{missing_ages}; check refresh-phase balancing"
        )

    video_rows = []
    for video_name, age_groups in sorted(per_video.items()):
        for age, values in sorted(age_groups.items()):
            j = float(np.mean([value[0] for value in values]))
            f = float(np.mean([value[1] for value in values]))
            video_rows.append(
                {
                    "video": video_name,
                    "age": age,
                    "count": len(values),
                    "J": j,
                    "F": f,
                    "J&F": (j + f) / 2,
                }
            )

    payload = {
        "status": "pass",
        "protocol": "balanced_phase_v1"
        if args.refresh_phase_mode == "balanced"
        else "anchor_phase_v1",
        "refresh_interval": args.refresh_interval,
        "refresh_phase_mode": args.refresh_phase_mode,
        "refresh_phase_seed": args.refresh_phase_seed,
        "gt_only": True,
        "skip_first_and_last": not args.include_first_and_last,
        "videos": len(per_video),
        "by_age": rows,
        "per_video": video_rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["age", "count", "J", "F", "J&F"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in payload.items() if key != "per_video"}, indent=2))


if __name__ == "__main__":
    main()
