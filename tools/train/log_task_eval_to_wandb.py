#!/usr/bin/env python3
"""Attach completed task-evaluation metrics to an existing W&B stage run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path


METRICS_BY_MODE = {
    "image": (
        "num_images",
        "num_objects",
        "mIoU",
        "AP",
        "AP50",
        "AP75",
        "mean_set_image_seconds",
        "mean_prompt_seconds",
        "mean_total_object_seconds",
    ),
    "video_tracking": ("J&F", "J", "F", "elapsed_sec", "sec_per_video"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-file", required=True, type=Path)
    parser.add_argument(
        "--metrics",
        required=True,
        action="append",
        metavar="SPLIT=CSV",
        help="Evaluation split and metrics.csv path; repeat for each split.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def read_metrics(specs: list[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for spec in specs:
        split, separator, raw_path = spec.partition("=")
        if not separator or not split or not raw_path:
            raise ValueError(f"invalid --metrics value: {spec!r}")
        path = Path(raw_path)
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            mode = row.get("mode", "")
            if row.get("status") != "pass" or mode not in METRICS_BY_MODE:
                raise ValueError(f"incomplete evaluation row in {path}: {row}")
            for name in METRICS_BY_MODE[mode]:
                value = row.get(name, "").strip()
                if value:
                    output[f"eval/{split}/{mode}/{name}"] = float(value)
    return output


def main() -> None:
    args = parse_args()
    run_info = json.loads(args.run_file.read_text())
    metrics = read_metrics(args.metrics)
    import wandb

    run = wandb.init(
        entity=run_info.get("entity"),
        project=run_info["project"],
        id=run_info["run_id"],
        resume="must",
        dir=str(args.run_file.parent),
    )
    run.summary.update(metrics)
    run.finish()

    entity = run_info.get("entity")
    if not entity:
        raise ValueError(f"W&B run metadata has no entity: {args.run_file}")
    run_path = f"{entity}/{run_info['project']}/{run_info['run_id']}"
    verify_name, verify_value = next(iter(sorted(metrics.items())))
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        remote = wandb.Api().run(run_path)
        remote_value = remote.summary.get(verify_name)
        if remote_value is not None and math.isclose(
            float(remote_value), verify_value, rel_tol=1e-9, abs_tol=1e-12
        ):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"W&B summary did not expose {verify_name!r} for {run_path}"
            )
        time.sleep(5.0)
    print(
        f"W&B evaluation summary: PASS | run {run_path} | "
        f"metrics {len(metrics)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
