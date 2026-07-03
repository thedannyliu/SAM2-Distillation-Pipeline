#!/usr/bin/env python3
"""Stream SAM2 trainer text logs into W&B scalar metrics."""

from __future__ import annotations

import argparse
import json
import re
import signal
import time
from pathlib import Path


TRAIN_RE = re.compile(r"Train Epoch: \[(?P<epoch>\d+)\]\[(?P<step>\d+)/(?P<total>\d+)\]")
SCALAR_RE = re.compile(
    r"(?P<key>[A-Za-z0-9_./ -]+):\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
    r"(?:\s*\((?P<avg>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\))?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--poll-sec", type=float, default=5.0)
    parser.add_argument("--start-at-end", action="store_true")
    return parser.parse_args()


def metric_key(phase: str, key: str, suffix: str | None = None) -> str:
    cleaned = key.strip().replace(" ", "_")
    cleaned = cleaned.replace("Losses/", "loss/")
    base = f"{phase}/{cleaned}"
    return f"{base}_{suffix}" if suffix else base


def parse_train_line(line: str, phase: str, line_index: int) -> dict[str, float] | None:
    match = TRAIN_RE.search(line)
    if match is None:
        return None
    epoch = int(match.group("epoch"))
    step = int(match.group("step"))
    total = int(match.group("total"))
    metrics: dict[str, float] = {
        f"{phase}/epoch": float(epoch),
        f"{phase}/step_in_epoch": float(step),
        f"{phase}/steps_per_epoch": float(total),
        f"{phase}/progress_pct": 100.0 * (step + 1) / max(total, 1),
        f"{phase}/log_line": float(line_index),
    }
    for segment in line.split("|")[1:]:
        scalar = SCALAR_RE.search(segment)
        if scalar is None:
            continue
        key = scalar.group("key").strip()
        if key in {"Time Elapsed", "Mem"}:
            continue
        value = float(scalar.group("value"))
        avg = scalar.group("avg")
        metrics[metric_key(phase, key)] = value
        if avg is not None:
            metrics[metric_key(phase, key, "avg")] = float(avg)
    return metrics


def iter_new_lines(path: Path, start_at_end: bool, should_stop):
    while not path.exists() and not should_stop():
        time.sleep(1.0)
    if should_stop():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        if start_at_end:
            handle.seek(0, 2)
        while not should_stop():
            line = handle.readline()
            if line:
                yield line
            else:
                time.sleep(0.5)


def main() -> None:
    args = parse_args()
    import wandb

    wandb_dir = args.out_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=args.project,
        name=args.name,
        id=args.run_id,
        resume="allow",
        dir=str(wandb_dir),
        config={
            "live_log_source": str(args.log_file),
            "live_log_phase": args.phase,
            "out_dir": str(args.out_dir),
        },
    )
    should_stop = False

    def stop_handler(signum, frame):  # noqa: ARG001
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    (args.out_dir / f"wandb_live_{args.phase}.json").write_text(
        json.dumps(
            {
                "run_id": run.id,
                "project": args.project,
                "name": args.name,
                "url": run.url,
                "phase": args.phase,
                "log_file": str(args.log_file),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    line_index = 0
    for line in iter_new_lines(args.log_file, start_at_end=args.start_at_end, should_stop=lambda: should_stop):
        line_index += 1
        metrics = parse_train_line(line, args.phase, line_index)
        if metrics:
            wandb.log(metrics)
    run.finish()


if __name__ == "__main__":
    main()
