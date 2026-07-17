#!/usr/bin/env python3
"""Verify that an online W&B run contains a required training metric."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-file", required=True, type=Path)
    parser.add_argument("--metric", default="train/loss_total")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--poll-seconds", type=int, default=5)
    args = parser.parse_args()

    metadata = json.loads(args.run_file.read_text(encoding="utf-8"))
    required = ("entity", "project", "run_id")
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise SystemExit(f"W&B run metadata is missing: {missing}")
    run_path = "/".join(metadata[key] for key in required)

    import wandb

    deadline = time.monotonic() + args.timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            run = wandb.Api(timeout=args.poll_seconds + 10).run(run_path)
            rows = run.scan_history(keys=[args.metric], page_size=100)
            if any(row.get(args.metric) is not None for row in rows):
                print(
                    f"W&B history: PASS | run {run_path} | metric {args.metric}"
                )
                return
            last_error = RuntimeError(f"metric {args.metric!r} is not uploaded yet")
        except Exception as error:  # noqa: BLE001 - poll remote service
            last_error = error
        print(f"Waiting for W&B history: {last_error}", flush=True)
        time.sleep(args.poll_seconds)
    raise SystemExit(
        f"W&B history verification failed for {run_path}: {last_error}"
    )


if __name__ == "__main__":
    main()
