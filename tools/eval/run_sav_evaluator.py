#!/usr/bin/env python3
"""Run the official SAM2 SA-V evaluator on an existing prediction root."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--pred-root", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--do-not-skip-first-and-last-frame", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.evaluator.exists():
        raise FileNotFoundError(f"Missing evaluator: {args.evaluator}")
    if not args.gt_root.exists():
        raise FileNotFoundError(f"Missing GT root: {args.gt_root}")
    if not args.pred_root.exists():
        raise FileNotFoundError(f"Missing prediction root: {args.pred_root}")

    command = [
        sys.executable,
        str(args.evaluator),
        "--gt_root",
        str(args.gt_root),
        "--pred_root",
        str(args.pred_root),
        "--num_processes",
        str(args.num_processes),
    ]
    if args.strict:
        command.append("--strict")
    if args.do_not_skip_first_and_last_frame:
        command.append("--do_not_skip_first_and_last_frame")

    result = subprocess.run(
        command,
        check=False,
        cwd=str(args.evaluator.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    summary = {
        "status": "pass" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "command": " ".join(command),
        "output_tail": result.stdout[-4000:],
        "results_csv": str(args.pred_root / "results.csv"),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
