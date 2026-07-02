#!/usr/bin/env python3
"""Append a compact row to a Markdown experiment table."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path


HEADER = """# EdgeTAM TinyViT Smoke Experiments

This table tracks small PACE smoke runs only. Each dataset subset must stay at or below 500 images or frames.

| Date | Task | Data | Command | Seed | GPU | Output | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""


def cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="docs/experiments/edgetam_smoke.md")
    parser.add_argument("--task", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--seed", default="")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--result", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--date", default=date.today().isoformat())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")
    row = (
        f"| {cell(args.date)} | {cell(args.task)} | {cell(args.data)} | "
        f"`{cell(args.command)}` | {cell(args.seed)} | {cell(args.gpu)} | "
        f"{cell(args.output)} | {cell(args.result)} | {cell(args.notes)} |\n"
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(row)
    print(f"recorded={path}")


if __name__ == "__main__":
    main()

