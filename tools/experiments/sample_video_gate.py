#!/usr/bin/env python3
"""Create a deterministic hash sample from a video-list file."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--seed", default="edgetam-memory-gate-v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = [
        line.strip()
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if args.count < 1 or args.count > len(videos):
        raise ValueError(
            f"count must be in [1, {len(videos)}], got {args.count}"
        )
    ranked = sorted(
        videos,
        key=lambda video: hashlib.sha256(
            f"{args.seed}:{video}".encode("utf-8")
        ).digest(),
    )
    selected = sorted(ranked[: args.count])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(
        f"Gate list: {args.output} | selected {len(selected)} "
        f"of {len(videos)} | seed {args.seed}"
    )


if __name__ == "__main__":
    main()
