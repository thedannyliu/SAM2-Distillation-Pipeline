#!/usr/bin/env python
"""Create a deterministic frame-major teacher feature cache for smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=250107256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 1 or args.frames > 500:
        raise SystemExit("--frames must be in [1, 500] for smoke caches")
    generator = torch.Generator().manual_seed(args.seed)
    shape = (args.frames, args.channels, args.height, args.width)
    payload = {
        "schema": "edgetam_teacher_feature_cache_v1",
        "seed": args.seed,
        "teacher_distill_F16": torch.randn(shape, generator=generator, dtype=torch.float16),
        "teacher_distill_F_M": torch.randn(shape, generator=generator, dtype=torch.float16),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    summary = {
        "out": str(args.out),
        "frames": args.frames,
        "shape": list(shape),
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
