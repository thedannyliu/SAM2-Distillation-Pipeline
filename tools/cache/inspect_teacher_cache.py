#!/usr/bin/env python3
"""Inspect SAM2 Stage 1 teacher feature cache shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--check-values", action="store_true", help="Check all cached arrays for NaN/Inf.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.cache_root).expanduser().resolve()
    shards = sorted(root.glob("shard-*.zarr"))
    if not shards:
        raise SystemExit(f"No shards found under {root}")

    total_rows = 0
    for shard in shards:
        group = zarr.open_group(str(shard), mode="r")
        rows = int(group.attrs.get("num_rows", group["image_embed"].shape[0]))
        total_rows += rows
        print(f"{shard.name}: rows={rows}")
        for name, expected in [
            ("image_embed", (256, 64, 64)),
            ("high_res_s0", (32, 256, 256)),
            ("high_res_s1", (64, 128, 128)),
        ]:
            arr = group[name]
            print(f"  {name}: shape={arr.shape} dtype={arr.dtype}")
            if arr.shape[1:] != expected:
                raise SystemExit(f"{shard}/{name} has unexpected shape {arr.shape}")
            if args.check_values:
                values = arr[:]
                if not np.isfinite(values).all():
                    raise SystemExit(f"{shard}/{name} contains NaN or Inf")

    print(f"total_rows={total_rows}")


if __name__ == "__main__":
    main()
