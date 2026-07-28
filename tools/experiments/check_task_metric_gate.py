#!/usr/bin/env python3
"""Check whether full SA-V validation metrics pass continuation thresholds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--min-jf", required=True, type=float)
    parser.add_argument("--min-miou", type=float)
    parser.add_argument("--min-ap", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.metrics.is_file():
        raise SystemExit(f"missing metrics: {args.metrics}")

    with args.metrics.open(encoding="utf-8", newline="") as handle:
        rows = {row["mode"]: row for row in csv.DictReader(handle)}

    image = rows.get("image", {})
    video = rows.get("video_tracking", {})
    values = {
        "J&F": float(video["J&F"]),
        "mIoU": float(image["mIoU"]),
        "AP": float(image["AP"]),
    }
    checks = {
        "J&F": values["J&F"] >= args.min_jf,
        "mIoU": args.min_miou is None or values["mIoU"] >= args.min_miou,
        "AP": args.min_ap is None or values["AP"] >= args.min_ap,
    }
    print(
        "continuation gate: "
        + ", ".join(
            f"{name}={values[name]:.4f} ({'pass' if passed else 'fail'})"
            for name, passed in checks.items()
        ),
        flush=True,
    )
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
