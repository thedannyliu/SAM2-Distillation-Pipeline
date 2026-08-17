#!/usr/bin/env python3
"""Compare two VOS prediction trees by relative paths and decoded mask pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def compare(left: Path, right: Path) -> dict:
    left_files = {path.relative_to(left): path for path in left.rglob("*.png")}
    right_files = {path.relative_to(right): path for path in right.rglob("*.png")}
    shared = sorted(left_files.keys() & right_files.keys())
    mismatches = []
    for relative in shared:
        with Image.open(left_files[relative]) as left_image, Image.open(
            right_files[relative]
        ) as right_image:
            if not np.array_equal(np.asarray(left_image), np.asarray(right_image)):
                mismatches.append(str(relative))
    left_only = sorted(str(path) for path in left_files.keys() - right_files.keys())
    right_only = sorted(str(path) for path in right_files.keys() - left_files.keys())
    identical = not left_only and not right_only and not mismatches
    return {
        "status": "pass" if identical else "fail",
        "identical": identical,
        "left": str(left),
        "right": str(right),
        "left_pngs": len(left_files),
        "right_pngs": len(right_files),
        "shared_pngs": len(shared),
        "pixel_mismatch_count": len(mismatches),
        "left_only_count": len(left_only),
        "right_only_count": len(right_only),
        "pixel_mismatch_examples": mismatches[:20],
        "left_only_examples": left_only[:20],
        "right_only_examples": right_only[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.left, args.right):
        if not path.is_dir():
            raise FileNotFoundError(path)
    result = compare(args.left, args.right)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
