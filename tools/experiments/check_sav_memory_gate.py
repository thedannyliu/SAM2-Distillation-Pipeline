#!/usr/bin/env python3
"""Check a compact-memory SA-V mini-val result against a functional reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--reference-metrics", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--min-jf", type=float, default=60.0)
    parser.add_argument("--max-jf-drop", type=float, default=10.0)
    parser.add_argument("--max-miou-drop", type=float, default=0.005)
    parser.add_argument("--max-ap-drop", type=float, default=0.005)
    return parser.parse_args()


def read_metrics(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = {row["mode"]: row for row in csv.DictReader(handle)}
    image = rows.get("image", {})
    video = rows.get("video_tracking", {})
    if image.get("status") != "pass" or video.get("status") != "pass":
        raise ValueError(f"incomplete SA-V metrics: {path}")
    return {
        "mIoU": float(image["mIoU"]),
        "AP": float(image["AP"]),
        "J&F": float(video["J&F"]),
    }


def main() -> None:
    args = parse_args()
    metrics = read_metrics(args.metrics)
    reference = read_metrics(args.reference_metrics)
    deltas = {
        key: metrics[key] - reference[key]
        for key in ("mIoU", "AP", "J&F")
    }
    checks = {
        "absolute_J&F": metrics["J&F"] >= args.min_jf,
        "relative_J&F": deltas["J&F"] >= -args.max_jf_drop,
        "relative_mIoU": deltas["mIoU"] >= -args.max_miou_drop,
        "relative_AP": deltas["AP"] >= -args.max_ap_drop,
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "metrics_path": str(args.metrics),
        "reference_metrics_path": str(args.reference_metrics),
        "metrics": metrics,
        "reference": reference,
        "deltas": deltas,
        "thresholds": {
            "min_J&F": args.min_jf,
            "max_J&F_drop": args.max_jf_drop,
            "max_mIoU_drop": args.max_miou_drop,
            "max_AP_drop": args.max_ap_drop,
        },
        "checks": checks,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
