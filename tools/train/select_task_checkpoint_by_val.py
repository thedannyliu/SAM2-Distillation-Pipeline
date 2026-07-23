#!/usr/bin/env python3
"""Select a task checkpoint by full SA-V validation J&F."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=RUN_DIR",
    )
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--print-selected", action="store_true")
    return parser.parse_args()


def parse_candidate(spec: str) -> tuple[str, Path]:
    name, separator, raw_path = spec.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError(f"invalid --candidate value: {spec!r}")
    return name, Path(raw_path)


def read_metrics(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["mode"]: row for row in csv.DictReader(handle)}


def metric(row: dict[str, str], name: str) -> float | None:
    value = row.get(name, "").strip()
    return float(value) if value else None


def build_row(name: str, run_dir: Path) -> dict[str, Any]:
    output: dict[str, Any] = {
        "candidate": name,
        "selected": 0,
        "val_mIoU": "",
        "val_AP": "",
        "val_J&F": "",
        "val_J": "",
        "val_F": "",
        "test_mIoU": "",
        "test_AP": "",
        "test_J&F": "",
        "test_J": "",
        "test_F": "",
        "run_dir": str(run_dir),
    }
    for split in ("val", "test"):
        rows = read_metrics(
            run_dir / f"sav_{split}_box_benchmark" / "metrics.csv"
        )
        image = rows.get("image", {})
        video = rows.get("video_tracking", {})
        if image and image.get("status") != "pass":
            raise ValueError(f"{name} has incomplete {split} image metrics")
        if video and video.get("status") != "pass":
            raise ValueError(f"{name} has incomplete {split} VOS metrics")
        output[f"{split}_mIoU"] = image.get("mIoU", "")
        output[f"{split}_AP"] = image.get("AP", "")
        output[f"{split}_J&F"] = video.get("J&F", "")
        output[f"{split}_J"] = video.get("J", "")
        output[f"{split}_F"] = video.get("F", "")
    if output["val_J&F"] == "":
        raise ValueError(
            f"{name} is missing full validation J&F: "
            f"{run_dir}/sav_val_box_benchmark/metrics.csv"
        )
    return output


def main() -> None:
    args = parse_args()
    rows = [
        build_row(*parse_candidate(spec))
        for spec in args.candidate
    ]
    selected = max(
        rows,
        key=lambda row: (
            float(row["val_J&F"]),
            float(row["val_mIoU"] or "-inf"),
            float(row["val_AP"] or "-inf"),
        ),
    )
    selected["selected"] = 1
    rows.sort(key=lambda row: (-float(row["val_J&F"]), row["candidate"]))

    payload = {
        "selection_metric": "full_sav_val_J&F",
        "selected_candidate": selected["candidate"],
        "selected_run_dir": selected["run_dir"],
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.print_selected:
        print(selected["candidate"])
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
