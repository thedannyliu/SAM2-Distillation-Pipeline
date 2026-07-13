#!/usr/bin/env python3
"""Build comparison and incomplete-run reports from a Stage 1 progress audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_FIELDS = [
    "family",
    "queue",
    "name",
    "registered",
    "run_status",
    "split",
    "evaluation_complete",
    "expected_videos",
    "image_videos",
    "vos_videos",
    "image_mIoU",
    "image_AP",
    "image_AP50",
    "image_AP75",
    "image_set_image_sec",
    "image_prompt_sec",
    "image_total_object_sec",
    "video_J&F",
    "video_J",
    "video_F",
    "video_elapsed_sec",
    "video_sec_per_video",
    "evaluation_issues",
    "run_dir",
    "metrics_csv",
]

INCOMPLETE_FIELDS = [
    "family",
    "queue",
    "name",
    "registered",
    "status",
    "step",
    "target_steps",
    "progress_pct",
    "training_complete",
    "best_ready",
    "full_val_complete",
    "full_test_complete",
    "resumable",
    "wandb_run_id",
    "next_action",
    "issues",
    "run_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def first_mode(rows: list[dict[str, str]], mode: str) -> dict[str, str]:
    return next(
        (
            row
            for row in rows
            if row.get("mode") == mode and row.get("prompt", "box") == "box"
        ),
        {},
    )


def value(row: dict[str, str], key: str, fallback: Any = "") -> Any:
    result = row.get(key)
    return fallback if result in (None, "") else result


def metric_row(run: dict[str, Any], split: str) -> dict[str, Any]:
    detail_key = "full_val" if split == "sav_val" else "full_test"
    detail = run.get(detail_key) or {}
    benchmark_root = Path(run["run_dir"]) / f"{split}_box_benchmark"
    metrics_path = benchmark_root / "metrics.csv"
    rows = read_csv(metrics_path)
    image = first_mode(rows, "image")
    video = first_mode(rows, "video_tracking")
    reasons = detail.get("reasons") or []
    return {
        "family": run.get("family", ""),
        "queue": run.get("queue", ""),
        "name": run.get("name", ""),
        "registered": run.get("registered", False),
        "run_status": run.get("status", ""),
        "split": split,
        "evaluation_complete": bool(detail.get("complete")),
        "expected_videos": detail.get("expected_videos", ""),
        "image_videos": detail.get("image_videos", ""),
        "vos_videos": detail.get("vos_videos", ""),
        "image_mIoU": value(image, "mIoU", detail.get("mIoU", "")),
        "image_AP": value(image, "AP", detail.get("AP", "")),
        "image_AP50": value(image, "AP50"),
        "image_AP75": value(image, "AP75"),
        "image_set_image_sec": value(image, "mean_set_image_seconds"),
        "image_prompt_sec": value(image, "mean_prompt_seconds"),
        "image_total_object_sec": value(image, "mean_total_object_seconds"),
        "video_J&F": value(video, "J&F", detail.get("J&F", "")),
        "video_J": value(video, "J", detail.get("J", "")),
        "video_F": value(video, "F", detail.get("F", "")),
        "video_elapsed_sec": value(video, "elapsed_sec"),
        "video_sec_per_video": value(video, "sec_per_video"),
        "evaluation_issues": "; ".join(str(reason) for reason in reasons),
        "run_dir": run.get("run_dir", ""),
        "metrics_csv": str(metrics_path),
    }


def next_action(run: dict[str, Any]) -> str:
    status = str(run.get("status", ""))
    if status in {"complete", "unregistered_complete"}:
        return "none"
    if "needs_full_eval" in status:
        return "run full sav_val and sav_test evaluation on best.pt"
    if "needs_final_validation" in status:
        return "run final validation and write best.pt, then evaluate"
    if "resumable" in status:
        return "resume training from last.pt, then validate and evaluate"
    if status == "missing":
        return "start the registered experiment"
    return "inspect checkpoint and run metadata before recovery"


def incomplete_row(run: dict[str, Any]) -> dict[str, Any] | None:
    if run.get("status") in {"complete", "unregistered_complete"}:
        return None
    issues = []
    if not run.get("training_complete"):
        issues.append(f"training status: {run.get('status', 'unknown')}")
    if not run.get("best_ready"):
        issues.append("best.pt missing or invalid")
    for label, key in (("sav_val", "full_val"), ("sav_test", "full_test")):
        detail = run.get(key) or {}
        if not detail.get("complete"):
            reasons = detail.get("reasons") or ["evaluation incomplete"]
            issues.append(f"{label}: " + "; ".join(str(reason) for reason in reasons))
    wandb = run.get("wandb") or {}
    return {
        "family": run.get("family", ""),
        "queue": run.get("queue", ""),
        "name": run.get("name", ""),
        "registered": run.get("registered", False),
        "status": run.get("status", ""),
        "step": run.get("step", ""),
        "target_steps": run.get("target_steps", ""),
        "progress_pct": run.get("progress_pct", ""),
        "training_complete": run.get("training_complete", False),
        "best_ready": run.get("best_ready", False),
        "full_val_complete": bool((run.get("full_val") or {}).get("complete")),
        "full_test_complete": bool((run.get("full_test") or {}).get("complete")),
        "resumable": run.get("resumable", False),
        "wandb_run_id": wandb.get("run_id", ""),
        "next_action": next_action(run),
        "issues": " | ".join(issues),
        "run_dir": run.get("run_dir", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def display(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_report(
    source: dict[str, Any], metrics: list[dict[str, Any]], incomplete: list[dict[str, Any]]
) -> str:
    lines = [
        "# Stage 1 Experiment Metrics",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Only rows marked `complete` have full split coverage and current `best.pt` metrics.",
        "",
        "## Status",
        "",
        "| Status | Runs |",
        "| --- | ---: |",
    ]
    for status, count in sorted(Counter(row["status"] for row in source["rows"]).items()):
        lines.append(f"| {status} | {count} |")
    for split in ("sav_val", "sav_test"):
        lines.extend(
            [
                "",
                f"## {split} Key Metrics",
                "",
                "| Family | Experiment | Complete | mIoU | AP | Image sec/object | J&F | Video sec/video |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in (item for item in metrics if item["split"] == split):
            lines.append(
                "| {family} | {name} | {complete} | {miou} | {ap} | {image_latency} | "
                "{jf} | {video_latency} |".format(
                    family=row["family"],
                    name=row["name"],
                    complete="yes" if row["evaluation_complete"] else "no",
                    miou=display(row["image_mIoU"]),
                    ap=display(row["image_AP"]),
                    image_latency=display(row["image_total_object_sec"], 6),
                    jf=display(row["video_J&F"], 2),
                    video_latency=display(row["video_sec_per_video"], 4),
                )
            )
    lines.extend(
        [
            "",
            "## Incomplete Runs",
            "",
            "| Family | Experiment | Status | Step | Progress | Next action |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    if not incomplete:
        lines.append("| - | - | none | - | - | none |")
    for row in incomplete:
        progress = display(row["progress_pct"], 2)
        lines.append(
            f"| {row['family']} | {row['name']} | {row['status']} | {row['step']} | "
            f"{progress}% | {row['next_action']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source = json.loads(args.progress_json.read_text(encoding="utf-8"))
    runs = source.get("rows")
    if not isinstance(runs, list):
        raise SystemExit(f"Invalid progress report: {args.progress_json}")
    metrics = [metric_row(run, split) for run in runs for split in ("sav_val", "sav_test")]
    incomplete = [row for run in runs if (row := incomplete_row(run)) is not None]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "experiment_key_metrics.csv", metrics, METRIC_FIELDS)
    write_csv(args.out_dir / "incomplete_runs.csv", incomplete, INCOMPLETE_FIELDS)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "progress_report": str(args.progress_json),
        "status_counts": dict(Counter(run.get("status", "unknown") for run in runs)),
        "metrics": metrics,
        "incomplete_runs": incomplete,
    }
    (args.out_dir / "experiment_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (args.out_dir / "experiment_report.md").write_text(
        markdown_report(source, metrics, incomplete), encoding="utf-8"
    )
    print(json.dumps({"runs": len(runs), "incomplete": len(incomplete)}, indent=2))
    for name in (
        "experiment_key_metrics.csv",
        "incomplete_runs.csv",
        "experiment_report.md",
        "experiment_report.json",
    ):
        print(f"{name}: {args.out_dir / name}")


if __name__ == "__main__":
    main()
