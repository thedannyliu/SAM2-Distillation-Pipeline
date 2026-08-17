#!/usr/bin/env python3
"""Build the registered J&F/latency comparison table for two-clock validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPERIMENTS = ("O2", "A", "B", "C", "D", "E")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(model: str, interval: int, output: Path) -> dict:
    metrics = _load(output / "sav_eval.json")["metrics"]
    timing = _load(output / "pred/summary.json")
    p95_values = [
        float(value)
        for value in timing.get("rank_model_p95_ms", [])
        if value is not None
    ]
    return {
        "model": model,
        "refresh_interval": interval,
        "skip_frames": interval - 1,
        "J&F": float(metrics["J&F"]),
        "J": float(metrics["J"]),
        "F": float(metrics["F"]),
        "refresh_rate": float(timing["two_clock_refresh_rate"]),
        "model_mean_ms": float(timing["single_stream_model_mean_ms"]),
        "model_rank_p95_max_ms": max(p95_values) if p95_values else None,
        "wall_ms_per_frame": float(timing["single_stream_wall_ms_per_frame"]),
        "offload_video_to_cpu": bool(timing.get("offload_video_to_cpu", False)),
        "output": str(output),
    }


def collect(run_root: Path) -> tuple[list[dict], list[str]]:
    specs = [
        ("O0", 1, run_root / "controls/O0/R1"),
        ("W0", 1, run_root / "controls/W0/R1"),
    ]
    specs.extend(
        ("O1", interval, run_root / f"controls/O1/R{interval}")
        for interval in range(2, 7)
    )
    specs.extend(
        (model, interval, run_root / model / f"val/selected/R{interval}")
        for model in EXPERIMENTS
        for interval in range(1, 7)
    )
    rows = []
    missing = []
    for model, interval, output in specs:
        required = (output / "sav_eval.json", output / "pred/summary.json")
        absent = [str(path) for path in required if not path.is_file()]
        if absent:
            missing.extend(absent)
            continue
        rows.append(_row(model, interval, output))

    o0 = next((row for row in rows if row["model"] == "O0"), None)
    matched = {
        row["refresh_interval"]: row
        for row in rows
        if row["model"] in {"O0", "O1"}
    }
    for row in rows:
        baseline = matched.get(row["refresh_interval"])
        row["delta_J&F_vs_O0_R1"] = (
            row["J&F"] - o0["J&F"] if o0 is not None else None
        )
        row["delta_J&F_vs_official_matched"] = (
            row["J&F"] - baseline["J&F"] if baseline is not None else None
        )
        row["delta_model_ms_vs_O0_R1"] = (
            row["model_mean_ms"] - o0["model_mean_ms"]
            if o0 is not None
            else None
        )
        row["model_speedup_vs_O0_R1"] = (
            o0["model_mean_ms"] / row["model_mean_ms"]
            if o0 is not None and row["model_mean_ms"] > 0
            else None
        )
        row["wall_speedup_vs_O0_R1"] = (
            o0["wall_ms_per_frame"] / row["wall_ms_per_frame"]
            if o0 is not None and row["wall_ms_per_frame"] > 0
            else None
        )
        if not row["offload_video_to_cpu"]:
            missing.append(
                f"protocol mismatch: {row['output']} did not use CPU video storage"
            )
    return rows, missing


def collect_quick(run_root: Path, epoch: int) -> tuple[list[dict], list[str]]:
    specs = [
        ("O0", 1, run_root / "quick_val/controls/O0/R1"),
        ("O1", 4, run_root / "quick_val/controls/O1/R4"),
    ]
    specs.extend(
        (
            model,
            4,
            run_root / model / f"quick_val/epoch_{epoch}/R4",
        )
        for model in EXPERIMENTS
    )
    rows = []
    missing = []
    for model, interval, output in specs:
        required = (output / "sav_eval.json", output / "pred/summary.json")
        absent = [str(path) for path in required if not path.is_file()]
        if absent:
            missing.extend(absent)
            continue
        rows.append(_row(model, interval, output))

    o0 = next((row for row in rows if row["model"] == "O0"), None)
    o1 = next((row for row in rows if row["model"] == "O1"), None)
    for row in rows:
        matched = o0 if row["refresh_interval"] == 1 else o1
        row["delta_J&F_vs_O0_R1"] = (
            row["J&F"] - o0["J&F"] if o0 is not None else None
        )
        row["delta_J&F_vs_official_matched"] = (
            row["J&F"] - matched["J&F"] if matched is not None else None
        )
        row["delta_model_ms_vs_O0_R1"] = (
            row["model_mean_ms"] - o0["model_mean_ms"] if o0 is not None else None
        )
        row["model_speedup_vs_O0_R1"] = (
            o0["model_mean_ms"] / row["model_mean_ms"]
            if o0 is not None and row["model_mean_ms"] > 0
            else None
        )
        row["wall_speedup_vs_O0_R1"] = (
            o0["wall_ms_per_frame"] / row["wall_ms_per_frame"]
            if o0 is not None and row["wall_ms_per_frame"] > 0
            else None
        )
        if not row["offload_video_to_cpu"]:
            missing.append(
                f"protocol mismatch: {row['output']} did not use CPU video storage"
            )
    return rows, missing


def collect_screen50(run_root: Path, epoch: int) -> tuple[list[dict], list[str]]:
    specs = [
        ("O0", 1, run_root / "screen50/controls/O0/R1"),
        ("W0", 1, run_root / "screen50/controls/W0/R1"),
        ("O1", 4, run_root / "screen50/controls/O1/R4"),
    ]
    specs.extend(
        (model, 4, run_root / model / f"screen50/epoch_{epoch}/R4")
        for model in EXPERIMENTS
    )
    rows = []
    missing = []
    for model, interval, output in specs:
        required = (output / "sav_eval.json", output / "pred/summary.json")
        absent = [str(path) for path in required if not path.is_file()]
        if absent:
            missing.extend(absent)
            continue
        rows.append(_row(model, interval, output))

    o0 = next((row for row in rows if row["model"] == "O0"), None)
    o1 = next((row for row in rows if row["model"] == "O1"), None)
    for row in rows:
        matched = o0 if row["refresh_interval"] == 1 else o1
        row["delta_J&F_vs_O0_R1"] = (
            row["J&F"] - o0["J&F"] if o0 is not None else None
        )
        row["delta_J&F_vs_official_matched"] = (
            row["J&F"] - matched["J&F"] if matched is not None else None
        )
        row["delta_model_ms_vs_O0_R1"] = (
            row["model_mean_ms"] - o0["model_mean_ms"] if o0 is not None else None
        )
        row["model_speedup_vs_O0_R1"] = (
            o0["model_mean_ms"] / row["model_mean_ms"]
            if o0 is not None and row["model_mean_ms"] > 0
            else None
        )
        row["wall_speedup_vs_O0_R1"] = (
            o0["wall_ms_per_frame"] / row["wall_ms_per_frame"]
            if o0 is not None and row["wall_ms_per_frame"] > 0
            else None
        )
        if not row["offload_video_to_cpu"]:
            missing.append(
                f"protocol mismatch: {row['output']} did not use CPU video storage"
            )
    return rows, missing


def _fmt(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def write_markdown(
    path: Path, rows: list[dict], missing: list[str], description: str
) -> None:
    lines = [
        "# SAM2.1-L two-clock validation comparison",
        "",
        description,
        "",
        "| Model | R | Skip | J&F | J | F | Refresh % | CPU video | Model ms | Rank P95 max ms | Wall ms/frame | ΔJ&F vs O0 R1 | ΔJ&F vs official matched | Model speedup | Wall speedup |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | R{refresh_interval} | {skip_frames} | {jf} | {j} | {f} | {refresh} | {storage} | {model_ms} | {p95} | {wall_ms} | {delta_o0} | {delta_matched} | {model_speedup}x | {wall_speedup}x |".format(
                **row,
                jf=_fmt(row["J&F"], 1),
                j=_fmt(row["J"], 1),
                f=_fmt(row["F"], 1),
                refresh=_fmt(100 * row["refresh_rate"], 1),
                storage="yes" if row["offload_video_to_cpu"] else "no/legacy",
                model_ms=_fmt(row["model_mean_ms"]),
                p95=_fmt(row["model_rank_p95_max_ms"]),
                wall_ms=_fmt(row["wall_ms_per_frame"]),
                delta_o0=_fmt(row["delta_J&F_vs_O0_R1"], 1),
                delta_matched=_fmt(row["delta_J&F_vs_official_matched"], 1),
                model_speedup=_fmt(row["model_speedup_vs_O0_R1"]),
                wall_speedup=_fmt(row["wall_speedup_vs_O0_R1"]),
            )
        )
    if missing:
        lines.extend(["", "Missing artifacts:", *[f"- `{item}`" for item in missing]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--quick-epoch", type=int)
    parser.add_argument("--screen50-epoch", type=int)
    args = parser.parse_args()
    if args.quick_epoch is not None and args.screen50_epoch is not None:
        parser.error("choose only one of --quick-epoch and --screen50-epoch")
    if args.screen50_epoch is not None:
        rows, missing = collect_screen50(args.run_root, args.screen50_epoch)
        protocol = f"screen50_hash_v1_epoch{args.screen50_epoch}"
        description = (
            "PROVISIONAL SCREEN: fixed 50-video hash sample of SA-V val, "
            f"checkpoint_{args.screen50_epoch}, R4 except O0/W0 R1. O0 is the "
            "pure official predictor; W0 is the two-clock wrapper identity audit; "
            "O1 changes only image-feature refresh."
        )
    elif args.quick_epoch is None:
        rows, missing = collect(args.run_root)
        protocol = "balanced_phase_v1"
        description = (
            "Phase-balanced full SA-V val. Latency is the matched single-stream mean; "
            "refresh rate is measured from actual encoder calls."
        )
    else:
        rows, missing = collect_quick(args.run_root, args.quick_epoch)
        protocol = f"quick10_hash_v1_epoch{args.quick_epoch}"
        description = (
            "PROVISIONAL SCREEN: fixed 10-video hash sample of SA-V val, "
            f"checkpoint_{args.quick_epoch}, R4 except O0 R1. Do not use for final "
            "checkpoint selection or paper claims."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "pass" if not missing else "incomplete",
        "protocol": protocol,
        "rows": rows,
        "missing": missing,
    }
    (args.out_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    fields = list(rows[0]) if rows else []
    with (args.out_dir / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    write_markdown(args.out_dir / "comparison.md", rows, missing, description)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "missing": missing}, indent=2))
    if args.require_complete and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
