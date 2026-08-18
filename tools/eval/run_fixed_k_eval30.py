#!/usr/bin/env python3
"""Evaluate one fixed-K model and matched controls on the registered 30 videos."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.eval.run_two_clock_eval30 import (
    Target,
    _complete,
    _make_gt_view,
    _prediction_complete,
    _run,
    _write_cohort,
)


def _targets(args: argparse.Namespace, interval: int) -> list[Target]:
    requested = args.target
    a_target = f"A{interval}"
    target_config = args.run_root / requested / "resolved_config.yaml"
    a1_config = args.run_root / "A1/resolved_config.yaml"
    targets = [
        Target("O0", "official", "O0", 1, args.official_checkpoint, None),
        Target(
            "O1",
            "official-reuse",
            "O1",
            interval,
            args.official_checkpoint,
            target_config,
        ),
        Target(
            "A1",
            "two-clock",
            "A",
            interval,
            args.run_root / "A1/checkpoints/last.pt",
            a1_config,
        ),
    ]
    if a_target != "A1":
        targets.append(
            Target(
                a_target,
                "two-clock",
                "A",
                interval,
                args.run_root / a_target / "checkpoints/last.pt",
                args.run_root / a_target / "resolved_config.yaml",
            )
        )
    if requested.startswith("D"):
        targets.append(
            Target(
                requested,
                "two-clock",
                "D",
                interval,
                args.run_root / requested / "checkpoints/last.pt",
                target_config,
            )
        )
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--sav-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--official-checkpoint", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument(
        "--target",
        required=True,
        choices=(
            "A1",
            "A4",
            "A8",
            "A12",
            "A16",
            "A20",
            "D4",
            "D8",
            "D12",
            "D16",
            "D20",
        ),
    )
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=250107256)
    parser.add_argument("--eval-processes", type=int, default=16)
    return parser.parse_args()


def _existing_world_sizes(out_root: Path) -> set[int]:
    sizes = set()
    for path in out_root.glob("*/R*/phase_*/pred/summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("status") == "pass":
            sizes.add(int(payload.get("world_size", 1)))
    return sizes


def main() -> None:
    args = parse_args()
    if not 1 <= args.world_size <= 4:
        raise ValueError("world size must be in [1, 4]")
    existing_world_sizes = _existing_world_sizes(args.out_root)
    if existing_world_sizes and existing_world_sizes != {args.world_size}:
        raise RuntimeError(
            "evaluation GPU count must stay fixed within one target: "
            f"existing={sorted(existing_world_sizes)}, requested={args.world_size}"
        )
    interval = int(args.target[1:])
    image_root = args.sav_root / "sav_val/JPEGImages_24fps"
    gt_root = args.sav_root / "sav_val/Annotations_6fps"
    cohort = args.out_root / f"cohort_{args.count}_seed{args.seed}.txt"
    gt_view = args.out_root / f"gt_{args.count}_seed{args.seed}/Annotations_6fps"
    videos = _write_cohort(
        args.sav_root / "sav_val/sav_val.txt", cohort, args.count, args.seed
    )
    _make_gt_view(gt_root, gt_view, videos)
    targets = _targets(args, interval)
    for target in targets:
        if not target.checkpoint.is_file():
            raise FileNotFoundError(target.checkpoint)
        if target.resolved_config is not None and not target.resolved_config.is_file():
            raise FileNotFoundError(target.resolved_config)

    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(
        [str(args.repo_root), str(args.sam2_root), env.get("PYTHONPATH", "")]
    )
    total = sum(target.interval for target in targets)
    completed = 0
    for target in targets:
        for phase in range(target.interval):
            completed += 1
            unit = args.out_root / target.name / f"R{target.interval}" / f"phase_{phase}"
            print(
                f"===== [{completed}/{total}] {target.name} "
                f"R{target.interval} phase {phase} =====",
                flush=True,
            )
            if _complete(unit, target, phase, videos):
                print(f"Skip verified completed unit: {unit}", flush=True)
                continue
            pred = unit / "pred"
            pred.mkdir(parents=True, exist_ok=True)
            inference = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={args.world_size}",
                str(args.repo_root / "tools/eval/run_edgetam_vos_dataset.py"),
                "--model-kind",
                target.kind,
                "--sam2-root",
                str(args.sam2_root),
                "--sam2-cfg",
                str(args.sam2_root / "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"),
                "--checkpoint",
                str(target.checkpoint),
                "--experiment",
                target.experiment,
                "--refresh-interval",
                str(target.interval),
                "--refresh-phase-mode",
                "fixed",
                "--refresh-phase",
                str(phase),
                "--max-feature-age",
                "19",
                "--image-root",
                str(image_root),
                "--input-mask-root",
                str(gt_root),
                "--video-list-file",
                str(cohort),
                "--out-dir",
                str(pred),
                "--per-obj-png-file",
                "--track-object-appearing-later-in-video",
                "--offload-video-to-cpu",
                "--device",
                "cuda",
            ]
            if target.resolved_config is not None:
                inference.extend(["--resolved-config", str(target.resolved_config)])
            if _prediction_complete(unit, target, phase, videos):
                print(f"Resume completed inference; compute metrics: {unit}", flush=True)
            else:
                _run(inference, env=env)
                _run(
                    [
                        sys.executable,
                        str(args.repo_root / "tools/eval/merge_vos_rank_summaries.py"),
                        "--run-dir",
                        str(pred),
                    ],
                    env=env,
                )
            _run(
                [
                    sys.executable,
                    str(args.repo_root / "tools/eval/run_sav_evaluator.py"),
                    "--evaluator",
                    str(args.sam2_root / "sav_dataset/sav_evaluator.py"),
                    "--gt-root",
                    str(gt_view),
                    "--pred-root",
                    str(pred),
                    "--out-json",
                    str(unit / "sav_eval.json"),
                    "--num-processes",
                    str(args.eval_processes),
                    "--strict",
                ],
                env=env,
            )
            _run(
                [
                    sys.executable,
                    str(args.repo_root / "tools/eval/evaluate_two_clock_age.py"),
                    "--sam2-root",
                    str(args.sam2_root),
                    "--gt-root",
                    str(gt_view),
                    "--pred-root",
                    str(pred),
                    "--refresh-interval",
                    str(target.interval),
                    "--refresh-phase-mode",
                    "fixed",
                    "--refresh-phase",
                    str(phase),
                    "--video-list-file",
                    str(cohort),
                    "--out-json",
                    str(unit / "age_metrics.json"),
                    "--out-csv",
                    str(unit / "age_metrics.csv"),
                ],
                env=env,
            )
            if not _complete(unit, target, phase, videos):
                raise RuntimeError(f"unit failed completion audit: {unit}")

    models = [target.name for target in targets]
    references = ["O0", "O1"]
    if args.target.startswith("D"):
        references.append(f"A{interval}")
    report = args.out_root / "report"
    summary = [
        sys.executable,
        str(args.repo_root / "tools/eval/summarize_two_clock_phase_sweep.py"),
        "--root",
        str(args.out_root),
        "--out-dir",
        str(report),
        "--models",
        ",".join(models),
        "--references",
        ",".join(references),
        "--bootstrap-samples",
        "10000",
        "--seed",
        str(args.seed),
    ]
    for target in targets:
        summary.extend(["--model-interval", f"{target.name}={target.interval}"])
    _run(summary, env=env)
    print(f"Detailed report: {report / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
