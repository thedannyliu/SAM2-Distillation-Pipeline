#!/usr/bin/env python3
"""Run the registered 30-video current-model audit sequentially on one GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Target:
    name: str
    kind: str
    experiment: str
    interval: int
    checkpoint: Path
    resolved_config: Path | None


def _run(command: list[str], *, env: dict[str, str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def _write_cohort(source: Path, output: Path, count: int, seed: int) -> list[str]:
    names = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    ranked = sorted(
        names, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).digest()
    )
    selected = ranked[:count]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return selected


def _make_gt_view(source: Path, output: Path, videos: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for video in videos:
        target = source / video
        if not target.is_dir():
            raise FileNotFoundError(target)
        link = output / video
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"wrong GT link: {link}")
        elif link.exists():
            raise RuntimeError(f"GT view entry is not a symlink: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)


def _prediction_complete(
    unit: Path, target: Target, phase: int, videos: list[str]
) -> bool:
    summary_path = unit / "pred/summary.json"
    if not summary_path.is_file():
        return False
    timing = json.loads(summary_path.read_text())
    build = timing.get("build", {})
    png_count = sum(1 for _ in (unit / "pred").rglob("*.png"))
    return bool(
        timing.get("status") == "pass"
        and timing.get("model_kind") == target.kind
        and timing.get("two_clock_refresh_phase_mode") == "fixed"
        and timing.get("two_clock_refresh_phase") == phase
        and set(timing.get("video_names", [])) == set(videos)
        and build.get("checkpoint") == str(target.checkpoint)
        and build.get("experiment") == target.experiment
        and build.get("refresh_interval") == target.interval
        and png_count == timing.get("num_prediction_pngs")
    )


def _complete(unit: Path, target: Target, phase: int, videos: list[str]) -> bool:
    if not _prediction_complete(unit, target, phase, videos):
        return False
    required = (unit / "sav_eval.json", unit / "age_metrics.json")
    if not all(path.is_file() for path in required):
        return False
    sav, age = (json.loads(path.read_text()) for path in required)
    return bool(
        sav.get("status") == "pass"
        and age.get("status") == "pass"
        and age.get("protocol") == "fixed_phase_v1"
        and age.get("refresh_interval") == target.interval
        and age.get("refresh_phase") == phase
        and age.get("videos") == len(videos)
    )


def _targets(args: argparse.Namespace) -> list[Target]:
    old = args.old_run_root
    official_config = old / "O2/resolved_config.yaml"
    targets = [
        Target("O0", "official", "O0", 1, args.official_checkpoint, None),
        Target("W0", "two-clock", "O0", 1, args.official_checkpoint, official_config),
        Target("O1", "official-reuse", "O1", 4, args.official_checkpoint, official_config),
    ]
    targets.extend(
        Target(
            name,
            "two-clock",
            name,
            4,
            old / name / "checkpoints/checkpoint_1.pt",
            old / name / "resolved_config.yaml",
        )
        for name in ("O2", "A", "B", "C", "D", "E")
    )
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--sav-root", required=True, type=Path)
    parser.add_argument("--old-run-root", required=True, type=Path)
    parser.add_argument("--official-checkpoint", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=250107256)
    parser.add_argument("--eval-processes", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_root = args.sav_root / "sav_val/JPEGImages_24fps"
    gt_root = args.sav_root / "sav_val/Annotations_6fps"
    cohort = args.out_root / f"cohort_{args.count}_seed{args.seed}.txt"
    gt_view = args.out_root / f"gt_{args.count}_seed{args.seed}/Annotations_6fps"
    videos = _write_cohort(args.sav_root / "sav_val/sav_val.txt", cohort, args.count, args.seed)
    _make_gt_view(gt_root, gt_view, videos)
    targets = _targets(args)
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
                f"===== [{completed}/{total}] {target.name} R{target.interval} phase {phase} =====",
                flush=True,
            )
            if _complete(unit, target, phase, videos):
                print(f"Skip verified completed unit: {unit}", flush=True)
                continue
            pred = unit / "pred"
            pred.mkdir(parents=True, exist_ok=True)
            inference = [
                sys.executable,
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
                inference.extend(
                    [
                        "--resolved-config",
                        str(target.resolved_config),
                    ]
                )
            if _prediction_complete(unit, target, phase, videos):
                print(f"Resume completed inference; compute metrics: {unit}", flush=True)
            else:
                _run(inference, env=env)
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

    report = args.out_root / "report"
    _run(
        [
            sys.executable,
            str(args.repo_root / "tools/eval/compare_vos_prediction_trees.py"),
            "--left",
            str(args.out_root / "O0/R1/phase_0/pred"),
            "--right",
            str(args.out_root / "W0/R1/phase_0/pred"),
            "--out-json",
            str(report / "o0_w0_identity.json"),
        ],
        env=env,
    )
    _run(
        [
            sys.executable,
            str(args.repo_root / "tools/eval/summarize_two_clock_phase_sweep.py"),
            "--root",
            str(args.out_root),
            "--out-dir",
            str(report),
            "--bootstrap-samples",
            "10000",
            "--seed",
            str(args.seed),
        ],
        env=env,
    )
    print(f"Detailed report: {report / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
