#!/usr/bin/env python3
"""Audit SAM2 and SAM3.1 Stage 1 training and full SA-V evaluation progress."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ExpectedRun:
    family: str
    queue: str
    name: str
    relative_dir: str
    target_steps: int
    gpus: int
    launcher: str


def expected_runs() -> list[ExpectedRun]:
    sam2_root = "sav_stage1_ablation_v2"
    sam31_root = "sam31_stage1_ablation_v1"
    rows: list[ExpectedRun] = []

    def add_sam2(queue: str, launcher: str, specs: list[tuple[str, int, int]]) -> None:
        for name, target_steps, gpus in specs:
            rows.append(
                ExpectedRun(
                    "sam2.1",
                    queue,
                    name,
                    f"{sam2_root}/{queue}/{name}",
                    target_steps,
                    gpus,
                    launcher,
                )
            )

    add_sam2(
        "8gpu_tv21_main",
        "scripts/company/20_queue_sav_stage1_ablation_8gpu.sh",
        [
            ("tv21_proj_sam21l_msehr", 126135, 8),
            ("tv21_proj_sam21l_msehr_cos025", 126135, 8),
            ("tv21_adapter_sam21l_msehr", 126135, 8),
        ],
    )
    add_sam2(
        "4gpu_size_scaling",
        "scripts/company/21_queue_sav_stage1_ablation_4gpu_size.sh",
        [
            ("tv11_proj_sam21l_msehr", 126135, 4),
            ("tv5_proj_sam21l_msehr", 63070, 4),
            ("tv11_proj_sam21l_msehr_cos025", 126135, 4),
        ],
    )
    add_sam2(
        "4gpu_loss_ablation",
        "scripts/company/22_queue_sav_stage1_ablation_4gpu_loss.sh",
        [
            ("tv5_proj_sam21l_msehr_cos025", 63070, 4),
            ("tv21_proj_sam21l_image_only", 252265, 4),
            ("tv21_proj_sam21l_hr025", 252265, 4),
        ],
    )
    add_sam2(
        "4gpu_adapter_teacher",
        "scripts/company/23_queue_sav_stage1_ablation_4gpu_adapter_teacher.sh",
        [
            ("tv21_proj_sam21l_msehr_l1_025", 252265, 4),
            ("tv21_adapter_sam21l_msehr_cos025", 252265, 4),
            ("tv21_proj_sam21bplus_msehr", 252265, 4),
        ],
    )
    add_sam2(
        "4gpu_extra_adapter_cos",
        "scripts/company/24_queue_sav_stage1_ablation_4gpu_extra.sh",
        [
            ("tv11_adapter_sam21l_msehr", 126135, 4),
            ("tv5_adapter_sam21l_msehr", 63070, 4),
            ("tv21_proj_sam21l_msehr_cos1", 252265, 4),
        ],
    )

    sam31_queues = {
        "node1_cosine": (
            "scripts/company/27_queue_sam31_4gpu_cosine.sh",
            [
                "n1_cos000_adapter_ft_w2k",
                "n1_cos025_adapter_ft_w2k",
                "n1_cos100_adapter_ft_w2k",
            ],
        ),
        "node2_interface": (
            "scripts/company/28_queue_sam31_4gpu_interface.sh",
            [
                "n2_projection_cos025_ft_w2k",
                "n2_adapter_cos025_frozen",
                "n2_adapter_cos025_ft_w0",
            ],
        ),
        "node3_relations": (
            "scripts/company/29_queue_sam31_4gpu_relations.sh",
            [
                "n3_cos150_adapter_ft_w2k",
                "n3_relation010_adapter_ft_w2k",
                "n3_cos025_relation010_adapter_ft_w2k",
            ],
        ),
    }
    for queue, (launcher, names) in sam31_queues.items():
        for name in names:
            rows.append(
                ExpectedRun(
                    "sam3.1",
                    queue,
                    name,
                    f"{sam31_root}/{queue}/{name}",
                    252265,
                    4,
                    launcher,
                )
            )
    return rows


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_checkpoint_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"valid": False, "error": "missing checkpoint"}
    try:
        try:
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=False, mmap=True
            )
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - audit must report corrupt checkpoints
        return {"valid": False, "error": str(exc), "path": str(path)}
    if not isinstance(checkpoint, dict):
        return {"valid": False, "error": "checkpoint is not a dictionary", "path": str(path)}
    args = checkpoint.get("args")
    args = args if isinstance(args, dict) else {}
    step = checkpoint.get("step", checkpoint.get("global_step"))
    epoch = checkpoint.get("epoch")
    model_ready = any(key in checkpoint for key in ("model_state", "model", "state_dict"))
    optimizer_ready = any(key in checkpoint for key in ("optimizer_state", "optimizer"))
    result = {
        "valid": step is not None and model_ready,
        "path": str(path),
        "bytes": path.stat().st_size,
        "mtime": path.stat().st_mtime,
        "step": int(step) if step is not None else None,
        "epoch": float(epoch) if epoch is not None else None,
        "checkpoint_max_steps": args.get("max_steps"),
        "model_ready": model_ready,
        "optimizer_ready": optimizer_ready,
        "wandb_run_id": checkpoint.get("wandb_run_id"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "manifest": args.get("manifest"),
        "model_name": args.get("model_name"),
        "adapter_mode": args.get("adapter_mode"),
        "wandb_project": args.get("wandb_project"),
        "wandb_name": args.get("wandb_name"),
    }
    del checkpoint
    gc.collect()
    return result


def checkpoint_candidates(run_dir: Path) -> list[Path]:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return []
    last = checkpoint_dir / "last.pt"
    best = checkpoint_dir / "best.pt"
    if last.is_file():
        return [last]
    if best.is_file():
        return [best]
    candidates = list(checkpoint_dir.glob("*.pt"))

    def rank(path: Path) -> tuple[int, float]:
        numbers = re.findall(r"\d+", path.stem)
        return (int(numbers[-1]) if numbers else -1, path.stat().st_mtime)

    return [max(candidates, key=rank)] if candidates else []


def read_wandb_metadata(run_dir: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    metadata = read_json(run_dir / "wandb_run.json")
    run_id = checkpoint.get("wandb_run_id") or metadata.get("run_id")
    if not run_id:
        wandb_root = run_dir / "wandb" / "wandb"
        matches = sorted(wandb_root.glob("*run-*-*")) if wandb_root.is_dir() else []
        if matches:
            run_id = matches[-1].name.rsplit("-", 1)[-1]
    return {
        "run_id": run_id,
        "project": checkpoint.get("wandb_project") or metadata.get("project"),
        "name": checkpoint.get("wandb_name") or metadata.get("name"),
        "url": metadata.get("url"),
    }


def split_video_ids(sav_root: Path, split: str) -> set[str]:
    path = sav_root / split / f"{split}.txt"
    if not path.is_file():
        return set()
    return {
        Path(line.strip()).stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def image_benchmark_video_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["video"] for row in csv.DictReader(handle) if row.get("video")}


def audit_full_split_evaluation(
    run_dir: Path,
    run_name: str,
    split: str,
    expected_videos: set[str],
    best_checkpoint: Path,
) -> dict[str, Any]:
    benchmark_root = run_dir / f"{split}_box_benchmark"
    image_dir = benchmark_root / "image" / run_name / "box"
    vos_dir = benchmark_root / "vos" / run_name / "box"
    image_summary_path = image_dir / "summary.json"
    image_rows_path = image_dir / "per_object_metrics.csv"
    vos_summary_path = vos_dir / "run_summary.json"
    vos_eval_path = vos_dir / "eval_summary.json"
    metrics_path = benchmark_root / "metrics.csv"
    image_summary = read_json(image_summary_path)
    vos_summary = read_json(vos_summary_path)
    vos_eval = read_json(vos_eval_path)
    image_videos = image_benchmark_video_ids(image_rows_path)
    vos_videos = set(vos_summary.get("video_names", []))
    reasons = []
    if not expected_videos:
        reasons.append(f"missing {split}.txt or empty split list")
    if image_summary.get("status") != "pass":
        reasons.append("image summary missing or failed")
    elif Path(str(image_summary.get("checkpoint", ""))).resolve() != best_checkpoint.resolve():
        reasons.append("image benchmark did not use this run's best.pt")
    if image_videos != expected_videos:
        reasons.append(
            f"image coverage {len(image_videos)}/{len(expected_videos)} videos"
        )
    if vos_summary.get("status") != "pass":
        reasons.append("VOS run summary missing or failed")
    elif Path(str(vos_summary.get("checkpoint", ""))).resolve() != best_checkpoint.resolve():
        reasons.append("VOS benchmark did not use this run's best.pt")
    if vos_videos != expected_videos:
        reasons.append(f"VOS coverage {len(vos_videos)}/{len(expected_videos)} videos")
    if vos_eval.get("status") != "pass" or not vos_eval.get("metrics"):
        reasons.append("VOS evaluator metrics missing or failed")
    if not metrics_path.is_file():
        reasons.append("combined metrics.csv missing")
    artifact_paths = [image_summary_path, image_rows_path, vos_summary_path, vos_eval_path, metrics_path]
    if all(path.is_file() for path in artifact_paths):
        oldest_artifact = min(path.stat().st_mtime for path in artifact_paths)
        if oldest_artifact < best_checkpoint.stat().st_mtime:
            reasons.append("evaluation artifacts are older than best.pt")
    return {
        "complete": not reasons,
        "expected_videos": len(expected_videos),
        "image_videos": len(image_videos),
        "vos_videos": len(vos_videos),
        "mIoU": image_summary.get("mIoU"),
        "AP": image_summary.get("AP"),
        "J&F": vos_eval.get("metrics", {}).get("J&F"),
        "J": vos_eval.get("metrics", {}).get("J"),
        "F": vos_eval.get("metrics", {}).get("F"),
        "reasons": reasons,
        "benchmark_root": str(benchmark_root),
    }


def choose_run_dir(roots: list[Path], relative_dir: str) -> tuple[Path, list[str]]:
    candidates = [root / relative_dir for root in roots if (root / relative_dir).is_dir()]
    if not candidates:
        return roots[0] / relative_dir, []
    if len(candidates) == 1:
        return candidates[0], []
    ranked = []
    for candidate in candidates:
        checkpoints = checkpoint_candidates(candidate)
        metadata = load_checkpoint_metadata(checkpoints[0] if checkpoints else None)
        ranked.append((int(metadata.get("step") or -1), candidate))
    selected = max(ranked, key=lambda item: item[0])[1]
    return selected, [str(candidate) for candidate in candidates if candidate != selected]


def classify_registered_run(
    spec: ExpectedRun,
    run_dir: Path,
    duplicates: list[str],
    val_videos: set[str],
    test_videos: set[str],
) -> dict[str, Any]:
    last_path = run_dir / "checkpoints" / "last.pt"
    best_path = run_dir / "checkpoints" / "best.pt"
    selected_path = last_path if last_path.is_file() else best_path if best_path.is_file() else None
    checkpoint = load_checkpoint_metadata(selected_path)
    best = load_checkpoint_metadata(best_path if best_path.is_file() else None)
    step = int(checkpoint.get("step") or 0)
    training_complete = bool(checkpoint.get("valid")) and step >= spec.target_steps
    best_ready = bool(best.get("valid"))
    wandb = read_wandb_metadata(run_dir, checkpoint)
    resumable = (
        bool(checkpoint.get("valid"))
        and bool(checkpoint.get("optimizer_ready"))
        and selected_path == last_path
        and step < spec.target_steps
    )
    if training_complete and best_ready:
        val_eval = audit_full_split_evaluation(run_dir, spec.name, "sav_val", val_videos, best_path)
        test_eval = audit_full_split_evaluation(run_dir, spec.name, "sav_test", test_videos, best_path)
    else:
        val_eval = {"complete": False, "reasons": ["training or best checkpoint incomplete"]}
        test_eval = {"complete": False, "reasons": ["training or best checkpoint incomplete"]}
    if training_complete and best_ready and val_eval["complete"] and test_eval["complete"]:
        status = "complete"
    elif training_complete and best_ready:
        status = "needs_full_eval"
    elif training_complete:
        status = "needs_final_validation"
    elif resumable and wandb.get("run_id"):
        status = "resumable"
    elif resumable:
        status = "resumable_missing_wandb_id"
    elif selected_path is None:
        status = "missing"
    else:
        status = "invalid"
    return {
        **asdict(spec),
        "registered": True,
        "run_dir": str(run_dir),
        "duplicate_run_dirs": duplicates,
        "status": status,
        "step": step,
        "progress_pct": min(100.0 * step / spec.target_steps, 100.0),
        "training_complete": training_complete,
        "best_ready": best_ready,
        "resumable": resumable,
        "last_checkpoint": str(last_path) if last_path.is_file() else None,
        "best_checkpoint": str(best_path) if best_path.is_file() else None,
        "checkpoint": checkpoint,
        "wandb": wandb,
        "full_val": val_eval,
        "full_test": test_eval,
    }


def discover_run_dirs(roots: list[Path]) -> set[Path]:
    discovered = set()
    for root in roots:
        if not root.is_dir():
            continue
        for checkpoint_dir in root.rglob("checkpoints"):
            if checkpoint_dir.is_dir() and any(checkpoint_dir.glob("*.pt")):
                discovered.add(checkpoint_dir.parent)
    return discovered


def classify_unregistered_run(
    run_dir: Path, val_videos: set[str], test_videos: set[str]
) -> dict[str, Any]:
    candidates = checkpoint_candidates(run_dir)
    checkpoint_path = candidates[0] if candidates else None
    checkpoint = load_checkpoint_metadata(checkpoint_path)
    step = int(checkpoint.get("step") or 0)
    target = int(checkpoint.get("checkpoint_max_steps") or 0)
    wandb = read_wandb_metadata(run_dir, checkpoint)
    family = "sam3.1" if "sam31" in str(run_dir).lower() else "sam2.1_or_legacy"
    training_complete = bool(target and step >= target)
    best_path = run_dir / "checkpoints" / "best.pt"
    best_ready = best_path.is_file() and bool(load_checkpoint_metadata(best_path).get("valid"))
    resumable = bool(
        checkpoint.get("optimizer_ready") and step and (not target or step < target)
    )
    if training_complete and best_ready:
        val_eval = audit_full_split_evaluation(
            run_dir, run_dir.name, "sav_val", val_videos, best_path
        )
        test_eval = audit_full_split_evaluation(
            run_dir, run_dir.name, "sav_test", test_videos, best_path
        )
    else:
        val_eval = {"complete": False, "reasons": ["training or best checkpoint incomplete"]}
        test_eval = {"complete": False, "reasons": ["training or best checkpoint incomplete"]}
    if training_complete and best_ready and val_eval["complete"] and test_eval["complete"]:
        status = "unregistered_complete"
    elif training_complete and best_ready:
        status = "unregistered_needs_full_eval"
    elif training_complete:
        status = "unregistered_needs_final_validation"
    elif resumable and wandb.get("run_id"):
        status = "unregistered_resumable"
    elif resumable:
        status = "unregistered_resumable_missing_wandb_id"
    else:
        status = "unregistered_review"
    return {
        "family": family,
        "queue": "unregistered",
        "name": run_dir.name,
        "relative_dir": "",
        "target_steps": target,
        "gpus": None,
        "launcher": None,
        "registered": False,
        "run_dir": str(run_dir),
        "duplicate_run_dirs": [],
        "status": status,
        "step": step,
        "progress_pct": 100.0 * step / target if target else None,
        "training_complete": training_complete,
        "best_ready": best_ready,
        "resumable": resumable,
        "last_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "best_checkpoint": str(best_path) if best_path.is_file() else None,
        "checkpoint": checkpoint,
        "wandb": wandb,
        "full_val": val_eval,
        "full_test": test_eval,
    }


def csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": row["family"],
        "queue": row["queue"],
        "name": row["name"],
        "registered": row["registered"],
        "status": row["status"],
        "step": row["step"],
        "target_steps": row["target_steps"],
        "progress_pct": f"{row['progress_pct']:.2f}" if row["progress_pct"] is not None else "",
        "training_complete": row["training_complete"],
        "best_ready": row["best_ready"],
        "full_val_complete": row["full_val"]["complete"],
        "full_test_complete": row["full_test"]["complete"],
        "resumable": row["resumable"],
        "wandb_run_id": row["wandb"].get("run_id"),
        "wandb_project": row["wandb"].get("project"),
        "run_dir": row["run_dir"],
        "launcher": row["launcher"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--sav-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--no-discover", action="store_true")
    args = parser.parse_args()

    roots = [root.resolve() for root in args.runs_root if root.is_dir()]
    if not roots:
        raise SystemExit("None of the supplied run roots exist")
    val_videos = split_video_ids(args.sav_root, "sav_val")
    test_videos = split_video_ids(args.sav_root, "sav_test")
    if not val_videos or not test_videos:
        raise SystemExit(f"Missing complete sav_val/sav_test lists under {args.sav_root}")

    rows = []
    registered_dirs = {
        (root / spec.relative_dir).resolve()
        for root in roots
        for spec in expected_runs()
    }
    for spec in expected_runs():
        run_dir, duplicates = choose_run_dir(roots, spec.relative_dir)
        rows.append(classify_registered_run(spec, run_dir, duplicates, val_videos, test_videos))
    if not args.no_discover:
        for run_dir in sorted(discover_run_dirs(roots)):
            if run_dir.resolve() not in registered_dirs:
                rows.append(classify_unregistered_run(run_dir, val_videos, test_videos))

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs_roots": [str(root) for root in roots],
        "sav_root": str(args.sav_root),
        "full_val_videos": len(val_videos),
        "full_test_videos": len(test_videos),
        "status_counts": status_counts,
        "definition_of_complete": (
            "target training steps reached, best.pt exists, and best.pt has full sav_val and "
            "sav_test image/VOS box-prompt metrics"
        ),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    flat_rows = [csv_row(row) for row in rows]
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(f"CSV: {args.out_csv}")
    print(f"JSON: {args.out_json}")


if __name__ == "__main__":
    main()
