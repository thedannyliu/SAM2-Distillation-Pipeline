#!/usr/bin/env python3
"""Summarize registered and discovered SAM2 experiments in one CSV."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from audit_stage1_run_progress import expected_runs as expected_stage1_runs


FIELDS = [
    "family",
    "suite",
    "experiment",
    "stage",
    "registered",
    "status",
    "next_action",
    "progress_pct",
    "checkpoint_epoch",
    "checkpoint_step",
    "target_unit",
    "target_value",
    "wandb_project",
    "wandb_run_id",
    "val_status",
    "val_mIoU",
    "val_AP",
    "val_image_latency_s",
    "val_J&F",
    "val_video_latency_s",
    "test_status",
    "test_mIoU",
    "test_AP",
    "test_image_latency_s",
    "test_J&F",
    "test_video_latency_s",
    "issues",
    "run_dir",
    "relative_dir",
    "duplicate_run_dirs",
]


@dataclass(frozen=True)
class Expected:
    relative_dir: str
    family: str
    experiment: str
    stage: str = ""
    target_unit: str = ""
    target_value: int | None = None


def expected_experiments() -> list[Expected]:
    rows = [
        Expected(
            spec.relative_dir,
            spec.family,
            spec.name,
            target_unit="step",
            target_value=spec.target_steps,
        )
        for spec in expected_stage1_runs()
    ]
    for name in (
        "repvit_m09_proj_sam21l_msehr_cos025_l1010",
        "repvit_m23_proj_sam21l_msehr_cos025_l1010",
    ):
        rows.append(Expected(f"repvit_stage1_v1/{name}", "sam2.1_repvit", name))
    task_suites = {
        "sam2_task_finetune_tv21_v1": (
            ("stage1_encoder_task_2ep", 2),
            ("stage2_encoder_decoder_task_2ep", 2),
            ("stage3_encoder_decoder_memory_task_1ep", 1),
        ),
        "sam2_task_finetune_tv21_v2": (
            ("stage1_encoder_task_2ep_v2", 2),
            ("stage2_decoder_only_task_1ep_v2", 1),
            ("stage3_encoder_decoder_memory_task_1ep_v2", 1),
        ),
    }
    for suite, stages in task_suites.items():
        for stage, epochs in stages:
            rows.append(
                Expected(
                    f"{suite}/{stage}",
                    "sam2.1_task",
                    stage,
                    stage,
                    "epoch",
                    epochs,
                )
            )
    for variant in (
        "decoder_lr2e7",
        "decoder_lr5e7",
        "decoder_lr2e6",
        "encdec_low_frozenbn",
        "encdec_low_trainbn",
        "decoder_lr5e7_boxonly",
    ):
        stage = f"mask_{variant}"
        rows.append(
            Expected(
                f"sam2_mask_finetune_ablation_v1/{variant}/{stage}",
                "sam2.1_task",
                variant,
                stage,
                "epoch",
                1,
            )
        )
    for variant in (
        "A00_e2e_t4_box1_control",
        "A01_e2e_t4_box0",
        "A02_e2e_t4_official_prompt",
        "A03_decmem_t4",
        "A04_memory_t4",
        "A05_e2e_t8",
        "A06_e2e_t8_s4_t16_hard",
        "A07_e2e_t4_warmup5",
        "A08_e2e_t4_gb8",
        "A09_e2e_t4_hard50x2",
        "A10_e2e_t4_box0_imgkd",
        "A11_e2e_t4_box0_imgmemkd",
    ):
        stage = "refine_t16" if variant == "A06_e2e_t8_s4_t16_hard" else "main"
        rows.append(
            Expected(
                f"sam2_mask_finetune_ablation_v2/{variant}/{stage}",
                "sam2.1_task",
                variant,
                stage,
                "epoch",
                1,
            )
        )
    for variant in (
        "M0_sam2_mem4",
        "M1_sam2_mem2",
        "M2a_edgetam_hybrid2_official",
        "M2b_edgetam_hybrid2_current",
        "R0_edgetam_e2e_t4_task",
        "R1_edgetam_e2e_t4_imgkd",
        "R2_edgetam_e2e_t4_imgmemkd",
        "R3_edgetam_e2e_t8_imgmemkd",
    ):
        rows.append(
            Expected(
                f"edgetam_memory_ablation_v1/{variant}/main",
                "sam2.1_edgetam",
                variant,
                "main",
                "epoch",
                1,
            )
        )
    for variant, epochs in (
        ("C0_coherent_m0mem_align", 1),
        ("C1_partial_m0mem_align", 1),
        ("C2_coherent_m0mem_joint2ep", 2),
        ("C3_coherent_m0mem_staged", 1),
    ):
        rows.append(
            Expected(
                f"edgetam_memory_recovery_v2/{variant}/main",
                "sam2.1_edgetam",
                variant,
                "main",
                "epoch",
                epochs,
            )
        )
    rows.append(
        Expected(
            "sam31_stage1/tv21m_adapter_mse_cos025_5ep_v1",
            "sam3.1",
            "tv21m_adapter_mse_cos025_5ep_v1",
        )
    )
    return rows


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_checkpoint(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"valid": False, "error": "missing"}
    try:
        try:
            payload = torch.load(
                path, map_location="cpu", weights_only=False, mmap=True
            )
        except TypeError:
            payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - corrupt runs belong in the report
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {"valid": False, "error": "checkpoint is not a dictionary"}
    args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    steps = payload.get("steps")
    task_step = steps.get("train") if isinstance(steps, dict) else None
    step = payload.get("step", payload.get("global_step", task_step))
    epoch = payload.get("epoch")
    model_ready = any(
        key in payload
        for key in ("model", "model_state", "state_dict", "task_model_state")
    )
    result = {
        "valid": model_ready,
        "error": "" if model_ready else "model state missing",
        "step": int(step) if step is not None else None,
        "epoch": int(epoch) if epoch is not None else None,
        "max_steps": args.get("max_steps"),
        "wandb_project": args.get("wandb_project"),
        "wandb_run_id": payload.get("wandb_run_id"),
    }
    del payload
    gc.collect()
    return result


def checkpoint_paths(run_dir: Path, family: str) -> tuple[Path | None, Path | None]:
    root = run_dir / "checkpoints"
    if not root.is_dir():
        return None, None
    training = next(
        (
            path
            for name in ("last.pt", "checkpoint.pt", "best.pt")
            if (path := root / name).is_file()
        ),
        None,
    )
    if family == "sam2.1_task":
        final = root / "stage.pt"
    elif (root / "last.pt").is_file():
        final = root / "best.pt"
    else:
        final = next(
            (
                path
                for name in ("best.pt", "checkpoint.pt", "stage.pt")
                if (path := root / name).is_file()
            ),
            training,
        )
    return training, final


def read_metrics(run_dir: Path, split: str) -> dict[str, Any]:
    path = run_dir / f"{split}_box_benchmark/metrics.csv"
    rows = {}
    if path.is_file():
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = {
                    row.get("mode", ""): row
                    for row in csv.DictReader(handle)
                    if row.get("prompt", "box") == "box"
                }
        except (OSError, csv.Error):
            rows = {}
    image = rows.get("image", {})
    video = rows.get("video_tracking", {})
    image_pass = image.get("status") == "pass"
    video_pass = video.get("status") == "pass"
    status = "pass" if image_pass and video_pass else "missing"
    if path.is_file() and status != "pass":
        status = "partial" if image_pass or video_pass else "failed"
    return {
        "status": status,
        "path": path,
        "mtime": path.stat().st_mtime if path.is_file() else 0.0,
        "mIoU": image.get("mIoU", ""),
        "AP": image.get("AP", ""),
        "image_latency": image.get("mean_total_object_seconds", ""),
        "J&F": video.get("J&F", ""),
        "video_latency": video.get("sec_per_video", ""),
    }


def infer_identity(relative: str) -> tuple[str, str, str, str]:
    parts = Path(relative).parts
    suite = parts[0] if parts else ""
    family = "sam2.1"
    if "sam31" in relative.lower():
        family = "sam3.1"
    elif "repvit" in relative.lower():
        family = "sam2.1_repvit"
    elif "task_finetune" in relative or "mask_finetune" in relative:
        family = "sam2.1_task"
    if "mask_finetune" in suite and len(parts) >= 3:
        return family, suite, parts[1], parts[2]
    if "task_finetune" in suite and len(parts) >= 2:
        return family, suite, parts[1], parts[1]
    return family, suite, parts[-1] if parts else "", ""


def discover(roots: list[Path]) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        candidates = {
            path.parent
            for path in root.rglob("checkpoints")
            if path.is_dir() and any(path.glob("*.pt"))
        }
        for split in ("sav_val", "sav_test"):
            candidates.update(
                path.parents[1]
                for path in root.rglob(f"{split}_box_benchmark/metrics.csv")
            )
        for run_dir in candidates:
            relative = str(run_dir.relative_to(root))
            if run_dir.name.startswith("shared_stage1"):
                continue
            if relative.endswith("A06_e2e_t8_s4_t16_hard/main"):
                continue
            found[relative].append(run_dir)
    return found


def wandb_metadata(run_dir: Path, checkpoint: dict[str, Any]) -> dict[str, str]:
    metadata = {}
    for path in (run_dir / "wandb/wandb_run.json", run_dir / "wandb_run.json"):
        if path.is_file():
            metadata = read_json(path)
            break
    return {
        "project": str(
            metadata.get("project") or checkpoint.get("wandb_project") or ""
        ),
        "run_id": str(
            metadata.get("run_id") or checkpoint.get("wandb_run_id") or ""
        ),
    }


def superseded_reason(relative: str) -> str:
    suite = Path(relative).parts[0]
    if suite.startswith("stage1_online_teacher_sav000_018_"):
        return "superseded by sav_stage1_ablation_v2"
    if suite == "sam31_stage1":
        return "superseded by sam31_stage1_ablation_v1"
    return ""


ACTIONS = {
    "complete": "none",
    "superseded": "none; retained as historical artifact",
    "not_started": "start experiment",
    "checkpoint_invalid": "inspect or restore checkpoint",
    "training_failed": "inspect logs and resume with the same W&B run",
    "training_incomplete": "resume training with the same W&B run",
    "training_state_unknown": "inspect checkpoint target and training logs",
    "final_checkpoint_incomplete": "finish validation/export of final checkpoint",
    "val_incomplete": "run full sav_val image and VOS evaluation",
    "test_incomplete": "run full sav_test image and VOS evaluation",
    "evaluation_stale": "rerun sav_val and sav_test for the current checkpoint",
    "finalization_incomplete": "rerun evaluation finalization and W&B sync",
}


def inspect_run(
    run_dir: Path, relative: str, expected: Expected | None
) -> dict[str, Any]:
    inferred = infer_identity(relative)
    family = expected.family if expected else inferred[0]
    suite = Path(relative).parts[0]
    experiment = expected.experiment if expected else inferred[2]
    stage = expected.stage if expected else inferred[3]
    training_path, final_path = checkpoint_paths(run_dir, family)
    checkpoint = load_checkpoint(training_path)
    final_checkpoint = (
        checkpoint if final_path == training_path else load_checkpoint(final_path)
    )
    status_json = read_json(run_dir / "training_status.json")
    target_unit = expected.target_unit if expected else ""
    target_value = expected.target_value if expected else None
    if not target_unit and checkpoint.get("max_steps"):
        target_unit = "step"
        target_value = int(checkpoint["max_steps"])
    current = checkpoint.get("epoch") if target_unit == "epoch" else checkpoint.get("step")
    if target_value is not None and current is not None:
        training_complete = int(current) >= target_value
    else:
        training_complete = target_value is None and status_json.get("status") == "complete"
    progress = (
        min(100.0 * float(current) / target_value, 100.0)
        if current is not None and target_value
        else None
    )
    val = read_metrics(run_dir, "sav_val")
    test = read_metrics(run_dir, "sav_test")
    issues = []
    if status_json.get("status") == "failed":
        status = "training_failed"
    elif not checkpoint.get("valid"):
        status = "checkpoint_invalid"
        issues.append(str(checkpoint.get("error", "invalid checkpoint")))
    elif not training_complete:
        status = "training_incomplete" if target_value else "training_state_unknown"
    elif not final_checkpoint.get("valid"):
        status = "final_checkpoint_incomplete"
    elif val["status"] != "pass":
        status = "val_incomplete"
        issues.append(f"sav_val={val['status']}")
    elif test["status"] != "pass":
        status = "test_incomplete"
        issues.append(f"sav_test={test['status']}")
    else:
        status = "complete"
    if reason := superseded_reason(relative):
        status = "superseded"
        issues.append(reason)
    reference = next(
        (
            path
            for path in (
                run_dir / "checkpoints/best.pt",
                run_dir / "checkpoints/checkpoint.pt",
                training_path,
            )
            if path is not None and path.is_file()
        ),
        None,
    )
    if status == "complete" and reference:
        if min(val["mtime"], test["mtime"]) < reference.stat().st_mtime:
            status = "evaluation_stale"
        elif (run_dir / ".full_eval_required").is_file():
            status = "finalization_incomplete"
    wandb = wandb_metadata(run_dir, checkpoint)
    row = {
        "family": family,
        "suite": suite,
        "experiment": experiment,
        "stage": stage,
        "registered": bool(expected),
        "status": status,
        "next_action": ACTIONS[status],
        "progress_pct": "" if progress is None else f"{progress:.2f}",
        "checkpoint_epoch": checkpoint.get("epoch", ""),
        "checkpoint_step": checkpoint.get("step", ""),
        "target_unit": target_unit,
        "target_value": target_value or "",
        "wandb_project": wandb["project"],
        "wandb_run_id": wandb["run_id"],
        "issues": "; ".join(issues),
        "run_dir": str(run_dir),
        "relative_dir": relative,
        "duplicate_run_dirs": "",
        "_mtime": max(
            training_path.stat().st_mtime if training_path else 0,
            val["mtime"],
            test["mtime"],
        ),
    }
    for prefix, metrics in (("val", val), ("test", test)):
        row.update(
            {
                f"{prefix}_status": metrics["status"],
                f"{prefix}_mIoU": metrics["mIoU"],
                f"{prefix}_AP": metrics["AP"],
                f"{prefix}_image_latency_s": metrics["image_latency"],
                f"{prefix}_J&F": metrics["J&F"],
                f"{prefix}_video_latency_s": metrics["video_latency"],
            }
        )
    return row


def missing_row(expected: Expected, root: Path) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    reason = superseded_reason(expected.relative_dir)
    status = "superseded" if reason else "not_started"
    row.update(
        {
            "family": expected.family,
            "suite": Path(expected.relative_dir).parts[0],
            "experiment": expected.experiment,
            "stage": expected.stage,
            "registered": True,
            "status": status,
            "next_action": ACTIONS[status],
            "progress_pct": "0.00",
            "target_unit": expected.target_unit,
            "target_value": expected.target_value or "",
            "val_status": "missing",
            "test_status": "missing",
            "run_dir": str(root / expected.relative_dir),
            "relative_dir": expected.relative_dir,
            "issues": reason,
            "_mtime": 0.0,
        }
    )
    return row


STATUS_SCORE = {
    name: score
    for score, name in enumerate(
        (
            "not_started",
            "checkpoint_invalid",
            "training_failed",
            "training_state_unknown",
            "training_incomplete",
            "final_checkpoint_incomplete",
            "val_incomplete",
            "test_incomplete",
            "evaluation_stale",
            "finalization_incomplete",
            "superseded",
            "complete",
        )
    )
}


def choose(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = max(rows, key=lambda row: (STATUS_SCORE[row["status"]], row["_mtime"]))
    selected["duplicate_run_dirs"] = ";".join(
        row["run_dir"] for row in rows if row["run_dir"] != selected["run_dir"]
    )
    return selected


def metric(value: Any) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def print_report(rows: list[dict[str, Any]], output: Path) -> None:
    print("===== All experiment status counts =====")
    for status, count in sorted(Counter(row["status"] for row in rows).items()):
        print(f"{status:32} {count:4d}")
    print("\n===== All experiments =====")
    columns = (
        "suite", "experiment", "stage", "status", "progress_pct",
        "val_mIoU", "val_AP", "val_image_latency_s", "val_J&F",
        "val_video_latency_s", "test_mIoU", "test_AP",
        "test_image_latency_s", "test_J&F", "test_video_latency_s",
    )
    print("RESULT\t" + "\t".join(columns))
    for row in rows:
        values = [row[column] or "-" for column in columns[:5]]
        values.extend(metric(row[column]) for column in columns[5:])
        print("RESULT\t" + "\t".join(values))
    print(f"\nAll-experiment CSV: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", action="append", type=Path, required=True)
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()
    roots = []
    for root in args.runs_root:
        if root.is_dir() and root.resolve() not in roots:
            roots.append(root.resolve())
    if not roots:
        raise SystemExit("None of the supplied run roots exist")
    expected = {row.relative_dir: row for row in expected_experiments()}
    discovered = discover(roots)
    rows = []
    for relative in sorted(set(expected) | set(discovered)):
        candidates = [
            inspect_run(run_dir, relative, expected.get(relative))
            for run_dir in discovered.get(relative, [])
        ]
        rows.append(
            choose(candidates)
            if candidates
            else missing_row(expected[relative], roots[0])
        )
    rows.sort(key=lambda row: (row["suite"], row["experiment"], row["stage"]))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_csv.with_suffix(args.out_csv.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)
    temporary.replace(args.out_csv)
    print_report(rows, args.out_csv)
    print(f"Generated at: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
