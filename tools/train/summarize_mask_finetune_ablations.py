#!/usr/bin/env python3
"""Record and atomically summarize SAM2 mask fine-tuning ablations."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
from pathlib import Path
from typing import Any


REGISTRY = {
    "D1_staged_image_align_1ep": (
        "official_behavior_curriculum",
        "Align the transplanted TinyViT image path to the official EdgeTAM teacher before temporal tuning.",
        "E1_a02_official_nonimage",
    ),
    "D2_staged_temporal_2ep": (
        "official_behavior_curriculum",
        "Tune only the official temporal stack with mask, memory, and object-pointer behavior targets.",
        "D1_staged_image_align_1ep",
    ),
    "D3_staged_t8_refine_1ep": (
        "official_behavior_horizon",
        "Refine the staged temporal stack on longer eligible T8 clips.",
        "D2_staged_temporal_2ep",
    ),
    "J1_joint_behavior_2ep": (
        "official_behavior_joint",
        "Jointly align TinyViT image and official temporal behavior from the strict transplant.",
        "E1_a02_official_nonimage",
    ),
    "J2_joint_temporal_refine_1ep": (
        "official_behavior_joint",
        "Protect the aligned image path while refining the temporal stack.",
        "J1_joint_behavior_2ep",
    ),
    "J3_joint_t8_refine_1ep": (
        "official_behavior_horizon",
        "Extend the joint curriculum to longer eligible T8 clips.",
        "J2_joint_temporal_refine_1ep",
    ),
    "S0_scratch_temporal_task_2ep": (
        "scratch_temporal_control",
        "Train a randomly initialized EdgeTAM Perceiver, memory, and object-pointer stack using SA-V task loss.",
        "E1_a02_official_nonimage",
    ),
    "S1_scratch_behavior_2ep": (
        "scratch_temporal_distillation",
        "Add official EdgeTAM temporal behavior targets after task-only scratch training.",
        "S0_scratch_temporal_task_2ep",
    ),
    "S2_scratch_t8_refine_1ep": (
        "scratch_temporal_horizon",
        "Refine the scratch-trained temporal stack on longer eligible T8 clips.",
        "S1_scratch_behavior_2ep",
    ),
    "C0_coherent_m0mem_align": (
        "temporal_initialization",
        "Align a coherent official EdgeTAM temporal stack to the functional M0 TinyViT memory output.",
        "M0_sam2_mem4",
    ),
    "C1_partial_m0mem_align": (
        "temporal_initialization_control",
        "Hold M0 memory-output distillation fixed while retaining the partial M2 initializer.",
        "C0_coherent_m0mem_align",
    ),
    "C2_coherent_m0mem_joint2ep": (
        "memory_curriculum",
        "Joint task and M0 memory distillation for two epochs from coherent initialization.",
        "C0_coherent_m0mem_align",
    ),
    "C3_coherent_m0mem_staged": (
        "memory_curriculum",
        "Add one task-plus-memory epoch after the pure C0 alignment epoch.",
        "C2_coherent_m0mem_joint2ep",
    ),
    "R0_edgetam_e2e_t4_task": (
        "edgetam_reproduction_scope",
        "Train the full TinyViT EdgeTAM student with official prompt simulation and task loss only.",
        "M2a_edgetam_hybrid2_official",
    ),
    "R1_edgetam_e2e_t4_imgkd": (
        "edgetam_reproduction_image_kd",
        "Add the official-weight image feature distillation term to full EdgeTAM training.",
        "R0_edgetam_e2e_t4_task",
    ),
    "R2_edgetam_e2e_t4_imgmemkd": (
        "edgetam_reproduction_memory_kd",
        "Add memory-output distillation at the official unit weight.",
        "R1_edgetam_e2e_t4_imgkd",
    ),
    "R3_edgetam_e2e_t8_imgmemkd": (
        "edgetam_reproduction_horizon",
        "Increase the full EdgeTAM distillation recipe from four to eight frames.",
        "R2_edgetam_e2e_t4_imgmemkd",
    ),
    "M0_sam2_mem4": (
        "memory_control",
        "Continue the current four-layer uncompressed SAM2 memory stack.",
        "",
    ),
    "M1_sam2_mem2": (
        "memory_depth",
        "Isolate the effect of reducing memory attention from four to two layers.",
        "M0_sam2_mem4",
    ),
    "M2a_edgetam_hybrid2_official": (
        "memory_compression",
        "Test EdgeTAM global-plus-2D Perceiver with the official memory pair.",
        "M1_sam2_mem2",
    ),
    "M2b_edgetam_hybrid2_current": (
        "memory_initialization",
        "Test the same Perceiver with current E2E memory-attention initialization.",
        "M2a_edgetam_hybrid2_official",
    ),
    "A00_e2e_t4_box1_control": (
        "control",
        "Full E2E T4, box plus one error-correction click.",
        "",
    ),
    "A01_e2e_t4_box0": (
        "correction",
        "An exact initial-box control tests whether error correction helps.",
        "A00_e2e_t4_box1_control",
    ),
    "A02_e2e_t4_official_prompt": (
        "prompt_mix",
        "The official mask/point/box and iterative-click mix improves robustness.",
        "A00_e2e_t4_box1_control",
    ),
    "A03_decmem_t4": (
        "trainable_scope",
        "Decoder plus temporal memory is sufficient without encoder updates.",
        "A00_e2e_t4_box1_control",
    ),
    "A04_memory_t4": (
        "trainable_scope",
        "Memory-only tuning isolates temporal adaptation.",
        "A00_e2e_t4_box1_control",
    ),
    "A05_e2e_t8": (
        "clip_length",
        "Longer T8 clips improve temporal consistency.",
        "A00_e2e_t4_box1_control",
    ),
    "A06_e2e_t8_s4_t16_hard": (
        "hard_refinement",
        "A frozen-encoder T16 hard-video refinement improves on the T8 model.",
        "A05_e2e_t8",
    ),
    "A07_e2e_t4_warmup5": (
        "optimization",
        "Five-percent LR warmup stabilizes full E2E tuning.",
        "A00_e2e_t4_box1_control",
    ),
    "A08_e2e_t4_gb8": (
        "batch_size",
        "Global batch eight improves gradients at fixed one-epoch data exposure.",
        "A00_e2e_t4_box1_control",
    ),
    "A09_e2e_t4_hard50x2": (
        "data_selection",
        "Twice-repeated bottom-50% base-error videos outperform uniform data.",
        "A00_e2e_t4_box1_control",
    ),
    "A10_e2e_t4_box0_imgkd": (
        "distillation",
        "Online SAM2.1-L image-feature KD improves exact-box E2E tuning.",
        "A01_e2e_t4_box0",
    ),
    "A11_e2e_t4_box0_imgmemkd": (
        "distillation",
        "Adding memory-feature KD improves temporal quality beyond image KD.",
        "A10_e2e_t4_box0_imgkd",
    ),
}

FIELDNAMES = [
    "variant",
    "suite",
    "axis",
    "hypothesis",
    "control_variant",
    "status",
    "git_commit",
    "seed",
    "manifest",
    "base_checkpoint",
    "base_stage_checkpoint",
    "video_ids_file",
    "train_samples",
    "epochs",
    "num_frames",
    "batch_per_gpu",
    "world_size",
    "global_batch",
    "trainable_mode",
    "total_parameters",
    "trainable_parameters",
    "encoder_lr",
    "encoder_lr_end",
    "head_lr",
    "head_lr_end",
    "warmup_fraction",
    "prob_point",
    "prob_box_given_point",
    "prob_gt_click",
    "correction_frames",
    "correction_points",
    "lambda_img",
    "lambda_mem",
    "lambda_task",
    "lambda_mask_logits",
    "lambda_obj_ptr",
    "teacher_checkpoint",
    "memory_topology",
    "memory_layers",
    "memory_initializer",
    "memory_layout",
    "num_global_latents",
    "num_2d_latents",
    "planned_updates_per_epoch",
    "optimizer_updates",
    "train_elapsed_seconds",
    "wandb_run_id",
    "wandb_url",
    "val_mIoU",
    "val_AP",
    "val_J&F",
    "val_J",
    "val_F",
    "val_image_seconds",
    "val_vos_seconds_per_video",
    "test_mIoU",
    "test_AP",
    "test_J&F",
    "test_J",
    "test_F",
    "test_image_seconds",
    "test_vos_seconds_per_video",
    "val_J&F_delta",
    "val_mIoU_delta",
    "val_AP_delta",
    "guardrail_pass",
    "selection_rank",
    "gate_J&F",
    "gate_pass",
    "gate_metrics_path",
    "stage_dir",
    "val_metrics_path",
    "test_metrics_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    record = subparsers.add_parser("record")
    record.add_argument("--variant-dir", required=True, type=Path)
    record.add_argument("--stage-dir", required=True, type=Path)
    record.add_argument("--central-csv", required=True, type=Path)
    scan = subparsers.add_parser("scan")
    scan.add_argument("--root", required=True, type=Path)
    scan.add_argument("--legacy-root", type=Path, action="append", default=[])
    scan.add_argument("--central-csv", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def count_train_samples(path: str, manifest: str) -> int | str:
    if path and Path(path).is_file():
        return sum(
            1
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not manifest or not Path(manifest).is_file():
        return ""
    import pandas as pd

    frame = pd.read_parquet(manifest, columns=["video_id", "split"])
    return int(frame.loc[frame["split"] == "train", "video_id"].nunique())


def metadata_from_env(variant_dir: Path, stage_dir: Path) -> dict[str, Any]:
    variant = variant_dir.name
    axis, hypothesis, control = REGISTRY.get(variant, ("", "", ""))
    video_ids_file = os.environ.get("TASK_VIDEO_IDS_FILE", "")
    manifest = os.environ.get("MANIFEST", os.environ.get("TASK_MANIFEST", ""))
    world_size = len(os.environ.get("GPUS", "0,1,2,3").split(","))
    batch = int(os.environ.get("TASK_TRAIN_BATCH_SIZE", "1"))
    return {
        "variant": variant,
        "suite": os.environ.get("TASK_EXPERIMENT_SUITE", "v2"),
        "axis": axis,
        "hypothesis": hypothesis,
        "control_variant": control,
        "git_commit": git_commit(),
        "seed": os.environ.get("TASK_SEED", "250107256"),
        "manifest": manifest,
        "base_checkpoint": os.environ.get("BASE_CHECKPOINT", ""),
        "base_stage_checkpoint": os.environ.get("BASE_STAGE_CHECKPOINT", ""),
        "video_ids_file": video_ids_file,
        "train_samples": count_train_samples(video_ids_file, manifest),
        "epochs": os.environ.get("TASK_EPOCHS", "1"),
        "num_frames": os.environ.get("TASK_NUM_FRAMES", ""),
        "batch_per_gpu": batch,
        "world_size": world_size,
        "global_batch": batch * world_size,
        "trainable_mode": os.environ.get("TASK_TRAINABLE_MODE", ""),
        "encoder_lr": os.environ.get("TASK_ENCODER_LR", ""),
        "encoder_lr_end": os.environ.get("TASK_ENCODER_LR_END", ""),
        "head_lr": os.environ.get("TASK_HEAD_LR", ""),
        "head_lr_end": os.environ.get("TASK_HEAD_LR_END", ""),
        "warmup_fraction": os.environ.get("TASK_LR_WARMUP_FRACTION", "0"),
        "prob_point": os.environ.get("TASK_PROB_USE_POINT", "1"),
        "prob_box_given_point": os.environ.get("TASK_PROB_USE_BOX", "1"),
        "prob_gt_click": os.environ.get("TASK_PROB_SAMPLE_GT", "0"),
        "correction_frames": os.environ.get("TASK_NUM_FRAMES_TO_CORRECT", "1"),
        "correction_points": os.environ.get("TASK_NUM_CORRECTION_POINTS", "1"),
        "lambda_img": os.environ.get("TASK_LAMBDA_IMG", "0"),
        "lambda_mem": os.environ.get("TASK_LAMBDA_MEM", "0"),
        "lambda_task": os.environ.get("TASK_LAMBDA_TASK", "1"),
        "lambda_mask_logits": os.environ.get(
            "TASK_LAMBDA_MASK_LOGITS", "0"
        ),
        "lambda_obj_ptr": os.environ.get("TASK_LAMBDA_OBJ_PTR", "0"),
        "teacher_checkpoint": os.environ.get("TASK_TEACHER_CHECKPOINT", ""),
        "memory_topology": os.environ.get("TASK_MEMORY_TOPOLOGY", ""),
        "memory_layers": os.environ.get("TASK_MEMORY_LAYERS", ""),
        "memory_initializer": os.environ.get("TASK_MEMORY_INITIALIZER", ""),
        "memory_layout": os.environ.get("TASK_MEMORY_LAYOUT", "legacy"),
        "num_global_latents": os.environ.get("TASK_NUM_GLOBAL_LATENTS", ""),
        "num_2d_latents": os.environ.get("TASK_NUM_2D_LATENTS", ""),
        "stage_dir": str(stage_dir),
    }


def metric_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["mode"]: row for row in csv.DictReader(handle)}


def add_split_metrics(row: dict[str, Any], split: str, path: Path) -> None:
    metrics = metric_rows(path)
    image = metrics.get("image", {})
    video = metrics.get("video_tracking", {})
    row[f"{split}_mIoU"] = image.get("mIoU", "")
    row[f"{split}_AP"] = image.get("AP", "")
    row[f"{split}_J&F"] = video.get("J&F", "")
    row[f"{split}_J"] = video.get("J", "")
    row[f"{split}_F"] = video.get("F", "")
    row[f"{split}_image_seconds"] = image.get(
        "mean_total_object_seconds", ""
    )
    row[f"{split}_vos_seconds_per_video"] = video.get("sec_per_video", "")
    row[f"{split}_metrics_path"] = str(path)


def checkpoint_updates(stage_dir: Path) -> str:
    path = stage_dir / "checkpoints/checkpoint.pt"
    if not path.is_file():
        return ""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return str(payload.get("steps", {}).get("train", ""))


def build_row(metadata: dict[str, Any]) -> dict[str, Any]:
    stage_dir = Path(metadata["stage_dir"])
    row = {key: "" for key in FIELDNAMES}
    row.update(metadata)
    model = read_json(stage_dir / "training_model_summary.json")
    status = read_json(stage_dir / "training_status.json")
    wandb = read_json(stage_dir / "wandb/wandb_run.json")
    row["status"] = status.get("status", "pending")
    row["total_parameters"] = model.get("total_parameters", "")
    row["trainable_parameters"] = model.get("trainable_parameters", "")
    row["train_samples"] = model.get(
        "train_dataset_samples", row.get("train_samples", "")
    )
    row["planned_updates_per_epoch"] = model.get(
        "optimizer_updates_per_epoch", ""
    )
    row["optimizer_updates"] = checkpoint_updates(stage_dir)
    row["train_elapsed_seconds"] = status.get("elapsed_seconds", "")
    row["wandb_run_id"] = wandb.get("run_id", "")
    row["wandb_url"] = wandb.get("url", "")
    gate = read_json(stage_dir / "gate_status.json")
    gate_metrics = gate.get("metrics", {})
    row["gate_J&F"] = gate_metrics.get("J&F", "")
    row["gate_pass"] = (
        int(gate["status"] == "pass") if gate.get("status") else ""
    )
    row["gate_metrics_path"] = gate.get("metrics_path", "")
    add_split_metrics(
        row, "val", stage_dir / "sav_val_box_benchmark/metrics.csv"
    )
    add_split_metrics(
        row, "test", stage_dir / "sav_test_box_benchmark/metrics.csv"
    )
    if row["status"] == "complete" and (
        not row["val_J&F"] or not row["test_J&F"]
    ):
        row["status"] = "evaluation_incomplete"
    return row


def as_float(value: Any) -> float | None:
    try:
        return float(value) if value != "" else None
    except (TypeError, ValueError):
        return None


def add_comparisons(rows: list[dict[str, Any]]) -> None:
    by_variant = {row["variant"]: row for row in rows}
    for row in rows:
        control = by_variant.get(row.get("control_variant", ""))
        if control is None:
            continue
        for metric in ("val_J&F", "val_mIoU", "val_AP"):
            value = as_float(row.get(metric))
            base = as_float(control.get(metric))
            row[f"{metric}_delta"] = "" if value is None or base is None else value - base
        miou_delta = as_float(row.get("val_mIoU_delta"))
        ap_delta = as_float(row.get("val_AP_delta"))
        if miou_delta is not None and ap_delta is not None:
            row["guardrail_pass"] = int(miou_delta >= -0.005 and ap_delta >= -0.005)
    ranked = [
        row
        for row in rows
        if row.get("suite") == "v2" and as_float(row.get("val_J&F")) is not None
    ]
    ranked.sort(key=lambda row: as_float(row["val_J&F"]), reverse=True)
    for rank, row in enumerate(ranked, 1):
        row["selection_rank"] = rank


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def update_central(path: Path, new_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = []
        if path.is_file():
            with path.open(encoding="utf-8", newline="") as handle:
                existing = list(csv.DictReader(handle))
        replace = {(row["suite"], row["variant"]) for row in new_rows}
        rows = [
            row
            for row in existing
            if (row.get("suite", ""), row.get("variant", "")) not in replace
        ]
        rows.extend(new_rows)
        rows.sort(key=lambda row: (row.get("suite", ""), row.get("variant", "")))
        add_comparisons(rows)
        write_csv(path, rows)
        fcntl.flock(lock, fcntl.LOCK_UN)


def record(variant_dir: Path, stage_dir: Path, central_csv: Path) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    metadata = metadata_from_env(variant_dir, stage_dir)
    metadata_path = variant_dir / "experiment.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    row = build_row(metadata)
    write_csv(variant_dir / "summary.csv", [row])
    update_central(central_csv, [row])
    print(json.dumps(row, indent=2))


def legacy_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    if not root.is_dir():
        return rows
    for val_path in sorted(root.glob("*/*/sav_val_box_benchmark/metrics.csv")):
        stage_dir = val_path.parents[1]
        variant = stage_dir.parents[0].name
        metadata = {
            "variant": variant,
            "suite": "v1",
            "axis": "legacy_lr_scope_prompt",
            "hypothesis": "Existing six-run mask fine-tuning ablation.",
            "control_variant": "",
            "stage_dir": str(stage_dir),
        }
        rows.append(build_row(metadata))
    return rows


def scan(root: Path, legacy_roots: list[Path], central_csv: Path) -> None:
    rows = []
    for metadata_path in sorted(root.glob("*/experiment.json")):
        rows.append(build_row(read_json(metadata_path)))
    for legacy_root in legacy_roots:
        rows.extend(legacy_rows(legacy_root))
    rows = list(
        {
            (row.get("suite", ""), row.get("variant", "")): row
            for row in rows
        }.values()
    )
    if not rows:
        print(
            "No experiment metadata rows found; preserving "
            f"{central_csv}"
        )
        return
    update_central(central_csv, rows)
    print(f"Wrote {len(rows)} rows to {central_csv}")


def main() -> None:
    args = parse_args()
    if args.action == "record":
        record(args.variant_dir, args.stage_dir, args.central_csv)
    else:
        scan(args.root, args.legacy_root, args.central_csv)


if __name__ == "__main__":
    main()
