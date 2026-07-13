#!/usr/bin/env python3
"""Measure and optionally export SAM2.1 + Stage 1 TinyViT inference weights."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam2_distill.models.stage1_checkpoint import (
    extract_state_dict,
    infer_adapter_mode,
    infer_tinyvit_model_name,
)


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam2-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--student",
        action="append",
        required=True,
        metavar="LABEL=CHECKPOINT",
        help="Stage 1 checkpoint; repeat for each TinyViT variant",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--export-dtype",
        action="append",
        choices=sorted(DTYPES),
        help="Write pure inference bundles in this dtype; may be repeated",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint is not a dictionary: {path}")
    return value


def parse_student(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--student must be LABEL=CHECKPOINT, got: {value}")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise ValueError(f"--student must be LABEL=CHECKPOINT, got: {value}")
    return label, Path(raw_path)


def tensor_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state.items() if torch.is_tensor(value)}


def tensor_stats(state: dict[str, torch.Tensor]) -> tuple[int, int]:
    elements = sum(value.numel() for value in state.values())
    size = sum(value.numel() * value.element_size() for value in state.values())
    return elements, size


def mib(size: int | float) -> float:
    return float(size) / 1024**2


def cast_state(
    state: dict[str, torch.Tensor], dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach()
        .cpu()
        .to(dtype=dtype if value.is_floating_point() else value.dtype)
        .contiguous()
        for key, value in state.items()
    }


def expected_variant(label: str) -> str | None:
    normalized = label.lower().replace("_", "")
    for size in ("21m", "11m", "5m"):
        if f"tv{size}" in normalized or f"tinyvit{size}" in normalized:
            return size
    return None


def inferred_variant(model_name: str) -> str:
    for size in ("21m", "11m", "5m"):
        if f"_{size}_" in model_name:
            return size
    return "unknown"


def export_bundle(
    path: Path,
    label: str,
    model_name: str,
    adapter_mode: str,
    sam2_checkpoint: Path,
    non_image: dict[str, torch.Tensor],
    student: dict[str, torch.Tensor],
    dtype_name: str,
) -> int:
    dtype = DTYPES[dtype_name]
    payload = {
        "format": "sam2_stage1_hybrid_inference_v1",
        "metadata": {
            "label": label,
            "tinyvit_model_name": model_name,
            "adapter_mode": adapter_mode,
            "sam2_source_checkpoint": str(sam2_checkpoint),
            "floating_dtype": dtype_name,
        },
        "sam2_non_image_state": cast_state(non_image, dtype),
        "student_image_encoder_state": cast_state(student, dtype),
    }
    torch.save(payload, path)
    return path.stat().st_size


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sam2_checkpoint = load_checkpoint(args.sam2_checkpoint)
    sam2_state = tensor_state(extract_state_dict(sam2_checkpoint))
    image_prefix = "image_encoder."
    if not any(key.startswith(image_prefix) for key in sam2_state):
        raise SystemExit("SAM2 checkpoint has no image_encoder.* tensors")
    non_image_state = {
        key: value for key, value in sam2_state.items() if not key.startswith(image_prefix)
    }
    non_image_elements, non_image_source_bytes = tensor_stats(non_image_state)

    rows = []
    for student_arg in args.student:
        label, checkpoint_path = parse_student(student_arg)
        checkpoint = load_checkpoint(checkpoint_path)
        student_state = tensor_state(extract_state_dict(checkpoint))
        model_name = infer_tinyvit_model_name(student_state, "unknown")
        adapter_mode = infer_adapter_mode(checkpoint, student_state)
        detected = inferred_variant(model_name)
        expected = expected_variant(label)
        architecture_match = expected is None or expected == detected
        student_elements, student_source_bytes = tensor_stats(student_state)
        hybrid_elements = non_image_elements + student_elements
        row: dict[str, Any] = {
            "label": label,
            "checkpoint": str(checkpoint_path),
            "checkpoint_file_MiB": mib(checkpoint_path.stat().st_size),
            "inferred_model": model_name,
            "adapter_mode": adapter_mode,
            "architecture_match": architecture_match,
            "student_state_elements": student_elements,
            "student_source_tensor_MiB": mib(student_source_bytes),
            "sam2_non_image_elements": non_image_elements,
            "sam2_non_image_source_tensor_MiB": mib(non_image_source_bytes),
            "sam2_non_image_fp32_MiB": mib(non_image_elements * 4),
            "sam2_non_image_fp16_MiB": mib(non_image_elements * 2),
            "hybrid_state_elements": hybrid_elements,
            "hybrid_fp32_tensor_MiB": mib(hybrid_elements * 4),
            "hybrid_fp16_tensor_MiB": mib(hybrid_elements * 2),
        }
        for dtype_name in args.export_dtype or []:
            export_path = args.out_dir / f"{label}.sam2_hybrid.{dtype_name}.pt"
            size = export_bundle(
                export_path,
                label,
                model_name,
                adapter_mode,
                args.sam2_checkpoint,
                non_image_state,
                student_state,
                dtype_name,
            )
            row[f"export_{dtype_name}_file_MiB"] = mib(size)
            row[f"export_{dtype_name}_path"] = str(export_path)
        rows.append(row)

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    csv_path = args.out_dir / "sam2_hybrid_sizes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "sam2_checkpoint": str(args.sam2_checkpoint),
        "sam2_checkpoint_file_MiB": mib(args.sam2_checkpoint.stat().st_size),
        "sam2_non_image_elements": non_image_elements,
        "sam2_non_image_source_tensor_MiB": mib(non_image_source_bytes),
        "rows": rows,
    }
    json_path = args.out_dir / "sam2_hybrid_sizes.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    mismatches = [row["label"] for row in rows if not row["architecture_match"]]
    if mismatches:
        raise SystemExit(f"TinyViT label/architecture mismatch: {', '.join(mismatches)}")


if __name__ == "__main__":
    main()
