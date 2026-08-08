#!/usr/bin/env python3
"""Audit an ETV run's EdgeTAM initialization and recorded loss outliers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml


TEMPORAL_PREFIXES = (
    "memory_attention.",
    "memory_encoder.",
    "spatial_perceiver.",
    "obj_ptr_proj.",
)
TEMPORAL_PARAMETERS = {
    "maskmem_tpos_enc",
    "no_mem_embed",
    "no_mem_pos_enc",
    "no_obj_ptr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--edgetam-checkpoint", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--expected-sha256", default="")
    parser.add_argument("--top-outliers", type=int, default=10)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "task_model_state", "model_state", "state_dict"):
        state = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(state, dict) and state:
            return {
                name.removeprefix("module."): value
                for name, value in state.items()
                if torch.is_tensor(value)
            }
    if isinstance(payload, dict) and payload and all(
        torch.is_tensor(value) for value in payload.values()
    ):
        return {
            name.removeprefix("module."): value
            for name, value in payload.items()
        }
    raise KeyError(f"No model tensor state found in {path}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_get(payload: dict[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [number for item in value for number in flatten_numbers(item)]
    return []


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize_outliers(path: Path, top_n: int) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path), "count": 0}
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mask_areas = [
        area
        for record in records
        for area in flatten_numbers(record.get("mask_areas", []))
        if area > 0
    ]

    def total_loss(record: dict[str, Any]) -> float:
        losses = record.get("losses", {})
        return float(losses.get("train/loss_total", 0.0))

    top = sorted(records, key=total_loss, reverse=True)[:top_n]
    return {
        "status": "pass",
        "path": str(path),
        "count": len(records),
        "positive_mask_area": {
            "count": len(mask_areas),
            "min": min(mask_areas) if mask_areas else None,
            "p10": percentile(mask_areas, 0.1),
            "median": percentile(mask_areas, 0.5),
            "p90": percentile(mask_areas, 0.9),
            "max": max(mask_areas) if mask_areas else None,
        },
        "top": [
            {
                "global_step": record.get("global_step"),
                "epoch": record.get("epoch"),
                "num_frames": record.get("num_frames"),
                "present_object_frames": record.get("present_object_frames"),
                "object_identifiers": record.get("object_identifiers", []),
                "losses": record.get("losses", {}),
                "positive_mask_area": [
                    area
                    for area in flatten_numbers(record.get("mask_areas", []))
                    if area > 0
                ],
            }
            for record in top
        ],
    }


def main() -> None:
    args = parse_args()
    if not args.run_dir.is_dir():
        raise FileNotFoundError(args.run_dir)
    if not args.edgetam_checkpoint.is_file():
        raise FileNotFoundError(args.edgetam_checkpoint)

    state = checkpoint_state(args.edgetam_checkpoint)
    temporal = {
        key: value
        for key, value in state.items()
        if key.startswith(TEMPORAL_PREFIXES) or key in TEMPORAL_PARAMETERS
    }
    checkpoint_hash = file_sha256(args.edgetam_checkpoint)
    initializer = read_json(args.run_dir / "initialization_summary.json")
    model_summary = read_json(args.run_dir / "training_model_summary.json")
    resolved_path = args.run_dir / "resolved_config.yaml"
    resolved = (
        yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if resolved_path.is_file()
        else {}
    )

    checks = {
        "initializer_passed": bool(
            initializer and initializer.get("status") == "pass"
        ),
        "official_temporal_requested": bool(
            initializer
            and initializer.get("memory_initializer") == "official_temporal"
        ),
        "official_temporal_provenance_matches_checkpoint": bool(
            initializer
            and initializer.get("tensor_provenance", {}).get(
                "official_edgetam"
            )
            == len(temporal)
        ),
        "trainable_tensor_count_matches_official_temporal": bool(
            model_summary
            and model_summary.get("trainable_tensors") == len(temporal)
        ),
        "trainable_parameter_count_matches_official_temporal": bool(
            model_summary
            and model_summary.get("trainable_parameters")
            == sum(value.numel() for value in temporal.values())
        ),
        "full_sav_train_cardinality": bool(
            model_summary
            and model_summary.get("train_dataset_samples") == 50337
        ),
        "batchnorm_frozen": nested_get(
            resolved, "trainer.model.freeze_batchnorm"
        )
        is True,
        "two_memory_attention_layers": nested_get(
            resolved, "trainer.model.memory_attention.num_layers"
        )
        == 2,
        "expected_checkpoint_hash": (
            not args.expected_sha256
            or checkpoint_hash == args.expected_sha256.lower()
        ),
    }
    summary = {
        "status": "pass" if all(checks.values()) else "audit_warning",
        "run_dir": str(args.run_dir),
        "edgetam_checkpoint": {
            "path": str(args.edgetam_checkpoint),
            "sha256": checkpoint_hash,
            "total_tensors": len(state),
            "official_temporal_tensors": len(temporal),
            "official_temporal_parameters": sum(
                value.numel() for value in temporal.values()
            ),
        },
        "initialization": initializer,
        "training_model": model_summary,
        "resolved_contract": {
            "freeze_batchnorm": nested_get(
                resolved, "trainer.model.freeze_batchnorm"
            ),
            "memory_attention_layers": nested_get(
                resolved, "trainer.model.memory_attention.num_layers"
            ),
            "gradient_clip_max_norm": nested_get(
                resolved, "trainer.optim.gradient_clip.max_norm"
            ),
            "memory_latents": nested_get(
                resolved, "trainer.model.spatial_perceiver.num_latents"
            ),
            "memory_latents_2d": nested_get(
                resolved, "trainer.model.spatial_perceiver.num_latents_2d"
            ),
            "loss_mask_weight": nested_get(
                resolved, "trainer.loss.all.task_loss.weight_dict.loss_mask"
            ),
            "loss_dice_weight": nested_get(
                resolved, "trainer.loss.all.task_loss.weight_dict.loss_dice"
            ),
            "lambda_mem": nested_get(
                resolved, "trainer.loss.all.lambda_mem"
            ),
            "lambda_mask_logits": nested_get(
                resolved, "trainer.loss.all.lambda_mask_logits"
            ),
        },
        "loss_outliers": summarize_outliers(
            args.run_dir / "loss_outliers.jsonl", args.top_outliers
        ),
        "checks": checks,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
