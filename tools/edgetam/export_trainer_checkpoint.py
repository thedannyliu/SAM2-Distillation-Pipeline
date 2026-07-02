#!/usr/bin/env python3
"""Export a SAM2 Trainer checkpoint to a model-only EdgeTAM checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model-config", type=Path, help="Optional model-only YAML for strict load validation.")
    parser.add_argument("--edgetam-root", type=Path)
    parser.add_argument("--sam2-training-root", type=Path)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def add_import_roots(edgetam_root: Path | None, sam2_training_root: Path | None) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    for root in (sam2_training_root, edgetam_root):
        if root is None:
            continue
        if not root.exists():
            raise FileNotFoundError(root)
        sys.path.insert(0, str(root))


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def extract_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model", "model_state", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return strip_module_prefix(value)
    raise KeyError("checkpoint must contain one of: model, model_state, state_dict")


def validate_model_config(model_config: Path, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(model_config)
    model = instantiate(cfg.model, _recursive_=True)
    incompatible = model.load_state_dict(state_dict, strict=True)
    return {
        "model_config": str(model_config),
        "model_class": type(model).__name__,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def main() -> None:
    args = parse_args()
    add_import_roots(args.edgetam_root, args.sam2_training_root)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"expected dict checkpoint, got {type(checkpoint).__name__}")
    state_dict = extract_model_state(checkpoint)

    validation = None
    if args.model_config is not None:
        validation = validate_model_config(args.model_config, state_dict)

    metadata = {
        "source_checkpoint": str(args.checkpoint),
        "format": "sam2_model_checkpoint_v1",
        "num_tensors": len(state_dict),
        "epoch": checkpoint.get("epoch"),
        "steps": checkpoint.get("steps"),
        "validation": validation,
    }
    payload = {
        "model": state_dict,
        "metadata": metadata,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)

    summary = {
        "result": "pass",
        "checkpoint": str(args.checkpoint),
        "out": str(args.out),
        "num_tensors": len(state_dict),
        "epoch": checkpoint.get("epoch"),
        "steps": checkpoint.get("steps"),
        "validation": validation,
    }
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
