#!/usr/bin/env python3
"""Create a strict EdgeTAM non-image plus TinyViT image checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sam2_distill.models.task_finetune import _checkpoint_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tinyvit-task-checkpoint", required=True, type=Path)
    parser.add_argument("--official-edgetam-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--name", default="E1_a02_official_nonimage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tinyvit_state = _checkpoint_model_state(args.tinyvit_task_checkpoint)
    official_state = _checkpoint_model_state(args.official_edgetam_checkpoint)

    image_state = {
        key: value
        for key, value in tinyvit_state.items()
        if key.startswith("image_encoder.")
    }
    non_image_state = {
        key: value
        for key, value in official_state.items()
        if not key.startswith("image_encoder.")
    }
    if not image_state:
        raise KeyError("TinyViT task checkpoint has no image_encoder tensors")
    if not non_image_state:
        raise KeyError("Official EdgeTAM checkpoint has no non-image tensors")
    overlap = set(image_state).intersection(non_image_state)
    if overlap:
        raise RuntimeError(f"Unexpected transplant key overlap: {sorted(overlap)[:5]}")

    merged = {**image_state, **non_image_state}
    payload = {
        "model": merged,
        "task_model_state": merged,
        "epoch": 0,
        "steps": {"train": 0},
        "transplant": {
            "name": args.name,
            "tinyvit_task_checkpoint": str(args.tinyvit_task_checkpoint),
            "official_edgetam_checkpoint": str(
                args.official_edgetam_checkpoint
            ),
            "image_tensors": len(image_state),
            "official_non_image_tensors": len(non_image_state),
            "total_tensors": len(merged),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(payload["transplant"], indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["transplant"], indent=2))


if __name__ == "__main__":
    main()
