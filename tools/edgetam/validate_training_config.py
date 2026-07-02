#!/usr/bin/env python
"""Validate EdgeTAM training config targets without launching training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sam2-training-root", type=Path, required=True)
    parser.add_argument("--edgetam-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--instantiate-loss", action="store_true")
    parser.add_argument("--instantiate-model", action="store_true")
    return parser.parse_args()


def add_import_roots(edgetam_root: Path, sam2_training_root: Path) -> None:
    for root in (sam2_training_root, edgetam_root):
        if not root.exists():
            raise FileNotFoundError(root)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(sam2_training_root))
    sys.path.insert(0, str(edgetam_root))


def main() -> None:
    args = parse_args()
    add_import_roots(args.edgetam_root, args.sam2_training_root)

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("times", lambda a, b: a * b, replace=True)
    OmegaConf.register_new_resolver("divide", lambda a, b: a / b, replace=True)

    cfg = OmegaConf.load(args.config)
    model_target = cfg.trainer.model._target_
    loss_target = cfg.trainer.loss.all._target_
    task_loss_target = cfg.trainer.loss.all.task_loss._target_

    summary = {
        "config": str(args.config),
        "model_target": model_target,
        "loss_target": loss_target,
        "task_loss_target": task_loss_target,
        "instantiate_loss": bool(args.instantiate_loss),
        "instantiate_model": bool(args.instantiate_model),
        "result": "pass",
    }

    if args.instantiate_loss:
        loss = instantiate(cfg.trainer.loss.all, _convert_="all")
        summary["loss_class"] = type(loss).__name__
        summary["task_loss_class"] = type(loss.task_loss).__name__

    if args.instantiate_model:
        model = instantiate(cfg.trainer.model, _convert_="all")
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total = sum(param.numel() for param in model.parameters())
        summary["model_class"] = type(model).__name__
        summary["model_parameters"] = int(total)
        summary["model_trainable_parameters"] = int(trainable)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
