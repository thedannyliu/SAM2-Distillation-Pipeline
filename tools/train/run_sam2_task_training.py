#!/usr/bin/env python3
"""Run one SAM2 task-finetuning stage from a repo-owned OmegaConf file."""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--wandb-dir", required=True, type=Path)
    return parser.parse_args()


def init_wandb(args: argparse.Namespace):
    if int(os.environ.get("RANK", "0")) != 0:
        return None
    if os.environ.get("WANDB_MODE", "online") == "disabled":
        return None
    import wandb

    args.wandb_dir.mkdir(parents=True, exist_ok=True)
    run_file = args.wandb_dir / "wandb_run.json"
    run_id = None
    if run_file.is_file():
        run_id = json.loads(run_file.read_text())["run_id"]
    tensorboard_dir = Path(os.environ["TASK_RUN_DIR"]) / "tensorboard"
    wandb.tensorboard.patch(root_logdir=str(tensorboard_dir))
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        id=run_id,
        resume="must" if run_id else None,
        dir=str(args.wandb_dir),
        sync_tensorboard=True,
        config={
            "task_stage": os.environ.get("TASK_STAGE_NAME"),
            "trainable_mode": os.environ.get("TASK_TRAINABLE_MODE"),
            "epochs": int(os.environ.get("TASK_EPOCHS", "0")),
            "frames": int(os.environ.get("TASK_NUM_FRAMES", "0")),
            "encoder_lr": float(os.environ.get("TASK_ENCODER_LR", "0")),
            "head_lr": float(os.environ.get("TASK_HEAD_LR", "0")),
        },
    )
    run_file.write_text(
        json.dumps({"run_id": run.id, "url": run.url, "project": args.wandb_project})
        + "\n"
    )
    return run


def patch_sam2_training_console_output() -> None:
    """Keep useful trainer summaries without dumping the full model/param sets."""
    import training.optimizer as optimizer_module
    import training.trainer as trainer_module

    def compact_model_summary(model, log_dir=""):
        del log_dir
        if int(os.environ.get("RANK", "0")) != 0:
            return
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        logging.info(
            "Model summary: %s, total %.1f M, trainable %.1f M, frozen %.1f M",
            type(model).__name__,
            total / 1e6,
            trainable / 1e6,
            (total - trainable) / 1e6,
        )

    def quiet_param_pattern_match(filter_param_names, parameter_names):
        if filter_param_names is None:
            return set()
        matches = []
        for pattern in filter_param_names:
            matched = set(fnmatch.filter(parameter_names, pattern))
            assert matched, f"No parameter names match pattern {pattern!r}"
            matches.append(matched)
        return set().union(*matches)

    trainer_module.print_model_summary = compact_model_summary
    optimizer_module.unix_param_pattern_to_parameter_names = quiet_param_pattern_match


def main() -> None:
    args = parse_args()
    sam2_root = Path(os.environ["SAM2_TRAINING_ROOT"])
    sys.path.insert(0, str(sam2_root))
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from training.utils.train_utils import register_omegaconf_resolvers

    patch_sam2_training_console_output()
    register_omegaconf_resolvers()
    config = OmegaConf.load(args.config)
    run = init_wandb(args)
    try:
        trainer = instantiate(config.trainer, _recursive_=False)
        trainer.run()
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
