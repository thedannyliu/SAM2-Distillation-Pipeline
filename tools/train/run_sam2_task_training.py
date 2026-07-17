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
    run_id = os.environ.get("WANDB_RUN_ID", "").strip() or None
    if run_id is None:
        os.environ.pop("WANDB_RUN_ID", None)
    if run_file.is_file():
        saved_run_id = json.loads(run_file.read_text())["run_id"]
        if run_id is not None and run_id != saved_run_id:
            raise RuntimeError(
                f"W&B run ID mismatch: environment={run_id}, saved={saved_run_id}"
            )
        run_id = saved_run_id
    checkpoint_path = Path(os.environ["TASK_RUN_DIR"]) / "checkpoints/checkpoint.pt"
    if run_id is None and checkpoint_path.is_file():
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        run_id = checkpoint.get("wandb_run_id")
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        id=run_id,
        resume="must" if run_id else None,
        dir=str(args.wandb_dir),
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
        json.dumps(
            {
                "run_id": run.id,
                "url": run.url,
                "entity": run.entity,
                "project": args.wandb_project,
                "name": args.wandb_name,
            }
        )
        + "\n"
    )
    print(f"W&B run: {run.url} (id={run.id})", flush=True)
    return run


def _scalar(value):
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _wandb_loss_name(name: str) -> str:
    aliases = {
        "Losses/train_all_loss": "train/loss_total",
        "Losses/train_all_core_loss": "train/loss_core",
        "Losses/train_all_loss_mask": "train/loss_mask",
        "Losses/train_all_loss_dice": "train/loss_dice",
        "Losses/train_all_loss_iou": "train/loss_iou",
        "Losses/train_all_loss_class": "train/loss_class",
    }
    return aliases.get(name, f"train/{name.replace('/', '_')}")


def patch_sam2_training_runtime(wandb_run=None) -> None:
    """Use compact console output and direct W&B metric logging."""
    import training.optimizer as optimizer_module
    import training.trainer as trainer_module
    import training.dataset.vos_dataset as vos_dataset_module

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

    def compact_model_initializer(self):
        initializer = trainer_module.instantiate(
            self.checkpoint_conf.model_weight_initializer
        )
        if initializer is not None:
            logging.info("Loading task model checkpoint initializer")
            self.model = initializer(model=self.model)

    original_run_step = trainer_module.Trainer._run_step
    original_save_checkpoint = trainer_module.Trainer._save_checkpoint
    loss_ema: dict[str, float] = {}
    ema_beta = float(os.environ.get("WANDB_LOSS_EMA_BETA", "0.98"))

    def run_step_with_wandb(
        self,
        batch,
        phase,
        loss_mts,
        extra_loss_mts,
        raise_on_error=True,
    ):
        result = original_run_step(
            self,
            batch,
            phase,
            loss_mts,
            extra_loss_mts,
            raise_on_error=raise_on_error,
        )
        completed_step = int(self.steps[phase])
        log_frequency = int(self.logging_conf.log_scalar_frequency)
        should_log = (completed_step - 1) % log_frequency == 0
        if wandb_run is not None and self.distributed_rank == 0:
            current_losses = {
                _wandb_loss_name(name): _scalar(meter.val)
                for name, meter in loss_mts.items()
            }
            current_losses.update(
                {
                    _wandb_loss_name(name): _scalar(meter.val)
                    for name, meter in extra_loss_mts.items()
                }
            )
            for name, value in current_losses.items():
                previous = loss_ema.get(name, value)
                loss_ema[name] = ema_beta * previous + (1.0 - ema_beta) * value
            if not should_log:
                return result
            metrics = dict(current_losses)
            metrics.update(
                {f"{name}_ema": value for name, value in loss_ema.items()}
            )
            metrics.update(
                {
                    "train/epoch": float(self.epoch),
                    "train/global_step": float(completed_step),
                }
            )
            for index, group in enumerate(self.optim.optimizer.param_groups):
                metrics[f"train/lr_group_{index}"] = float(group["lr"])
            wandb_run.log(metrics, step=completed_step)
        return result

    def save_checkpoint_with_wandb(self, checkpoint, checkpoint_path):
        if wandb_run is not None:
            checkpoint["wandb_run_id"] = wandb_run.id
        return original_save_checkpoint(self, checkpoint, checkpoint_path)

    trainer_module.print_model_summary = compact_model_summary
    trainer_module.log_env_variables = lambda: None
    trainer_module.Trainer._call_model_initializer = compact_model_initializer
    trainer_module.Trainer._run_step = run_step_with_wandb
    trainer_module.Trainer._save_checkpoint = save_checkpoint_with_wandb
    optimizer_module.unix_param_pattern_to_parameter_names = quiet_param_pattern_match
    vos_dataset_module.print = lambda *args, **kwargs: None


def main() -> None:
    args = parse_args()
    sam2_root = Path(os.environ["SAM2_TRAINING_ROOT"])
    sys.path.insert(0, str(sam2_root))
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from training.utils.train_utils import register_omegaconf_resolvers

    register_omegaconf_resolvers()
    config = OmegaConf.load(args.config)
    run = init_wandb(args)
    patch_sam2_training_runtime(run)
    succeeded = False
    try:
        trainer = instantiate(config.trainer, _recursive_=False)
        trainer.run()
        succeeded = True
    finally:
        if run is not None:
            run.summary["system/training_complete"] = int(succeeded)
            run.finish(exit_code=0 if succeeded else 1)


if __name__ == "__main__":
    main()
