#!/usr/bin/env python3
"""Run one SAM2 task-finetuning stage from a repo-owned OmegaConf file."""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import sys
import time
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
            "freeze_batchnorm": os.environ.get(
                "TASK_FREEZE_BATCHNORM", "true"
            ).lower()
            == "true",
            "num_correction_points": int(
                os.environ.get("TASK_NUM_CORRECTION_POINTS", "1")
            ),
            "train_batch_size_per_gpu": int(
                os.environ.get("TASK_TRAIN_BATCH_SIZE", "1")
            ),
            "video_ids_file": os.environ.get("TASK_VIDEO_IDS_FILE", ""),
            "lr_warmup_fraction": float(
                os.environ.get("TASK_LR_WARMUP_FRACTION", "0")
            ),
            "lr_warmup_start_factor": float(
                os.environ.get("TASK_LR_WARMUP_START_FACTOR", "0.1")
            ),
            "lambda_img": float(os.environ.get("TASK_LAMBDA_IMG", "0")),
            "lambda_mem": float(os.environ.get("TASK_LAMBDA_MEM", "0")),
            "prompt_pt_probability": float(
                os.environ.get("TASK_PROB_USE_POINT", "1")
            ),
            "prompt_box_given_point_probability": float(
                os.environ.get("TASK_PROB_USE_BOX", "1")
            ),
            "num_frames_to_correct": int(
                os.environ.get("TASK_NUM_FRAMES_TO_CORRECT", "1")
            ),
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
        "Losses/train_all_loss_img_distill": "train/loss_img_distill",
        "Losses/train_all_loss_mem_distill": "train/loss_mem_distill",
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

    warmup_fraction = float(os.environ.get("TASK_LR_WARMUP_FRACTION", "0"))
    warmup_start_factor = float(
        os.environ.get("TASK_LR_WARMUP_START_FACTOR", "0.1")
    )
    if warmup_fraction > 0:
        if not 0 < warmup_fraction <= 1:
            raise ValueError("TASK_LR_WARMUP_FRACTION must be in (0, 1]")
        if not 0 < warmup_start_factor <= 1:
            raise ValueError("TASK_LR_WARMUP_START_FACTOR must be in (0, 1]")
        original_step_schedulers = optimizer_module.Optimizer.step_schedulers

        def step_schedulers_with_warmup(self, where, step):
            result = original_step_schedulers(self, where, step)
            if where < warmup_fraction:
                progress = max(float(where), 0.0) / warmup_fraction
                scale = warmup_start_factor + (1.0 - warmup_start_factor) * progress
                for group in self.optimizer.param_groups:
                    group["lr"] *= scale
            return result

        optimizer_module.Optimizer.step_schedulers = step_schedulers_with_warmup


def apply_mask_ablation_overrides(config) -> None:
    """Apply opt-in v2 knobs without changing legacy task-training configs."""
    if os.environ.get("TASK_MASK_ABLATION_V2", "0") != "1":
        return

    from omegaconf import OmegaConf

    model = config.trainer.model
    data = config.trainer.data.train
    dataset = data.datasets[0].video_dataset
    sampler = data.datasets[0].sampler

    config.trainer.seed_value = int(os.environ.get("TASK_SEED", "250107256"))
    batch_size = int(os.environ.get("TASK_TRAIN_BATCH_SIZE", "1"))
    data.batch_sizes[0] = batch_size
    video_ids_file = os.environ.get("TASK_VIDEO_IDS_FILE", "").strip()
    if video_ids_file:
        dataset.video_ids_file = video_ids_file
    sampler.max_num_objects = int(os.environ.get("TASK_MAX_NUM_OBJECTS", "2"))

    prompt_values = {
        "prob_to_use_pt_input_for_train": float(
            os.environ.get("TASK_PROB_USE_POINT", "1")
        ),
        "prob_to_use_box_input_for_train": float(
            os.environ.get("TASK_PROB_USE_BOX", "1")
        ),
        "prob_to_sample_from_gt_for_train": float(
            os.environ.get("TASK_PROB_SAMPLE_GT", "0")
        ),
        "num_frames_to_correct_for_train": int(
            os.environ.get("TASK_NUM_FRAMES_TO_CORRECT", "1")
        ),
        "rand_frames_to_correct_for_train": os.environ.get(
            "TASK_RANDOM_CORRECTION_FRAMES", "false"
        ).lower()
        == "true",
        "num_init_cond_frames_for_train": int(
            os.environ.get("TASK_NUM_INIT_COND_FRAMES", "1")
        ),
        "rand_init_cond_frames_for_train": os.environ.get(
            "TASK_RANDOM_INIT_COND_FRAMES", "false"
        ).lower()
        == "true",
        "num_correction_pt_per_frame": int(
            os.environ.get("TASK_NUM_CORRECTION_POINTS", "1")
        ),
    }
    for key, value in prompt_values.items():
        model[key] = value

    lambda_img = float(os.environ.get("TASK_LAMBDA_IMG", "0"))
    lambda_mem = float(os.environ.get("TASK_LAMBDA_MEM", "0"))
    if lambda_img or lambda_mem:
        teacher_config = os.environ.get("TASK_TEACHER_MODEL_CONFIG", "").strip()
        teacher_checkpoint = os.environ.get("TASK_TEACHER_CHECKPOINT", "").strip()
        if not teacher_config or not teacher_checkpoint:
            raise ValueError(
                "KD requires TASK_TEACHER_MODEL_CONFIG and TASK_TEACHER_CHECKPOINT"
            )
        model._target_ = (
            "sam2_distill.edgetam.train_model.EdgeTAMTrainWithTeacher"
        )
        model.teacher_model_config = teacher_config
        model.teacher_checkpoint = teacher_checkpoint
        task_loss = config.trainer.loss.all
        config.trainer.loss.all = OmegaConf.create(
            {
                "_target_": (
                    "sam2_distill.edgetam.distillation_losses."
                    "EdgeTAMMultiStepDistillationLoss"
                ),
                "task_loss": task_loss,
                "lambda_img": lambda_img,
                "lambda_mem": lambda_mem,
            }
        )


def main() -> None:
    args = parse_args()
    sam2_root = Path(os.environ["SAM2_TRAINING_ROOT"])
    sys.path.insert(0, str(sam2_root))
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from training.utils.train_utils import register_omegaconf_resolvers

    register_omegaconf_resolvers()
    config = OmegaConf.load(args.config)
    apply_mask_ablation_overrides(config)
    resolved_config = Path(os.environ["TASK_RUN_DIR"]) / "resolved_config.yaml"
    if int(os.environ.get("RANK", "0")) == 0:
        resolved_config.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, resolved_config, resolve=True)
    run = init_wandb(args)
    patch_sam2_training_runtime(run)
    succeeded = False
    started_at = time.time()
    status_path = Path(os.environ["TASK_RUN_DIR"]) / "training_status.json"
    previous_elapsed = 0.0
    if status_path.is_file():
        previous_elapsed = float(
            json.loads(status_path.read_text(encoding="utf-8")).get(
                "elapsed_seconds", 0.0
            )
        )
    try:
        trainer = instantiate(config.trainer, _recursive_=False)
        model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        parameters = list(model.parameters())
        model_summary = {
            "total_parameters": int(sum(value.numel() for value in parameters)),
            "trainable_parameters": int(
                sum(value.numel() for value in parameters if value.requires_grad)
            ),
            "trainable_tensors": int(
                sum(1 for value in parameters if value.requires_grad)
            ),
        }
        if trainer.train_dataset is not None:
            model_summary["train_dataset_samples"] = int(
                sum(len(dataset) for dataset in trainer.train_dataset.datasets)
            )
            model_summary["optimizer_updates_per_epoch"] = int(
                len(trainer.train_dataset.get_loader(epoch=int(trainer.epoch)))
            )
        if int(os.environ.get("RANK", "0")) == 0:
            (status_path.parent / "training_model_summary.json").write_text(
                json.dumps(model_summary, indent=2) + "\n", encoding="utf-8"
            )
            logging.info("Training data summary: %s", model_summary)
            if run is not None:
                run.summary.update(
                    {
                        "data/train_samples": model_summary.get(
                            "train_dataset_samples", 0
                        ),
                        "train/planned_updates_per_epoch": model_summary.get(
                            "optimizer_updates_per_epoch", 0
                        ),
                    }
                )
        trainer.run()
        succeeded = True
    finally:
        if int(os.environ.get("RANK", "0")) == 0:
            status_path.write_text(
                json.dumps(
                    {
                        "status": "complete" if succeeded else "failed",
                        "started_at_unix": started_at,
                        "finished_at_unix": time.time(),
                        "elapsed_seconds": previous_elapsed
                        + time.time()
                        - started_at,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if run is not None:
            run.summary["system/training_complete"] = int(succeeded)
            run.finish(exit_code=0 if succeeded else 1)


if __name__ == "__main__":
    main()
