#!/usr/bin/env python
"""Run a tiny EdgeTAM/SAM2 Trainer smoke from a repo YAML config."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sam2-training-root", type=Path, required=True)
    parser.add_argument("--edgetam-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-num-objects", type=int, default=1)
    parser.add_argument("--image-encoder-forward-batch-size", type=int, default=0)
    parser.add_argument("--image-encoder-activation-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=250107256)
    return parser.parse_args()


def add_import_roots(edgetam_root: Path, sam2_training_root: Path) -> None:
    for root in (sam2_training_root, edgetam_root):
        if not root.exists():
            raise FileNotFoundError(root)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(sam2_training_root))
    sys.path.insert(0, str(edgetam_root))


def set_single_process_dist_env() -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(random.randint(10000, 65000)))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


def read_checkpoint_summary(checkpoint_path: Path) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {
        "epoch": int(checkpoint.get("epoch", -1)),
        "steps": {key: int(value) for key, value in checkpoint.get("steps", {}).items()},
    }


def main() -> None:
    args = parse_args()
    add_import_roots(args.edgetam_root, args.sam2_training_root)
    set_single_process_dist_env()

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from training.utils.train_utils import register_omegaconf_resolvers

    try:
        register_omegaconf_resolvers()
    except ValueError:
        pass

    cfg = OmegaConf.load(args.config)
    cfg.scratch.num_epochs = args.max_epochs
    cfg.scratch.phases_per_epoch = 1
    cfg.scratch.num_train_workers = args.num_workers
    cfg.scratch.num_frames = args.num_frames
    cfg.scratch.max_num_objects = args.max_num_objects
    cfg.trainer.max_epochs = args.max_epochs
    cfg.trainer.seed_value = args.seed
    cfg.trainer.data.train.num_workers = args.num_workers
    cfg.trainer.data.train.pin_memory = True
    cfg.trainer.data.train.drop_last = False
    cfg.trainer.model.num_init_cond_frames_for_train = 1
    cfg.trainer.model.rand_init_cond_frames_for_train = False
    cfg.trainer.model.num_frames_to_correct_for_train = 1
    cfg.trainer.model.rand_frames_to_correct_for_train = False
    cfg.trainer.model.num_correction_pt_per_frame = 1
    cfg.trainer.model.image_encoder_forward_batch_size = (
        args.image_encoder_forward_batch_size if args.image_encoder_forward_batch_size > 0 else None
    )
    cfg.trainer.model.image_encoder_activation_checkpoint = args.image_encoder_activation_checkpoint
    cfg.trainer.logging.log_freq = 1
    cfg.trainer.logging.log_scalar_frequency = 1
    cfg.trainer.logging.log_dir = str(args.out_dir / "logs")
    cfg.trainer.logging.tensorboard_writer.log_dir = str(args.out_dir / "tensorboard")
    cfg.trainer.checkpoint.save_dir = str(args.out_dir / "checkpoints")
    cfg.launcher.experiment_log_dir = str(args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "checkpoints" / "checkpoint.pt"
    checkpoint_before = read_checkpoint_summary(checkpoint_path)
    (args.out_dir / "config_resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()
    checkpoint_after = read_checkpoint_summary(checkpoint_path)

    summary = {
        "result": "pass",
        "config": str(args.config),
        "out_dir": str(args.out_dir),
        "max_epochs": args.max_epochs,
        "num_frames": args.num_frames,
        "max_num_objects": args.max_num_objects,
        "image_encoder_forward_batch_size": args.image_encoder_forward_batch_size,
        "image_encoder_activation_checkpoint": args.image_encoder_activation_checkpoint,
        "seed": args.seed,
        "checkpoint_before": checkpoint_before,
        "checkpoint_after": checkpoint_after,
        "resumed": checkpoint_before is not None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
