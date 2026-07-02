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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--dataset-mode", choices=("vos", "sa1b-image", "sav-json"), default="vos")
    parser.add_argument("--vos-image-root", type=Path)
    parser.add_argument("--vos-gt-root", type=Path)
    parser.add_argument("--vos-file-list", type=Path)
    parser.add_argument("--sav-image-root", type=Path)
    parser.add_argument("--sav-ann-root", type=Path)
    parser.add_argument("--sav-file-list", type=Path)
    parser.add_argument("--sav-ann-every", type=int, default=4)
    parser.add_argument("--sa1b-image-root", type=Path, default=Path("data/edgetam_smoke/sa1b_smoke/images/train"))
    parser.add_argument("--sa1b-ann-root", type=Path, default=Path("data/edgetam_smoke/sa1b_smoke/annotations/train"))
    parser.add_argument("--sa1b-file-list", type=Path)
    parser.add_argument("--sa1b-max-items", type=int, default=2)
    parser.add_argument("--tinyvit-checkpoint", type=Path)
    parser.add_argument("--image-encoder-forward-batch-size", type=int, default=0)
    parser.add_argument("--image-encoder-activation-checkpoint", action="store_true")
    parser.add_argument("--freeze-image-encoder", action="store_true")
    parser.add_argument(
        "--trainable-module-mode",
        choices=("image_neck_only", "image_encoder_only"),
        help="Freeze all modules except the selected image-encoder part.",
    )
    parser.add_argument("--lambda-img", type=float)
    parser.add_argument("--lambda-mem", type=float)
    parser.add_argument("--teacher-feature-cache", type=Path)
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


def configure_sa1b_image_mode(cfg: Any, args: argparse.Namespace) -> None:
    if args.sa1b_file_list is None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        image_stems = sorted(path.stem for path in args.sa1b_image_root.glob("*.jpg"))
        if not image_stems:
            raise FileNotFoundError(f"No SA-1B smoke images found under {args.sa1b_image_root}")
        image_stems = image_stems[: args.sa1b_max_items]
        file_list = args.out_dir / "sa1b_image_file_list.txt"
        file_list.write_text("\n".join(image_stems) + "\n", encoding="utf-8")
    else:
        file_list = args.sa1b_file_list

    cfg.scratch.num_frames = 1
    cfg.scratch.max_num_objects = args.max_num_objects
    cfg.trainer.model.num_init_cond_frames_for_train = 1
    cfg.trainer.model.rand_init_cond_frames_for_train = False
    cfg.trainer.model.num_frames_to_correct_for_train = 1
    cfg.trainer.model.rand_frames_to_correct_for_train = False
    cfg.trainer.model.num_correction_pt_per_frame = 1
    cfg.trainer.loss.all.lambda_img = 1.0
    cfg.trainer.loss.all.lambda_mem = 0.0

    dataset_cfg = cfg.trainer.data.train.datasets[0].dataset.datasets[0]
    dataset_cfg.video_dataset._target_ = "training.dataset.vos_raw_dataset.SA1BRawDataset"
    dataset_cfg.video_dataset.img_folder = str(args.sa1b_image_root)
    dataset_cfg.video_dataset.gt_folder = str(args.sa1b_ann_root)
    dataset_cfg.video_dataset.file_list_txt = str(file_list)
    if "ann_every" in dataset_cfg.video_dataset:
        del dataset_cfg.video_dataset.ann_every
    dataset_cfg.sampler.num_frames = 1
    dataset_cfg.sampler.max_num_objects = args.max_num_objects


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
    cfg.scratch.train_batch_size = args.batch_size
    cfg.scratch.resolution = args.resolution
    cfg.trainer.max_epochs = args.max_epochs
    cfg.trainer.seed_value = args.seed
    cfg.trainer.data.train.batch_sizes = [args.batch_size]
    cfg.trainer.data.train.num_workers = args.num_workers
    cfg.trainer.data.train.pin_memory = True
    cfg.trainer.data.train.drop_last = False
    cfg.trainer.model.num_init_cond_frames_for_train = 1
    cfg.trainer.model.rand_init_cond_frames_for_train = False
    cfg.trainer.model.num_frames_to_correct_for_train = 1
    cfg.trainer.model.rand_frames_to_correct_for_train = False
    cfg.trainer.model.num_correction_pt_per_frame = 1
    if args.dataset_mode == "sa1b-image":
        configure_sa1b_image_mode(cfg, args)
    elif args.dataset_mode == "sav-json":
        dataset_cfg = cfg.trainer.data.train.datasets[0].dataset.datasets[0]
        dataset_cfg.video_dataset._target_ = "training.dataset.vos_raw_dataset.JSONRawDataset"
        if args.sav_image_root is None or args.sav_ann_root is None or args.sav_file_list is None:
            raise ValueError("--sav-image-root, --sav-ann-root, and --sav-file-list are required for sav-json")
        dataset_cfg.video_dataset.img_folder = str(args.sav_image_root)
        dataset_cfg.video_dataset.gt_folder = str(args.sav_ann_root)
        dataset_cfg.video_dataset.file_list_txt = str(args.sav_file_list)
        dataset_cfg.video_dataset.ann_every = args.sav_ann_every
        dataset_cfg.sampler.num_frames = args.num_frames
        dataset_cfg.sampler.max_num_objects = args.max_num_objects
    else:
        dataset_cfg = cfg.trainer.data.train.datasets[0].dataset.datasets[0]
        if args.vos_image_root is not None:
            dataset_cfg.video_dataset.img_folder = str(args.vos_image_root)
        if args.vos_gt_root is not None:
            dataset_cfg.video_dataset.gt_folder = str(args.vos_gt_root)
        if args.vos_file_list is not None:
            dataset_cfg.video_dataset.file_list_txt = str(args.vos_file_list)
    if args.tinyvit_checkpoint is not None:
        cfg.trainer.model.image_encoder.trunk.checkpoint_path = str(args.tinyvit_checkpoint)
    cfg.trainer.model.image_encoder_forward_batch_size = (
        args.image_encoder_forward_batch_size if args.image_encoder_forward_batch_size > 0 else None
    )
    cfg.trainer.model.image_encoder_activation_checkpoint = args.image_encoder_activation_checkpoint
    cfg.trainer.model.freeze_image_encoder = args.freeze_image_encoder
    cfg.trainer.model.trainable_module_mode = args.trainable_module_mode
    if args.lambda_img is not None:
        cfg.trainer.loss.all.lambda_img = args.lambda_img
    if args.lambda_mem is not None:
        cfg.trainer.loss.all.lambda_mem = args.lambda_mem
    if args.teacher_feature_cache is not None:
        cfg.trainer.model.synthetic_teacher = False
        cfg.trainer.model.teacher_feature_cache_path = str(args.teacher_feature_cache)
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
    trainable_summary_before = {
        "total_parameters": int(sum(param.numel() for param in trainer.model.parameters())),
        "trainable_parameters": int(
            sum(param.numel() for param in trainer.model.parameters() if param.requires_grad)
        ),
    }
    trainer.run()
    checkpoint_after = read_checkpoint_summary(checkpoint_path)
    trainable_summary_after = {
        "total_parameters": int(sum(param.numel() for param in trainer.model.parameters())),
        "trainable_parameters": int(
            sum(param.numel() for param in trainer.model.parameters() if param.requires_grad)
        ),
    }

    summary = {
        "result": "pass",
        "config": str(args.config),
        "out_dir": str(args.out_dir),
        "max_epochs": args.max_epochs,
        "num_frames": args.num_frames,
        "max_num_objects": args.max_num_objects,
        "batch_size": args.batch_size,
        "resolution": args.resolution,
        "dataset_mode": args.dataset_mode,
        "vos_image_root": str(args.vos_image_root) if args.vos_image_root else None,
        "vos_gt_root": str(args.vos_gt_root) if args.vos_gt_root else None,
        "vos_file_list": str(args.vos_file_list) if args.vos_file_list else None,
        "sav_image_root": str(args.sav_image_root) if args.sav_image_root else None,
        "sav_ann_root": str(args.sav_ann_root) if args.sav_ann_root else None,
        "sav_file_list": str(args.sav_file_list) if args.sav_file_list else None,
        "sav_ann_every": args.sav_ann_every if args.dataset_mode == "sav-json" else None,
        "sa1b_max_items": args.sa1b_max_items if args.dataset_mode == "sa1b-image" else None,
        "tinyvit_checkpoint": str(args.tinyvit_checkpoint) if args.tinyvit_checkpoint else None,
        "image_encoder_forward_batch_size": args.image_encoder_forward_batch_size,
        "image_encoder_activation_checkpoint": args.image_encoder_activation_checkpoint,
        "freeze_image_encoder": args.freeze_image_encoder,
        "trainable_module_mode": args.trainable_module_mode,
        "trainable_summary_before": trainable_summary_before,
        "trainable_summary_after": trainable_summary_after,
        "lambda_img": float(cfg.trainer.loss.all.lambda_img),
        "lambda_mem": float(cfg.trainer.loss.all.lambda_mem),
        "teacher_feature_cache": str(args.teacher_feature_cache) if args.teacher_feature_cache else None,
        "seed": args.seed,
        "checkpoint_before": checkpoint_before,
        "checkpoint_after": checkpoint_after,
        "resumed": checkpoint_before is not None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
