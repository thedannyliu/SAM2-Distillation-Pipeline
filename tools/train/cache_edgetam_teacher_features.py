#!/usr/bin/env python
"""Cache EdgeTAM video teacher features from a real SAM2 trainer forward."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sam2-training-root", type=Path, required=True)
    parser.add_argument("--edgetam-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--max-num-objects", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--dataset-mode", choices=("vos", "sa1b-image"), default="vos")
    parser.add_argument("--sa1b-image-root", type=Path, default=Path("data/edgetam_smoke/sa1b_smoke/images/train"))
    parser.add_argument("--sa1b-ann-root", type=Path, default=Path("data/edgetam_smoke/sa1b_smoke/annotations/train"))
    parser.add_argument("--sa1b-file-list", type=Path)
    parser.add_argument("--sa1b-max-items", type=int, default=1)
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


def configure_sa1b_image_mode(cfg: Any, args: argparse.Namespace) -> None:
    if args.sa1b_file_list is None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        image_stems = sorted(path.stem for path in args.sa1b_image_root.glob("*.jpg"))
        if not image_stems:
            raise FileNotFoundError(f"No SA-1B smoke images found under {args.sa1b_image_root}")
        image_stems = image_stems[: args.sa1b_max_items]
        file_list = args.work_dir / "sa1b_image_teacher_cache_file_list.txt"
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

    dataset_cfg = cfg.trainer.data.train.datasets[0].dataset.datasets[0]
    dataset_cfg.video_dataset._target_ = "training.dataset.vos_raw_dataset.SA1BRawDataset"
    dataset_cfg.video_dataset.img_folder = str(args.sa1b_image_root)
    dataset_cfg.video_dataset.gt_folder = str(args.sa1b_ann_root)
    dataset_cfg.video_dataset.file_list_txt = str(file_list)
    if "ann_every" in dataset_cfg.video_dataset:
        del dataset_cfg.video_dataset.ann_every
    dataset_cfg.sampler.num_frames = 1
    dataset_cfg.sampler.max_num_objects = args.max_num_objects


def configure_teacher_cfg(cfg: Any, args: argparse.Namespace) -> Any:
    cfg.scratch.num_epochs = 1
    cfg.scratch.phases_per_epoch = args.max_batches
    cfg.scratch.num_train_workers = args.num_workers
    cfg.scratch.num_frames = args.num_frames
    cfg.scratch.max_num_objects = args.max_num_objects
    cfg.trainer.max_epochs = 1
    cfg.trainer.mode = "train_only"
    cfg.trainer.seed_value = args.seed
    cfg.trainer.data.train.num_workers = args.num_workers
    cfg.trainer.data.train.pin_memory = True
    cfg.trainer.data.train.drop_last = False
    cfg.trainer.model._target_ = "sam2_distill.edgetam.train_model.EdgeTAMTrain"
    for key in ("teacher_model", "teacher_feature_cache_path", "synthetic_teacher", "synthetic_teacher_offset"):
        if key in cfg.trainer.model:
            del cfg.trainer.model[key]
    cfg.trainer.model.num_init_cond_frames_for_train = 1
    cfg.trainer.model.rand_init_cond_frames_for_train = False
    cfg.trainer.model.num_frames_to_correct_for_train = 1
    cfg.trainer.model.rand_frames_to_correct_for_train = False
    cfg.trainer.model.num_correction_pt_per_frame = 1
    cfg.trainer.model.image_encoder_forward_batch_size = (
        args.image_encoder_forward_batch_size if args.image_encoder_forward_batch_size > 0 else None
    )
    cfg.trainer.model.image_encoder_activation_checkpoint = args.image_encoder_activation_checkpoint
    if args.dataset_mode == "sa1b-image":
        configure_sa1b_image_mode(cfg, args)
    cfg.trainer.logging.log_freq = 1
    cfg.trainer.logging.log_scalar_frequency = 1
    cfg.trainer.logging.log_dir = str(args.work_dir / "logs")
    cfg.trainer.logging.tensorboard_writer.log_dir = str(args.work_dir / "tensorboard")
    cfg.trainer.checkpoint.save_dir = str(args.work_dir / "checkpoints")
    cfg.launcher.experiment_log_dir = str(args.work_dir)
    return cfg


def stack_feature(outputs: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    values = []
    for frame_idx, frame in enumerate(outputs):
        if key not in frame:
            raise KeyError(f"teacher output frame {frame_idx} missing {key}")
        values.append(frame[key].detach().cpu().to(torch.float16))
    return torch.stack(values, dim=0)


def destroy_process_group() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.num_frames < 1 or args.num_frames > 500:
        raise SystemExit("--num-frames must be in [1, 500]")
    if args.max_batches != 1:
        raise SystemExit("Only --max-batches 1 is currently supported for frame-major smoke caches")

    add_import_roots(args.edgetam_root, args.sam2_training_root)
    set_single_process_dist_env()

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from training.trainer import unwrap_ddp_if_wrapped
    from training.utils.train_utils import register_omegaconf_resolvers

    try:
        register_omegaconf_resolvers()
    except ValueError:
        pass

    cfg = OmegaConf.load(args.config)
    cfg = configure_teacher_cfg(cfg, args)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "teacher_cache_config_resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )

    trainer = instantiate(cfg.trainer, _recursive_=False)
    dataloader = trainer.train_dataset.get_loader(epoch=int(trainer.epoch))
    batch = next(iter(dataloader)).to(trainer.device, non_blocking=True)
    model = unwrap_ddp_if_wrapped(trainer.model)
    model.eval()

    with torch.no_grad():
        outputs = model(batch)

    payload = {
        "schema": "edgetam_teacher_feature_cache_v1",
        "source": "real_forward",
        "config": str(args.config),
        "seed": args.seed,
        "num_frames": args.num_frames,
        "max_num_objects": args.max_num_objects,
        "dataset_mode": args.dataset_mode,
        "sa1b_max_items": args.sa1b_max_items if args.dataset_mode == "sa1b-image" else None,
        "teacher_distill_F16": stack_feature(outputs, "distill_F16"),
        "teacher_distill_F_M": stack_feature(outputs, "distill_F_M"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)

    summary = {
        "result": "pass",
        "out": str(args.out),
        "work_dir": str(args.work_dir),
        "config": str(args.config),
        "seed": args.seed,
        "num_frames": args.num_frames,
        "max_num_objects": args.max_num_objects,
        "dataset_mode": args.dataset_mode,
        "sa1b_max_items": args.sa1b_max_items if args.dataset_mode == "sa1b-image" else None,
        "model_class": type(model).__name__,
        "teacher_distill_F16_shape": list(payload["teacher_distill_F16"].shape),
        "teacher_distill_F_M_shape": list(payload["teacher_distill_F_M"].shape),
    }
    (args.work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    destroy_process_group()


if __name__ == "__main__":
    main()
