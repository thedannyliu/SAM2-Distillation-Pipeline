#!/usr/bin/env python3
"""Train TinyViT Stage 1 with online SAM2 teacher feature distillation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sam2_distill.models.tinyvit_adapter import TinyViTSAM2Adapter
from sam2_distill.training.stage1_losses import stage1_feature_distillation_loss


def init_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def is_main(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def read_manifest(path: Path, split: str, max_items: int | None = None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("manifest must be .parquet or .csv")
    df = df[df["split"] == split].reset_index(drop=True)
    if max_items:
        df = df.head(max_items).copy()
    return df


class ImagePathDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame) -> None:
        from sam2.utils.transforms import SAM2Transforms

        self.manifest = manifest
        self.transforms = SAM2Transforms(resolution=1024, mask_threshold=0.0)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str]:
        row = self.manifest.iloc[idx]
        path = str(row["image_path"])
        with Image.open(path) as image:
            image_tensor = self.transforms(image.convert("RGB"))
        return image_tensor, path, str(row["sample_id"])


def collate_image_paths(batch):
    images, paths, sample_ids = zip(*batch)
    return torch.stack(list(images), dim=0), list(paths), list(sample_ids)


def read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_teacher(config: str, checkpoint: Path, device: torch.device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(config, str(checkpoint), device=str(device), mode="eval")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return SAM2ImagePredictor(model)


def teacher_features_from_paths(predictor, paths: list[str], device: torch.device, amp_dtype: str) -> dict[str, torch.Tensor]:
    images = [read_rgb(path) for path in paths]
    with torch.inference_mode(), autocast_context(device, amp_dtype):
        predictor.set_image_batch(images)
        features = predictor._features
        high_res = features["high_res_feats"]
        teacher_features = {
            "image_embed": features["image_embed"].detach(),
            "high_res_s0": high_res[0].detach(),
            "high_res_s1": high_res[1].detach(),
        }
    return {name: tensor.clone() for name, tensor in teacher_features.items()}


def reduce_metrics(metrics: dict[str, torch.Tensor], world_size: int) -> dict[str, float]:
    reduced = {}
    for key, value in metrics.items():
        tensor = value.detach().float()
        if world_size > 1:
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            tensor /= world_size
        reduced[key] = float(tensor.cpu())
    return reduced


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    module = unwrap_model(model)
    for param in module.backbone.parameters():
        param.requires_grad_(trainable)


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (param for param in model.parameters() if param.requires_grad)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def lr_for_step(step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, float(step + 1) / float(warmup_steps))


def grad_norm(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    grads = [param.grad.detach().float().norm(2) for param in parameters if param.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(grads), ord=2)


def autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[amp_dtype])


def compute_loss(
    student: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return stage1_feature_distillation_loss(
        student,
        teacher,
        lambda_mse=args.lambda_mse,
        lambda_l1=args.lambda_l1,
        lambda_cos=args.lambda_cos,
        lambda_hr=args.lambda_hr,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    run_id: str | None,
    args: argparse.Namespace,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module = unwrap_model(model)
    torch.save(
        {
            "step": step,
            "model_state": module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "wandb_run_id": run_id,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--teacher-config", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--tinyvit-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", default="tiny_vit_21m_512.dist_in22k_ft_in1k")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--projection-warmup-steps", type=int, default=2000)
    parser.add_argument("--lr-warmup-steps", type=int, default=2000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val_sa1b")
    parser.add_argument("--max-train-items", type=int)
    parser.add_argument("--max-val-items", type=int)
    parser.add_argument("--lambda-mse", type=float, default=1.0)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-cos", type=float, default=0.0)
    parser.add_argument("--lambda-hr", type=float, default=1.0)
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--teacher-amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--val-max-batches", type=int, default=25)
    parser.add_argument("--resume", help="Checkpoint to resume.")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "sam2-distill-stage1-online"))
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID"))
    parser.add_argument("--wandb-name", default="sa1b-online-teacher-tinyvit21m")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def validate_input_paths(args: argparse.Namespace) -> None:
    missing = [
        str(path)
        for path in (
            Path(args.manifest),
            Path(args.teacher_checkpoint),
            Path(args.tinyvit_checkpoint),
        )
        if not path.exists()
    ]
    if missing:
        raise SystemExit("Missing required input file(s):\n  " + "\n  ".join(missing))


def main() -> None:
    args = parse_args()
    validate_input_paths(args)
    rank, world_size, _, device = init_distributed()
    out_dir = Path(args.out_dir).expanduser().resolve()
    tb_dir = out_dir / "tensorboard"
    ckpt_dir = out_dir / "checkpoints"

    train_df = read_manifest(Path(args.manifest), args.train_split, args.max_train_items)
    val_df = read_manifest(Path(args.manifest), args.val_split, args.max_val_items)
    if train_df.empty:
        raise SystemExit(f"No rows found for train split {args.train_split!r}")
    if val_df.empty:
        raise SystemExit(f"No rows found for val split {args.val_split!r}")
    train_dataset = ImagePathDataset(train_df)
    val_dataset = ImagePathDataset(val_df)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_image_paths,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_image_paths,
    )

    teacher = load_teacher(args.teacher_config, Path(args.teacher_checkpoint), device)
    model = TinyViTSAM2Adapter(
        model_name=args.model_name,
        checkpoint_path=args.tinyvit_checkpoint,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    wandb_run_id = args.wandb_run_id
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = int(ckpt["step"])
        wandb_run_id = wandb_run_id or ckpt.get("wandb_run_id")
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    initial_backbone_trainable = start_step >= args.projection_warmup_steps
    set_backbone_trainable(model, initial_backbone_trainable)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=args.projection_warmup_steps > start_step,
        )

    if is_main(rank):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(tb_dir))
    else:
        writer = None
    wandb_run = None
    global_batch_size = args.batch_size * world_size
    if is_main(rank) and not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            id=wandb_run_id,
            resume="allow" if wandb_run_id else None,
            dir=str(out_dir / "wandb"),
            config=vars(args),
        )
        wandb_run_id = wandb_run.id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "wandb_run.json").write_text(
            json.dumps({"run_id": wandb_run_id, "url": wandb_run.url, "project": args.wandb_project}) + "\n"
        )

    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
        print(
            "\n".join(
                [
                    "Stage 1 online-teacher training summary",
                    f"  manifest: {Path(args.manifest).expanduser().resolve()}",
                    f"  teacher_config: {args.teacher_config}",
                    f"  teacher_checkpoint: {Path(args.teacher_checkpoint).expanduser().resolve()}",
                    f"  tinyvit_checkpoint: {Path(args.tinyvit_checkpoint).expanduser().resolve()}",
                    f"  out_dir: {out_dir}",
                    f"  train_images: {len(train_dataset):,}",
                    f"  val_images: {len(val_dataset):,}",
                    f"  world_size: {world_size}",
                    f"  batch_size_per_gpu: {args.batch_size}",
                    f"  global_batch_size: {global_batch_size}",
                    f"  max_steps: {args.max_steps:,}",
                    "  teacher_feature_storage: none",
                ]
            ),
            flush=True,
        )

    step = start_step
    model.train()
    backbone_is_trainable = initial_backbone_trainable
    train_start_time = time.time()
    while step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(step)
        for images, paths, _sample_ids in train_loader:
            if step >= args.max_steps:
                break
            should_train_backbone = step >= args.projection_warmup_steps
            if should_train_backbone != backbone_is_trainable:
                set_backbone_trainable(model, should_train_backbone)
                backbone_is_trainable = should_train_backbone
            current_lr = lr_for_step(step, args.lr, args.lr_warmup_steps)
            set_lr(optimizer, current_lr)
            start_time = time.time()
            images = images.to(device, non_blocking=True)
            teacher_start = time.time()
            teacher_targets = teacher_features_from_paths(teacher, paths, device, args.teacher_amp_dtype)
            teacher_sec = time.time() - teacher_start
            with autocast_context(device, args.amp_dtype):
                student = model(images)
                loss, metrics = compute_loss(student, teacher_targets, args)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss.detach()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                clipped_grad_norm = torch.nn.utils.clip_grad_norm_(list(trainable_parameters(model)), args.max_grad_norm)
            else:
                clipped_grad_norm = grad_norm(trainable_parameters(model))
            optimizer.step()

            reduced = reduce_metrics(metrics, world_size)
            reduced["train/lr"] = current_lr
            reduced["train/sec_per_step"] = time.time() - start_time
            reduced["train/teacher_sec_per_step"] = teacher_sec
            reduced["train/backbone_trainable"] = float(should_train_backbone)
            reduced["train/grad_norm"] = float(clipped_grad_norm.detach().float().cpu())
            images_seen = (step + 1) * global_batch_size
            elapsed = time.time() - train_start_time
            avg_wall = elapsed / max(step - start_step + 1, 1)
            eta_hours = max(args.max_steps - step - 1, 0) * avg_wall / 3600.0
            reduced["train/images_seen"] = float(images_seen)
            reduced["train/epoch"] = float(images_seen / max(len(train_dataset), 1))
            reduced["train/progress_pct"] = 100.0 * float(step + 1) / float(max(args.max_steps, 1))
            reduced["train/avg_wall_sec_per_step"] = avg_wall
            reduced["train/eta_hours"] = eta_hours
            if is_main(rank) and step % args.log_every == 0:
                for key, value in reduced.items():
                    if writer:
                        writer.add_scalar(key, value, step)
                if wandb_run:
                    wandb_run.log(reduced, step=step)
                print(
                    " | ".join(
                        [
                            f"step {step + 1:,}/{args.max_steps:,}",
                            f"progress {reduced['train/progress_pct']:.2f}%",
                            f"epoch {reduced['train/epoch']:.3f}",
                            f"loss {reduced['loss_stage1_total']:.6f}",
                            f"mse {reduced['loss_image_mse']:.6f}",
                            f"hr_mse {reduced['loss_high_res_mse']:.6f}",
                            f"teacher {teacher_sec:.3f}s",
                            f"wall {reduced['train/sec_per_step']:.3f}s",
                            f"eta {eta_hours:.2f}h",
                        ]
                    ),
                    flush=True,
                )

            if is_main(rank) and step > 0 and step % args.save_every == 0:
                save_checkpoint(ckpt_dir / f"step_{step:07d}.pt", model, optimizer, step + 1, wandb_run_id, args, best_val_loss)
                save_checkpoint(ckpt_dir / "last.pt", model, optimizer, step + 1, wandb_run_id, args, best_val_loss)
            step += 1

    if is_main(rank):
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, step, wandb_run_id, args, best_val_loss)
        if writer:
            writer.close()
        if wandb_run:
            wandb_run.finish()
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
