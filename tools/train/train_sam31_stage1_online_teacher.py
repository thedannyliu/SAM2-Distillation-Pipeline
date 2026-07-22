#!/usr/bin/env python3
"""Distill the SAM3.1 raw vision trunk into a TinyViT image encoder."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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

from sam2_distill.models.sam31_teacher import SAM31VisionTeacher
from sam2_distill.models.tinyvit_sam3_adapter import TinyViTSAM3Adapter
from sam2_distill.training.sam31_stage1_losses import sam31_feature_distillation_loss


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


def set_seed(seed: int, rank: int) -> None:
    value = (int(seed) + int(rank)) % (2**32)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def read_manifest(
    path: Path,
    split: str,
    max_items: int | None,
    seed: int,
) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("manifest must be .parquet or .csv")
    frame = frame[frame["split"] == split].reset_index(drop=True)
    if max_items and len(frame) > max_items:
        frame = frame.sample(n=max_items, random_state=seed).reset_index(drop=True)
    return frame


def validate_manifest_images(manifest: pd.DataFrame, split: str) -> None:
    missing = [str(path) for path in manifest["image_path"] if not Path(path).is_file()]
    if missing:
        examples = "\n  ".join(missing[:10])
        raise SystemExit(
            f"Manifest split {split!r} has {len(missing):,} unavailable images. "
            "The mounted dataset version may have changed. First examples:\n  "
            f"{examples}"
        )


class SAM3ImageDataset(Dataset):
    """Official SAM3 square-resize and [-1, 1] image preprocessing."""

    def __init__(self, manifest: pd.DataFrame, resolution: int = 1008) -> None:
        self.manifest = manifest
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        row = self.manifest.iloc[index]
        path = str(row["image_path"])
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.resolution, self.resolution), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
        tensor.sub_(0.5).div_(0.5)
        return tensor, path, str(row["sample_id"])


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in unwrap_model(model).backbone.parameters():
        parameter.requires_grad_(trainable)


def wrap_ddp(model: nn.Module, device: torch.device) -> DistributedDataParallel:
    return DistributedDataParallel(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )


def autocast_context(device: torch.device, dtype: str):
    if device.type != "cuda" or dtype == "none":
        return nullcontext()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def teacher_features(
    teacher: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
) -> torch.Tensor:
    with torch.inference_mode(), autocast_context(device, amp_dtype):
        feature = teacher(images)
    return feature.detach().clone()


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def learning_rate_for_step(
    step: int,
    max_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(max_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def reduce_metrics(
    metrics: dict[str, torch.Tensor], world_size: int
) -> dict[str, float]:
    keys = list(metrics)
    values = torch.stack([metrics[key].detach().float() for key in keys])
    if world_size > 1:
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        values /= world_size
    return {key: float(value) for key, value in zip(keys, values.cpu())}


def evaluate(
    model: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    world_size: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, torch.Tensor] = {}
    count = torch.zeros((), device=device, dtype=torch.float64)
    for batch_index, (images, _paths, _sample_ids) in enumerate(loader):
        if args.val_max_batches and batch_index >= args.val_max_batches:
            break
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]
        target = teacher_features(teacher, images, device, args.teacher_amp_dtype)
        with torch.no_grad(), autocast_context(device, args.amp_dtype):
            prediction = model(images)
            _, metrics = sam31_feature_distillation_loss(
                prediction,
                target,
                lambda_mse=args.lambda_mse,
                lambda_cos=args.lambda_cos,
                lambda_relation=args.lambda_relation,
                relation_grid_size=args.relation_grid_size,
            )
        for key, value in metrics.items():
            weighted = value.detach().double() * batch_size
            totals[key] = totals.get(key, torch.zeros_like(weighted)) + weighted
        count += batch_size
    if world_size > 1:
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
        for value in totals.values():
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    model.train()
    denominator = max(float(count.cpu()), 1.0)
    result = {f"val/{key}": float(value.cpu()) / denominator for key, value in totals.items()}
    result["val/num_images"] = float(count.cpu())
    return result


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    wandb_run_id: str | None,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state": unwrap_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "wandb_run_id": wandb_run_id,
            "args": vars(args),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--tinyvit-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", default="tiny_vit_21m_512.dist_in22k_ft_in1k")
    parser.add_argument("--adapter-mode", choices=("projection", "residual_dwconv"), default="residual_dwconv")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val_sav")
    parser.add_argument("--max-train-items", type=int)
    parser.add_argument("--max-val-items", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--projection-warmup-steps", type=int, default=2000)
    parser.add_argument("--lr-warmup-steps", type=int, default=2000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lambda-mse", type=float, default=1.0)
    parser.add_argument("--lambda-cos", type=float, default=0.25)
    parser.add_argument("--lambda-relation", type=float, default=0.0)
    parser.add_argument("--relation-grid-size", type=int, default=18)
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--teacher-amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=310107256)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--print-every", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--val-max-batches", type=int, default=0)
    parser.add_argument("--resume")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "sam31-distill-stage1"))
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID"))
    parser.add_argument("--wandb-name", default="sam31-tv21m-adapter-mse-cos025")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    missing = [
        path
        for path in (args.manifest, args.teacher_checkpoint, args.tinyvit_checkpoint)
        if not Path(path).is_file()
    ]
    if missing:
        raise SystemExit("Missing required file(s):\n  " + "\n  ".join(missing))


def main() -> None:
    args = parse_args()
    if isinstance(args.wandb_run_id, str):
        args.wandb_run_id = args.wandb_run_id.strip() or None
    if not args.wandb_run_id:
        os.environ.pop("WANDB_RUN_ID", None)
    validate_paths(args)
    rank, world_size, _, device = init_distributed()
    set_seed(args.seed, rank)

    out_dir = Path(args.out_dir).expanduser().resolve()
    checkpoint_dir = out_dir / "checkpoints"
    train_frame = read_manifest(Path(args.manifest), args.train_split, args.max_train_items, args.seed)
    val_frame = read_manifest(Path(args.manifest), args.val_split, args.max_val_items, args.seed)
    if train_frame.empty or val_frame.empty:
        raise SystemExit(
            f"Empty split: train={len(train_frame)}, val={len(val_frame)} "
            f"for {args.train_split!r}/{args.val_split!r}"
        )
    validate_manifest_images(val_frame, args.val_split)

    train_dataset = SAM3ImageDataset(train_frame)
    val_dataset = SAM3ImageDataset(val_frame)
    train_sampler = (
        DistributedSampler(train_dataset, world_size, rank, shuffle=True, seed=args.seed)
        if world_size > 1
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, world_size, rank, shuffle=False)
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    teacher = SAM31VisionTeacher(args.teacher_checkpoint).to(device).eval()
    model = TinyViTSAM3Adapter(
        model_name=args.model_name,
        checkpoint_path=args.tinyvit_checkpoint,
        adapter_mode=args.adapter_mode,
        freeze_backbone_bn=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    best_val_loss = float("inf")
    wandb_run_id = args.wandb_run_id
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        step = int(checkpoint["step"])
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        wandb_run_id = wandb_run_id or checkpoint.get("wandb_run_id")
    if args.resume and not args.no_wandb and not wandb_run_id:
        raise RuntimeError(
            f"Cannot resume {args.resume} with W&B enabled: checkpoint has no wandb_run_id"
        )
    start_step = step

    backbone_trainable = step >= args.projection_warmup_steps
    set_backbone_trainable(model, backbone_trainable)
    if world_size > 1:
        model = wrap_ddp(model, device)
    model.train()

    writer = None
    wandb_run = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter

        out_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(out_dir / "tensorboard"))
        if not args.no_wandb:
            import wandb

            (out_dir / "wandb").mkdir(exist_ok=True)
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                id=wandb_run_id,
                resume="allow" if wandb_run_id else None,
                dir=str(out_dir / "wandb"),
                config=vars(args),
            )
            wandb_run_id = wandb_run.id
            (out_dir / "wandb_run.json").write_text(
                json.dumps(
                    {
                        "entity": wandb_run.entity,
                        "run_id": wandb_run.id,
                        "url": wandb_run.url,
                        "project": args.wandb_project,
                    },
                    indent=2,
                )
                + "\n"
            )
        (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
        print(
            "\n".join(
                [
                    "SAM3.1 Stage 1 online-teacher training",
                    f"  teacher_checkpoint: {args.teacher_checkpoint}",
                    f"  teacher_trunk_prefix: {teacher.checkpoint_prefix}",
                    f"  teacher_target: [B, 1024, 72, 72]",
                    f"  preprocessing: resize 1008x1008, mean/std 0.5",
                    f"  student: {args.model_name}",
                    f"  adapter_mode: {args.adapter_mode}",
                    f"  train_images: {len(train_dataset):,}",
                    f"  val_images: {len(val_dataset):,}",
                    f"  world_size: {world_size}",
                    f"  global_batch_size: {args.batch_size * world_size}",
                    f"  max_steps: {args.max_steps:,}",
                    f"  loss: {args.lambda_mse} * MSE + {args.lambda_cos} * cosine + "
                    f"{args.lambda_relation} * spatial_relation",
                    f"  out_dir: {out_dir}",
                ]
            ),
            flush=True,
        )

    global_batch_size = args.batch_size * world_size
    train_start = time.time()
    steps_per_data_epoch = len(train_loader)
    data_epoch = step // steps_per_data_epoch
    resume_batch_offset = step % steps_per_data_epoch
    while step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(data_epoch)
        for batch_index, (images, _paths, _sample_ids) in enumerate(train_loader):
            if batch_index < resume_batch_offset:
                continue
            if step >= args.max_steps:
                break
            should_train_backbone = step >= args.projection_warmup_steps
            if should_train_backbone != backbone_trainable:
                if world_size > 1:
                    torch.distributed.barrier()
                    raw_model = unwrap_model(model)
                    set_backbone_trainable(raw_model, should_train_backbone)
                    model = wrap_ddp(raw_model, device)
                else:
                    set_backbone_trainable(model, should_train_backbone)
                model.train()
                backbone_trainable = should_train_backbone

            images = images.to(device, non_blocking=True)
            iteration_start = time.time()
            teacher_start = time.time()
            target = teacher_features(teacher, images, device, args.teacher_amp_dtype)
            teacher_seconds = time.time() - teacher_start
            with autocast_context(device, args.amp_dtype):
                prediction = model(images)
                loss, metrics = sam31_feature_distillation_loss(
                    prediction,
                    target,
                    lambda_mse=args.lambda_mse,
                    lambda_cos=args.lambda_cos,
                    lambda_relation=args.lambda_relation,
                    relation_grid_size=args.relation_grid_size,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.detach()}")

            lr = learning_rate_for_step(
                step, args.max_steps, args.lr, args.min_lr, args.lr_warmup_steps
            )
            set_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(trainable_parameters(model)), args.max_grad_norm
            )
            optimizer.step()

            completed_step = step + 1
            reduced = reduce_metrics(metrics, world_size)
            elapsed = time.time() - train_start
            average_seconds = elapsed / max(completed_step - start_step, 1)
            images_seen = completed_step * global_batch_size
            reduced.update(
                {
                    "train/lr": lr,
                    "train/global_step": float(completed_step),
                    "train/images_seen": float(images_seen),
                    "train/epoch": images_seen / max(len(train_dataset), 1),
                    "train/progress_pct": 100.0 * completed_step / args.max_steps,
                    "train/sec_per_step": time.time() - iteration_start,
                    "train/teacher_sec_per_step": teacher_seconds,
                    "train/grad_norm": float(grad_norm.detach().cpu()),
                    "train/backbone_trainable": float(should_train_backbone),
                    "train/eta_hours": (args.max_steps - completed_step) * average_seconds / 3600.0,
                }
            )
            if rank == 0 and step % args.log_every == 0:
                for key, value in reduced.items():
                    writer.add_scalar(key, value, completed_step)
                if wandb_run:
                    wandb_run.log(reduced, step=completed_step)
            if rank == 0 and step % args.print_every == 0:
                print(
                    " | ".join(
                        [
                            f"step {completed_step:,}/{args.max_steps:,}",
                            f"epoch {reduced['train/epoch']:.3f}",
                            f"loss {reduced['loss_stage1_total']:.6f}",
                            f"mse {reduced['loss_feature_mse']:.6f}",
                            f"cos {reduced['loss_feature_cos']:.6f}",
                            f"relation {reduced['loss_spatial_relation']:.6f}",
                            f"teacher {teacher_seconds:.3f}s",
                            f"wall {reduced['train/sec_per_step']:.3f}s",
                            f"eta {reduced['train/eta_hours']:.2f}h",
                        ]
                    ),
                    flush=True,
                )

            should_evaluate = completed_step % args.eval_every == 0
            should_save = completed_step % args.save_every == 0
            if rank == 0 and (should_evaluate or should_save):
                save_checkpoint(
                    checkpoint_dir / "last.pt",
                    model,
                    optimizer,
                    completed_step,
                    best_val_loss,
                    wandb_run_id,
                    args,
                )
            if should_evaluate:
                val_metrics = evaluate(model, teacher, val_loader, device, world_size, args)
                if rank == 0:
                    for key, value in val_metrics.items():
                        writer.add_scalar(key, value, completed_step)
                    if wandb_run:
                        wandb_run.log(val_metrics, step=completed_step)
                    val_loss = val_metrics["val/loss_stage1_total"]
                    print(
                        f"val step {completed_step:,} | loss {val_loss:.6f} | "
                        f"mse {val_metrics['val/loss_feature_mse']:.6f} | "
                        f"cos {val_metrics['val/loss_feature_cos']:.6f}",
                        flush=True,
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            checkpoint_dir / "best.pt",
                            model,
                            optimizer,
                            completed_step,
                            best_val_loss,
                            wandb_run_id,
                            args,
                        )
                        save_checkpoint(
                            checkpoint_dir / "last.pt",
                            model,
                            optimizer,
                            completed_step,
                            best_val_loss,
                            wandb_run_id,
                            args,
                        )
            step = completed_step
        data_epoch += 1
        resume_batch_offset = 0

    final_metrics = evaluate(model, teacher, val_loader, device, world_size, args)
    if rank == 0:
        final_loss = final_metrics["val/loss_stage1_total"]
        if final_loss < best_val_loss:
            best_val_loss = final_loss
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                step,
                best_val_loss,
                wandb_run_id,
                args,
            )
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            step,
            best_val_loss,
            wandb_run_id,
            args,
        )
        for key, value in final_metrics.items():
            writer.add_scalar(key, value, step)
        if wandb_run:
            wandb_run.log(final_metrics, step=step)
            wandb_run.finish()
        writer.close()
        print(f"Training complete: {checkpoint_dir}", flush=True)

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
