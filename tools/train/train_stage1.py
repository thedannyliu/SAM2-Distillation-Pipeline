#!/usr/bin/env python3
"""Train TinyViT SAM2 Stage 1 feature distillation from cached teacher features."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import zarr
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

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


def read_manifest(path: Path, split: str) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("manifest must be .parquet or .csv")
    return df[df["split"] == split].reset_index(drop=True)


def load_cache_index(cache_root: Path) -> dict[str, tuple[Path, int]]:
    mapping: dict[str, tuple[Path, int]] = {}
    for shard in sorted(cache_root.glob("shard-*.zarr")):
        index_path = shard / "index.parquet"
        if not index_path.exists():
            continue
        index = pd.read_parquet(index_path)
        for row in index.itertuples(index=False):
            mapping[str(row.sample_id)] = (shard, int(row.row_in_shard))
    if not mapping:
        raise SystemExit(f"No cache index rows found under {cache_root}")
    return mapping


class Stage1CacheDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cache_root: Path) -> None:
        from sam2.utils.transforms import SAM2Transforms

        self.manifest = manifest
        self.cache_index = load_cache_index(cache_root)
        self.transforms = SAM2Transforms(resolution=1024, mask_threshold=0.0)
        self._groups: dict[Path, zarr.Group] = {}

        missing = sorted(set(manifest["sample_id"]) - set(self.cache_index))
        if missing:
            raise SystemExit(f"{len(missing)} manifest samples missing from teacher cache; first={missing[:5]}")

    def __len__(self) -> int:
        return len(self.manifest)

    def group(self, shard: Path) -> zarr.Group:
        if shard not in self._groups:
            self._groups[shard] = zarr.open_group(str(shard), mode="r")
        return self._groups[shard]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row = self.manifest.iloc[idx]
        with Image.open(row["image_path"]) as image:
            image_tensor = self.transforms(image.convert("RGB"))

        shard, row_in_shard = self.cache_index[str(row["sample_id"])]
        group = self.group(shard)
        teacher = {
            "image_embed": torch.from_numpy(group["image_embed"][row_in_shard]).float(),
            "high_res_s0": torch.from_numpy(group["high_res_s0"][row_in_shard]).float(),
            "high_res_s1": torch.from_numpy(group["high_res_s1"][row_in_shard]).float(),
        }
        return image_tensor, teacher


def move_teacher(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


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
    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    return torch.autocast(device_type="cuda", dtype=dtype_by_name[amp_dtype])


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    world_size: int,
    max_batches: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_idx, (images, teacher) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        teacher = move_teacher(teacher, device)
        with torch.no_grad(), autocast_context(device, args.amp_dtype):
            student = model(images)
            _, metrics = compute_loss(student, teacher, args)
        metrics = reduce_metrics(metrics, world_size)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    model.train()
    return {f"val/{key}": value / max(count, 1) for key, value in totals.items()}


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
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--tinyvit-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--projection-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--nonfinite-loss",
        choices=("error", "skip"),
        default="error",
        help="How to handle non-finite losses.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--lambda-mse", type=float, default=1.0)
    parser.add_argument("--lambda-l1", type=float, default=0.5)
    parser.add_argument("--lambda-cos", type=float, default=0.1)
    parser.add_argument("--lambda-hr", type=float, default=1.0)
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--val-max-batches", type=int, default=25)
    parser.add_argument("--resume", help="Checkpoint to resume.")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "sam2-distill-stage1"))
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID"))
    parser.add_argument("--wandb-name", default="coco-pilot-stage1")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, _, device = init_distributed()
    out_dir = Path(args.out_dir).expanduser().resolve()
    tb_dir = out_dir / "tensorboard"
    ckpt_dir = out_dir / "checkpoints"

    train_df = read_manifest(Path(args.manifest), args.train_split)
    val_df = read_manifest(Path(args.manifest), args.val_split)
    if train_df.empty:
        raise SystemExit(f"No rows found for train split {args.train_split!r}")
    if val_df.empty:
        raise SystemExit(f"No rows found for val split {args.val_split!r}")
    train_dataset = Stage1CacheDataset(train_df, Path(args.cache_root))
    val_dataset = Stage1CacheDataset(val_df, Path(args.cache_root))

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

    model = TinyViTSAM2Adapter(checkpoint_path=args.tinyvit_checkpoint).to(device)
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

    writer = SummaryWriter(str(tb_dir)) if is_main(rank) else None
    wandb_run = None
    global_batch_size = args.batch_size * world_size
    steps_per_epoch = max(len(train_loader), 1)
    train_images_per_epoch = steps_per_epoch * global_batch_size
    val_batches = len(val_loader) if args.val_max_batches <= 0 else min(len(val_loader), args.val_max_batches)
    val_images_per_eval = val_batches * global_batch_size
    if is_main(rank) and not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            id=wandb_run_id,
            resume="allow" if wandb_run_id else None,
            config=vars(args),
        )
        wandb_run_id = wandb_run.id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "wandb_run.json").write_text(json.dumps({"run_id": wandb_run_id}) + "\n")

    if is_main(rank):
        print(
            "\n".join(
                [
                    "Stage 1 training summary",
                    f"  manifest: {Path(args.manifest).expanduser().resolve()}",
                    f"  cache_root: {Path(args.cache_root).expanduser().resolve()}",
                    f"  out_dir: {out_dir}",
                    f"  train_images: {len(train_dataset):,} split={args.train_split}",
                    f"  val_images: {len(val_dataset):,} split={args.val_split}",
                    f"  world_size: {world_size}",
                    f"  batch_size_per_gpu: {args.batch_size}",
                    f"  global_batch_size: {global_batch_size}",
                    f"  max_steps: {args.max_steps:,}",
                    f"  start_step: {start_step:,}",
                    f"  steps_per_epoch: {steps_per_epoch:,}",
                    f"  approx_train_images_per_epoch: {train_images_per_epoch:,}",
                    f"  val_batches_per_eval: {val_batches:,}",
                    f"  approx_val_images_per_eval: {val_images_per_eval:,}",
                    f"  best_val_loss: {best_val_loss if best_val_loss != float('inf') else 'inf'}",
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
        for images, teacher in train_loader:
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
            teacher = move_teacher(teacher, device)
            with autocast_context(device, args.amp_dtype):
                student = model(images)
                loss, metrics = compute_loss(student, teacher, args)
            if not torch.isfinite(loss):
                if args.nonfinite_loss == "skip":
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    continue
                raise FloatingPointError(f"non-finite loss at step {step}: {loss.detach()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                clipped_grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(trainable_parameters(model)),
                    max_norm=args.max_grad_norm,
                )
            else:
                clipped_grad_norm = grad_norm(trainable_parameters(model))
            optimizer.step()

            reduced = reduce_metrics(metrics, world_size)
            reduced["train/lr"] = current_lr
            reduced["train/sec_per_step"] = time.time() - start_time
            reduced["train/backbone_trainable"] = float(should_train_backbone)
            reduced["train/projection_warmup_remaining"] = float(max(args.projection_warmup_steps - step, 0))
            reduced["train/grad_norm"] = float(clipped_grad_norm.detach().float().cpu())
            completed_steps = step - start_step + 1
            elapsed = time.time() - train_start_time
            avg_wall_sec_per_step = elapsed / max(completed_steps, 1)
            remaining_steps = max(args.max_steps - step - 1, 0)
            eta_seconds = remaining_steps * avg_wall_sec_per_step
            images_seen = (step + 1) * global_batch_size
            reduced["train/images_seen"] = float(images_seen)
            reduced["train/epoch"] = float(images_seen / max(len(train_dataset), 1))
            reduced["train/progress_pct"] = 100.0 * float(step + 1) / float(max(args.max_steps, 1))
            reduced["train/avg_wall_sec_per_step"] = avg_wall_sec_per_step
            reduced["train/eta_hours"] = eta_seconds / 3600.0
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
                            f"epoch {reduced['train/epoch']:.2f}",
                            f"images_seen {images_seen:,}",
                            f"eta {format_duration(eta_seconds)}",
                            f"loss {reduced['loss_stage1_total']:.6f}",
                            f"mse {reduced['loss_image_mse']:.6f}",
                            f"hr_mse {reduced['loss_high_res_mse']:.6f}",
                            f"lr {current_lr:.3e}",
                            f"grad {reduced['train/grad_norm']:.3f}",
                            f"wall_step {avg_wall_sec_per_step:.3f}s",
                        ]
                    ),
                    flush=True,
                )

            if step > 0 and step % args.eval_every == 0:
                val_metrics = evaluate(model, val_loader, device, world_size, args.val_max_batches, args)
                if is_main(rank):
                    for key, value in val_metrics.items():
                        if writer:
                            writer.add_scalar(key, value, step)
                    if wandb_run:
                        wandb_run.log(val_metrics, step=step)
                    val_loss = val_metrics.get("val/loss_stage1_total")
                    print(
                        " | ".join(
                            [
                                f"val step {step + 1:,}",
                                f"loss {val_metrics.get('val/loss_stage1_total', float('nan')):.6f}",
                                f"mse {val_metrics.get('val/loss_image_mse', float('nan')):.6f}",
                                f"hr_mse {val_metrics.get('val/loss_high_res_mse', float('nan')):.6f}",
                                f"best {best_val_loss if best_val_loss != float('inf') else float('nan'):.6f}",
                            ]
                        ),
                        flush=True,
                    )
                    if val_loss is not None and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        checkpoint_step = step + 1
                        save_checkpoint(
                            ckpt_dir / "best.pt",
                            model,
                            optimizer,
                            checkpoint_step,
                            wandb_run_id,
                            args,
                            best_val_loss,
                        )

            if is_main(rank) and step > 0 and step % args.save_every == 0:
                checkpoint_step = step + 1
                save_checkpoint(
                    ckpt_dir / f"step_{step:07d}.pt",
                    model,
                    optimizer,
                    checkpoint_step,
                    wandb_run_id,
                    args,
                    best_val_loss,
                )
                save_checkpoint(ckpt_dir / "last.pt", model, optimizer, checkpoint_step, wandb_run_id, args, best_val_loss)
            step += 1

    final_val_metrics = evaluate(model, val_loader, device, world_size, args.val_max_batches, args)
    if is_main(rank):
        for key, value in final_val_metrics.items():
            if writer:
                writer.add_scalar(key, value, step)
        if wandb_run:
            wandb_run.log(final_val_metrics, step=step)
        final_val_loss = final_val_metrics.get("val/loss_stage1_total")
        if final_val_loss is not None and final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, step, wandb_run_id, args, best_val_loss)
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, step, wandb_run_id, args, best_val_loss)
        if writer:
            writer.close()
        if wandb_run:
            wandb_run.finish()

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
