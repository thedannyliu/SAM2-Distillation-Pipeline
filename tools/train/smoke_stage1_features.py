#!/usr/bin/env python3
"""Run a bounded Stage 1 feature-training smoke test on real images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam2_distill.models.tinyvit_adapter import SAM2_STAGE1_TARGETS, TinyViTSAM2Adapter


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class JsonlImageDataset(Dataset):
    def __init__(self, manifest: Path, split: str, image_size: int, max_items: int) -> None:
        self.image_size = image_size
        rows = []
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("split") == split:
                    rows.append(row)
        if not rows:
            raise ValueError(f"No rows with split={split!r} in {manifest}")
        self.rows = rows[:max_items]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        row = self.rows[idx]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor, row["sample_id"]


def synthetic_teacher_targets(batch_size: int, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return {
        target.name: torch.randn(
            batch_size,
            target.channels,
            target.size,
            target.size,
            generator=generator,
            device=device,
        )
        for target in SAM2_STAGE1_TARGETS
    }


def mse_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    losses = {name: torch.mean((outputs[name].float() - target.float()) ** 2) for name, target in targets.items()}
    total = sum(losses.values())
    return total, {f"loss/{name}": float(value.detach().cpu()) for name, value in losses.items()}


def json_safe_args(args: argparse.Namespace) -> dict:
    safe = {}
    for key, value in vars(args).items():
        safe[key] = str(value) if isinstance(value, Path) else value
    return safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-name", default="tiny_vit_21m_512.dist_in22k_ft_in1k")
    parser.add_argument("--tinyvit-checkpoint", default=None)
    parser.add_argument("--adapter-mode", choices=("projection", "residual_dwconv"), default="projection")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-items", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=250107256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_items < 1 or args.max_items > 500:
        raise SystemExit("--max-items must be in [1, 500] for smoke tests")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = JsonlImageDataset(args.manifest, args.split, args.image_size, args.max_items)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = TinyViTSAM2Adapter(
        model_name=args.model_name,
        checkpoint_path=args.tinyvit_checkpoint,
        input_size=args.image_size,
        adapter_mode=args.adapter_mode,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    metrics_path = args.out_dir / "metrics.jsonl"
    iterator = iter(loader)
    last_loss = None
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(args.steps):
            try:
                images, sample_ids = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, sample_ids = next(iterator)

            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            targets = synthetic_teacher_targets(images.shape[0], device, args.seed + step)
            loss, parts = mse_loss(outputs, targets)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            row = {
                "step": step,
                "sample_ids": list(sample_ids),
                "loss/total": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "device": str(device),
                "image_size": args.image_size,
            }
            row.update(parts)
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            last_loss = row["loss/total"]

    checkpoint_path = args.out_dir / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": json_safe_args(args),
            "last_loss": last_loss,
            "targets": [target.__dict__ for target in SAM2_STAGE1_TARGETS],
        },
        checkpoint_path,
    )
    summary = {
        "status": "pass",
        "steps": args.steps,
        "items": len(dataset),
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
        "last_loss": last_loss,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
