#!/usr/bin/env python3
"""Run a bounded VOS mask-training smoke test on real video frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TinyMaskNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = images.shape
        logits = self.net(images.reshape(b * t, c, h, w))
        return logits.reshape(b, t, 1, h, w)


class VOSClipDataset(Dataset):
    def __init__(self, manifest: Path, image_size: int, clip_frames: int, max_clips: int) -> None:
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_video: dict[str, list[dict]] = {}
        for row in rows:
            by_video.setdefault(row["video_id"], []).append(row)
        clips = []
        for video_rows in by_video.values():
            video_rows = sorted(video_rows, key=lambda row: row["frame_id"])
            if len(video_rows) < clip_frames:
                continue
            for start in range(0, len(video_rows) - clip_frames + 1, clip_frames):
                clips.append(video_rows[start : start + clip_frames])
                if len(clips) >= max_clips:
                    break
            if len(clips) >= max_clips:
                break
        if not clips:
            raise ValueError(f"No {clip_frames}-frame clips found in {manifest}")
        self.clips = clips
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        images = []
        masks = []
        video_id = self.clips[idx][0]["video_id"]
        for row in self.clips[idx]:
            with Image.open(row["image_path"]) as image:
                image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
            with Image.open(row["mask_path"]) as mask:
                mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            image_arr = np.asarray(image, dtype=np.float32) / 255.0
            mask_arr = (np.asarray(mask) > 0).astype(np.float32)
            images.append(torch.from_numpy(image_arr).permute(2, 0, 1))
            masks.append(torch.from_numpy(mask_arr)[None, ...])
        return torch.stack(images), torch.stack(masks), video_id


def json_safe_args(args: argparse.Namespace) -> dict:
    safe = {}
    for key, value in vars(args).items():
        safe[key] = str(value) if isinstance(value, Path) else value
    return safe


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    numerator = 2 * (probs * targets).sum(dim=(-1, -2))
    denominator = probs.sum(dim=(-1, -2)) + targets.sum(dim=(-1, -2)) + 1e-6
    return 1 - (numerator + 1e-6) / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--clip-frames", type=int, default=4)
    parser.add_argument("--max-clips", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=250107256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phase-name", default="video_mask_train_smoke")
    parser.add_argument("--freeze-image-encoder", action="store_true")
    parser.add_argument("--teacher-disabled", action="store_true")
    parser.add_argument("--distill-disabled", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = VOSClipDataset(args.manifest, args.image_size, args.clip_frames, args.max_clips)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = TinyMaskNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    checkpoint_path = args.out_dir / "last.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1

    metrics_path = args.out_dir / "metrics.jsonl"
    iterator = iter(loader)
    last_loss = None
    with metrics_path.open("a" if args.resume else "w", encoding="utf-8") as metrics:
        for step in range(start_step, start_step + args.steps):
            try:
                images, masks, video_ids = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, masks, video_ids = next(iterator)
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss_bce = F.binary_cross_entropy_with_logits(logits, masks)
            loss_dice = dice_loss(logits, masks).mean()
            loss = loss_bce + loss_dice
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            row = {
                "step": step,
                "phase": args.phase_name,
                "video_ids": list(video_ids),
                "loss/total": float(loss.detach().cpu()),
                "loss/bce": float(loss_bce.detach().cpu()),
                "loss/dice": float(loss_dice.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "device": str(device),
                "clip_frames": args.clip_frames,
            }
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            last_loss = row["loss/total"]

    torch.save(
        {
            "step": start_step + args.steps - 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": json_safe_args(args),
            "last_loss": last_loss,
        },
        checkpoint_path,
    )
    summary = {
        "status": "pass",
        "phase": args.phase_name,
        "clips": len(dataset),
        "steps": args.steps,
        "start_step": start_step,
        "device": str(device),
        "freeze_image_encoder": bool(args.freeze_image_encoder),
        "teacher_disabled": bool(args.teacher_disabled),
        "distill_disabled": bool(args.distill_disabled),
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
        "last_loss": last_loss,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
