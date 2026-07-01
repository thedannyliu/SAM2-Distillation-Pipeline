#!/usr/bin/env python3
"""Benchmark SAM2 decoder masks using TinyViT Stage 1 student image features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam2_distill.models.tinyvit_adapter import TinyViTSAM2Adapter


def load_boxes(path: Path, split: str, limit: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != split:
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"No {split} boxes found in {path}")
    return rows


def load_predictor(config: str, checkpoint: Path, device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(config, str(checkpoint), device=device, mode="eval")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return SAM2ImagePredictor(model)


def load_student(checkpoint: Path, tinyvit_checkpoint: Path, device: str) -> TinyViTSAM2Adapter:
    student = TinyViTSAM2Adapter(checkpoint_path=str(tinyvit_checkpoint)).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    student.load_state_dict(ckpt["model_state"], strict=True)
    student.eval()
    return student


def set_student_features(predictor, student: TinyViTSAM2Adapter, image: Image.Image, device: str) -> None:
    predictor.reset_predictor()
    predictor._orig_hw = [(image.height, image.width)]
    input_image = predictor._transforms(image)
    input_image = input_image[None, ...].to(device)
    with torch.inference_mode():
        features = student(input_image)
    predictor._features = {
        "image_embed": features["image_embed"],
        "high_res_feats": [features["high_res_s0"], features["high_res_s1"]],
    }
    predictor._is_image_set = True
    predictor._is_batch = False


def overlay_mask(image: Image.Image, mask: np.ndarray, out_path: Path) -> None:
    base = image.convert("RGBA")
    mask_bool = mask.astype(bool)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    alpha = np.zeros((base.height, base.width), dtype=np.uint8)
    alpha[mask_bool] = 110
    red = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    red[..., 0] = 255
    red[..., 3] = alpha
    overlay = Image.fromarray(red, mode="RGBA")
    blended = Image.alpha_composite(base, overlay)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blended.convert("RGB").save(out_path)


def sync_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--tinyvit-checkpoint", required=True)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--save-overlays", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_boxes(Path(args.boxes), args.split, args.limit)
    predictor = load_predictor(args.sam2_config, Path(args.sam2_checkpoint), device)
    student = load_student(Path(args.student_checkpoint), Path(args.tinyvit_checkpoint), device)

    latencies = []
    for idx, row in enumerate(tqdm(rows, desc="benchmark")):
        image = Image.open(row["image_path"]).convert("RGB")
        box = np.asarray(row["bbox_xyxy"], dtype=np.float32)

        sync_if_needed(device)
        start = time.perf_counter()
        set_student_features(predictor, student, image, device)
        masks, scores, _ = predictor.predict(box=box, multimask_output=False)
        sync_if_needed(device)
        elapsed = time.perf_counter() - start

        mask = masks[0] if masks.ndim == 3 else masks
        score = float(np.asarray(scores).reshape(-1)[0])
        latencies.append(
            {
                "sample_id": row["sample_id"],
                "seconds": elapsed,
                "score": score,
                "image_path": row["image_path"],
            }
        )
        if idx < args.save_overlays:
            overlay_mask(image, mask, out_dir / "overlays" / f"{idx:04d}_{row['sample_id']}.jpg")

    seconds = [row["seconds"] for row in latencies]
    summary = {
        "num_images": len(latencies),
        "mean_seconds": float(np.mean(seconds)),
        "median_seconds": float(np.median(seconds)),
        "p95_seconds": float(np.percentile(seconds, 95)),
        "mean_fps": float(1.0 / np.mean(seconds)),
    }
    (out_dir / "latencies.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in latencies),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
