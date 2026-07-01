#!/usr/bin/env python3
"""Cache SAM2 teacher image features for Stage 1 encoder distillation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from PIL import Image
from tqdm import tqdm


def read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("manifest must be .parquet or .csv")


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def checkpoint_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_teacher(config: str, checkpoint: Path, device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(config, str(checkpoint), device=device, mode="eval")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return SAM2ImagePredictor(model)


def read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def create_array(group, name: str, shape: tuple[int, ...], chunks: tuple[int, ...]):
    return group.create_dataset(
        name,
        shape=shape,
        chunks=chunks,
        dtype="float16",
        overwrite=True,
    )


def write_shard(
    rows: pd.DataFrame,
    shard_id: int,
    shard_dir: Path,
    predictor,
    args: argparse.Namespace,
    checkpoint_hash: str,
) -> None:
    done_path = shard_dir / ".done"
    if done_path.exists() and not args.overwrite:
        print(f"skip_completed={shard_dir}")
        return

    shard_dir.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(shard_dir), mode="w")
    n = len(rows)

    image_embed = create_array(root, "image_embed", (n, 256, 64, 64), (1, 256, 64, 64))
    high_res_s0 = create_array(root, "high_res_s0", (n, 32, 256, 256), (1, 32, 256, 256))
    high_res_s1 = create_array(root, "high_res_s1", (n, 64, 128, 128), (1, 64, 128, 128))

    root.attrs.update(
        {
            "schema_version": "sam2_stage1_teacher_cache_v1",
            "teacher_config": args.config,
            "teacher_checkpoint": str(args.checkpoint),
            "teacher_checkpoint_sha256": checkpoint_hash,
            "dtype": "float16",
            "image_size": 1024,
            "git_commit": git_commit(),
            "shard_id": shard_id,
            "num_rows": n,
        }
    )

    index_rows = []
    offset = 0
    with torch.inference_mode():
        for start in tqdm(range(0, n, args.batch_size), desc=f"shard {shard_id}"):
            batch = rows.iloc[start : start + args.batch_size]
            images = [read_rgb(path) for path in batch["image_path"].tolist()]
            predictor.set_image_batch(images)
            features = predictor._features

            batch_image_embed = features["image_embed"].detach().cpu().to(torch.float16).numpy()
            high_res = features["high_res_feats"]
            batch_s0 = high_res[0].detach().cpu().to(torch.float16).numpy()
            batch_s1 = high_res[1].detach().cpu().to(torch.float16).numpy()

            end = offset + len(batch)
            image_embed[offset:end] = batch_image_embed
            high_res_s0[offset:end] = batch_s0
            high_res_s1[offset:end] = batch_s1

            for local_i, (_, row) in enumerate(batch.iterrows(), start=offset):
                index_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "source": row["source"],
                        "image_path": row["image_path"],
                        "split": row["split"],
                        "shard_id": shard_id,
                        "row_in_shard": local_i,
                    }
                )
            offset = end

    pd.DataFrame(index_rows).to_parquet(shard_dir / "index.parquet", index=False)
    done_path.write_text(json.dumps({"rows": n}, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True, help="SAM2 config path, e.g. configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8, help="Reserved for company job wrappers.")
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint = Path(args.checkpoint).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = read_manifest(Path(args.manifest).expanduser().resolve())
    required = {"sample_id", "source", "image_path", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"manifest missing columns: {sorted(missing)}")
    if args.limit:
        df = df.head(args.limit).copy()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available. Use --device cpu only for tiny smoke tests.")

    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    predictor = load_teacher(args.config, args.checkpoint, args.device)

    num_total_shards = math.ceil(len(df) / args.shard_size)
    end_shard = num_total_shards
    if args.num_shards is not None:
        end_shard = min(end_shard, args.start_shard + args.num_shards)

    for shard_id in range(args.start_shard, end_shard):
        start = shard_id * args.shard_size
        end = min(len(df), start + args.shard_size)
        shard_rows = df.iloc[start:end].reset_index(drop=True)
        if shard_rows.empty:
            continue
        write_shard(
            rows=shard_rows,
            shard_id=shard_id,
            shard_dir=out / f"shard-{shard_id:06d}.zarr",
            predictor=predictor,
            args=args,
            checkpoint_hash=checkpoint_hash,
        )

    print(f"cache_root={out}")


if __name__ == "__main__":
    main()
