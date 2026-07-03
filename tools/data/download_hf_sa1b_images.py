#!/usr/bin/env python3
"""Stream a bounded SA-1B image subset from Hugging Face to local JPEG files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def stable_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_for(sample_id: str, seed: str, val_fraction: float) -> str:
    value = int(stable_digest(f"{seed}|split|{sample_id}")[:8], 16) / 0xFFFFFFFF
    return "val_sa1b" if value < val_fraction else "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="hdtech/SA-1B")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=10000)
    parser.add_argument("--max-gb", type=float, default=0.0, help="Stop after local JPEG bytes exceed this limit; 0 disables.")
    parser.add_argument("--seed", default="sam2_stage1_hf_sa1b_v1")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--image-format", choices=("jpeg", "png"), default="jpeg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def save_image(image: Image.Image, path: Path, args: argparse.Namespace) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    if args.image_format == "jpeg":
        image.save(path, quality=args.jpeg_quality, optimize=True)
    else:
        image.save(path)
    return path.stat().st_size


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install Hugging Face datasets first: python -m pip install --user datasets") from exc

    if args.max_images <= 0:
        raise SystemExit("--max-images must be positive")
    if args.val_fraction < 0 or args.val_fraction >= 1:
        raise SystemExit("--val-fraction must be in [0, 1)")

    image_dir = args.out_root / "images"
    args.out_root.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total_bytes = 0
    if args.resume and args.manifest.exists():
        old = pd.read_parquet(args.manifest) if args.manifest.suffix == ".parquet" else pd.read_csv(args.manifest)
        rows = old.to_dict("records")
        total_bytes = sum(Path(row["image_path"]).stat().st_size for row in rows if Path(row["image_path"]).exists())

    dataset = load_dataset(
        args.repo_id,
        split=args.split,
        revision=args.revision,
        streaming=True,
    )
    if args.shuffle_buffer_size > 0:
        dataset = dataset.shuffle(seed=int(stable_digest(args.seed)[:8], 16), buffer_size=args.shuffle_buffer_size)

    max_bytes = int(args.max_gb * 1024**3) if args.max_gb > 0 else None
    start_count = len(rows)
    progress = tqdm(total=args.max_images, initial=start_count, desc="hf-sa1b")
    for stream_idx, sample in enumerate(dataset):
        if len(rows) >= args.max_images:
            break
        image = sample.get("image")
        if image is None:
            continue
        sample_id = f"hf_sa1b_{stable_digest(f'{args.repo_id}|{args.revision}|{args.split}|{stream_idx}')[:20]}"
        rel_name = f"{sample_id}.{ 'jpg' if args.image_format == 'jpeg' else 'png' }"
        path = image_dir / rel_name
        if args.resume and path.exists() and any(row["sample_id"] == sample_id for row in rows):
            continue
        bytes_written = save_image(image, path, args)
        total_bytes += bytes_written
        with Image.open(path) as saved_image:
            width, height = saved_image.size
        rows.append(
            {
                "sample_id": sample_id,
                "source": args.repo_id,
                "image_path": str(path),
                "height": int(height),
                "width": int(width),
                "sha256": "",
                "split": split_for(sample_id, args.seed, args.val_fraction),
                "hf_repo_id": args.repo_id,
                "hf_revision": args.revision,
                "hf_split": args.split,
                "hf_stream_index": int(stream_idx),
            }
        )
        progress.update(1)
        if max_bytes is not None and total_bytes >= max_bytes:
            break
    progress.close()

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    if args.manifest.suffix == ".parquet":
        df.to_parquet(args.manifest, index=False)
    elif args.manifest.suffix == ".csv":
        df.to_csv(args.manifest, index=False)
    else:
        raise SystemExit("--manifest must end in .parquet or .csv")

    summary = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "split": args.split,
        "out_root": str(args.out_root),
        "manifest": str(args.manifest),
        "images": int(len(df)),
        "local_image_bytes": int(total_bytes),
        "local_image_gb": total_bytes / 1024**3,
        "seed": args.seed,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "val_fraction": args.val_fraction,
        "splits": df["split"].value_counts().to_dict() if not df.empty else {},
    }
    (args.out_root / "download_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
