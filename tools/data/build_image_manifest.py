#!/usr/bin/env python3
"""Build deterministic image manifests for SAM2 Stage 1 distillation."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def stable_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def choose_subset(paths: list[Path], root: Path, source: str, seed: str, percent: float) -> list[Path]:
    if percent <= 0 or percent > 100:
        raise ValueError("--sample-percent must be in (0, 100]")
    count = math.ceil(len(paths) * percent / 100.0)
    ranked = sorted(
        paths,
        key=lambda p: stable_digest(f"{seed}|{source}|{p.relative_to(root).as_posix()}"),
    )
    return ranked[:count]


def split_for(relative_path: str, seed: str) -> str:
    value = int(stable_digest(f"{seed}|split|{relative_path}")[:8], 16) / 0xFFFFFFFF
    return "val_sa1b" if value < 0.01 else "train"


def build_manifest(args: argparse.Namespace) -> pd.DataFrame:
    image_root = Path(args.image_root).expanduser().resolve()
    paths = scan_images(image_root)
    if not paths:
        raise SystemExit(f"No images found under {image_root}")

    selected = choose_subset(
        paths=paths,
        root=image_root,
        source=args.source,
        seed=args.seed,
        percent=args.sample_percent,
    )

    rows = []
    for path in tqdm(selected, desc="manifest"):
        rel = path.relative_to(image_root).as_posix()
        with Image.open(path) as image:
            width, height = image.size

        sha256 = "" if args.skip_file_sha256 else file_sha256(path)
        sample_key = stable_digest(f"{args.source}|{rel}")
        rows.append(
            {
                "sample_id": f"{args.source}_{sample_key[:20]}",
                "source": args.source,
                "image_path": str(path),
                "height": int(height),
                "width": int(width),
                "sha256": sha256,
                "split": split_for(rel, args.seed),
            }
        )

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="sa1b", help="Dataset source label.")
    parser.add_argument("--image-root", required=True, help="Directory to scan for images.")
    parser.add_argument(
        "--sample-percent",
        type=float,
        default=1.0,
        help="Deterministic percent of images to keep. Use 1 for fixed SA-1B 1%%.",
    )
    parser.add_argument(
        "--seed",
        default="sam2_stage1_sa1b_1pct_v1",
        help="Stable sampling seed.",
    )
    parser.add_argument("--out", required=True, help="Output .parquet or .csv path.")
    parser.add_argument(
        "--skip-file-sha256",
        action="store_true",
        help="Leave sha256 blank for fast local smoke tests only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df = build_manifest(args)

    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    elif out.suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        raise SystemExit("--out must end in .parquet or .csv")

    print(f"wrote={out}")
    print(f"rows={len(df)}")
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
