#!/usr/bin/env python3
"""Build deterministic image manifests for SAM2 Stage 1 distillation."""

from __future__ import annotations

import argparse
import hashlib
import math
from concurrent.futures import ProcessPoolExecutor
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


def limit_per_parent_dir(
    paths: list[Path],
    root: Path,
    source: str,
    seed: str,
    max_per_parent: int | None,
) -> list[Path]:
    if max_per_parent is None or max_per_parent <= 0:
        return paths

    groups: dict[str, list[Path]] = {}
    for path in paths:
        rel = path.relative_to(root)
        groups.setdefault(rel.parent.as_posix(), []).append(path)

    selected = []
    for parent, group_paths in groups.items():
        ranked = sorted(
            group_paths,
            key=lambda p: stable_digest(
                f"{seed}|{source}|parent={parent}|{p.relative_to(root).as_posix()}"
            ),
        )
        selected.extend(ranked[:max_per_parent])
    return sorted(selected)


def split_for(relative_path: str, seed: str, val_fraction: float) -> str:
    if val_fraction < 0 or val_fraction >= 1:
        raise ValueError("--val-fraction must be in [0, 1)")
    value = int(stable_digest(f"{seed}|split|{relative_path}")[:8], 16) / 0xFFFFFFFF
    return "val_sa1b" if value < val_fraction else "train"


def read_image_row(task: tuple[str, str, str, str, bool, str, float]) -> dict:
    path_s, image_root_s, source, seed, skip_file_sha256, rel, val_fraction = task
    path = Path(path_s)
    with Image.open(path) as image:
        width, height = image.size

    sha256 = "" if skip_file_sha256 else file_sha256(path)
    sample_key = stable_digest(f"{source}|{rel}")
    return {
        "sample_id": f"{source}_{sample_key[:20]}",
        "source": source,
        "image_path": str(path),
        "height": int(height),
        "width": int(width),
        "sha256": sha256,
        "split": split_for(rel, seed, val_fraction),
    }


def build_manifest(args: argparse.Namespace) -> pd.DataFrame:
    image_root = Path(args.image_root).expanduser().resolve()
    paths = scan_images(image_root)
    if not paths:
        raise SystemExit(f"No images found under {image_root}")

    subset = choose_subset(
        paths=paths,
        root=image_root,
        source=args.source,
        seed=args.seed,
        percent=args.sample_percent,
    )
    selected = limit_per_parent_dir(
        paths=subset,
        root=image_root,
        source=args.source,
        seed=args.seed,
        max_per_parent=args.max_images_per_parent_dir,
    )

    tasks = [
        (
            str(path),
            str(image_root),
            args.source,
            args.seed,
            args.skip_file_sha256,
            path.relative_to(image_root).as_posix(),
            args.val_fraction,
        )
        for path in selected
    ]
    if args.num_workers and args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            rows = list(
                tqdm(
                    executor.map(read_image_row, tasks, chunksize=args.worker_chunk_size),
                    total=len(tasks),
                    desc="manifest",
                )
            )
    else:
        rows = [read_image_row(task) for task in tqdm(tasks, desc="manifest")]

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="sa1b", help="Dataset source label.")
    parser.add_argument("--image-root", required=True, help="Directory to scan for images.")
    parser.add_argument(
        "--sample-percent",
        type=float,
        default=100.0,
        help="Deterministic percent of images to keep from --image-root.",
    )
    parser.add_argument(
        "--max-images-per-parent-dir",
        type=int,
        help="After sampling, keep at most this many deterministic images from each parent directory.",
    )
    parser.add_argument(
        "--seed",
        default="sam2_stage1_sa1b_v1",
        help="Stable sampling seed.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Deterministic validation fraction after sampling. Use 0.1 for 90/10 train/val.",
    )
    parser.add_argument("--out", required=True, help="Output .parquet or .csv path.")
    parser.add_argument(
        "--skip-file-sha256",
        action="store_true",
        help="Leave sha256 blank for fast local smoke tests only.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel workers for reading selected image metadata. 0 or 1 runs single-process.",
    )
    parser.add_argument(
        "--worker-chunk-size",
        type=int,
        default=64,
        help="Chunk size for parallel metadata reads.",
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
