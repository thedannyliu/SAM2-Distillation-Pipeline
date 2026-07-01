#!/usr/bin/env python3
"""Prepare a fixed COCO Stage 1 pilot subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def stable_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_instances(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def best_box_by_image(instances: dict) -> dict[int, dict]:
    best: dict[int, dict] = {}
    for ann in instances.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        bbox = ann.get("bbox", [])
        if len(bbox) != 4 or bbox[2] <= 1 or bbox[3] <= 1:
            continue
        image_id = int(ann["image_id"])
        current = best.get(image_id)
        if current is None or float(ann.get("area", bbox[2] * bbox[3])) > float(current.get("area", 0)):
            best[image_id] = ann
    return best


def select_images(instances: dict, split: str, count: int, seed: str) -> list[dict]:
    boxes = best_box_by_image(instances)
    candidates = [img for img in instances["images"] if int(img["id"]) in boxes]
    ranked = sorted(
        candidates,
        key=lambda img: stable_digest(f"{seed}|{split}|{img['file_name']}|{img['id']}"),
    )
    if len(ranked) < count:
        raise SystemExit(f"{split} has only {len(ranked)} images with usable boxes, need {count}")
    return ranked[:count]


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def copy_split(
    split: str,
    count: int,
    coco_root: Path,
    out_root: Path,
    seed: str,
    skip_sha256: bool,
) -> tuple[list[dict], list[dict], dict]:
    instances_path = coco_root / "annotations" / f"instances_{split}2017.json"
    image_root = coco_root / f"{split}2017"
    instances = load_instances(instances_path)
    boxes = best_box_by_image(instances)
    selected = select_images(instances, split, count, seed)

    out_image_root = out_root / "images" / split
    out_image_root.mkdir(parents=True, exist_ok=True)

    selected_ids = {int(img["id"]) for img in selected}
    subset_annotations = [
        ann for ann in instances["annotations"] if int(ann["image_id"]) in selected_ids
    ]
    subset = {
        "info": instances.get("info", {}),
        "licenses": instances.get("licenses", []),
        "images": selected,
        "annotations": subset_annotations,
        "categories": instances.get("categories", []),
    }

    manifest_rows = []
    box_rows = []
    for image in tqdm(selected, desc=f"copy {split}"):
        src = image_root / image["file_name"]
        dst = out_image_root / image["file_name"]
        if not dst.exists():
            shutil.copy2(src, dst)

        with Image.open(dst) as pil_image:
            width, height = pil_image.size

        sha256 = "" if skip_sha256 else file_sha256(dst)
        sample_id = f"coco_{split}_{int(image['id'])}"
        ann = boxes[int(image["id"])]
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "source": f"coco_{split}2017",
                "image_path": str(dst),
                "height": int(height),
                "width": int(width),
                "sha256": sha256,
                "split": "train" if split == "train" else "val",
                "coco_image_id": int(image["id"]),
                "file_name": image["file_name"],
            }
        )
        box_rows.append(
            {
                "sample_id": sample_id,
                "split": "train" if split == "train" else "val",
                "image_path": str(dst),
                "height": int(height),
                "width": int(width),
                "coco_image_id": int(image["id"]),
                "annotation_id": int(ann["id"]),
                "category_id": int(ann["category_id"]),
                "bbox_xywh": [float(v) for v in ann["bbox"]],
                "bbox_xyxy": xywh_to_xyxy(ann["bbox"]),
            }
        )

    return manifest_rows, box_rows, subset


def remove_archives(root: Path) -> None:
    for pattern in ("*.zip", "*.tar", "*.tar.gz", "*.tgz"):
        for path in root.glob(pattern):
            path.unlink()
            print(f"removed={path}")


def remove_extracted_images(root: Path) -> None:
    for name in ("train2017", "val2017"):
        path = root / name
        if path.exists():
            shutil.rmtree(path)
            print(f"removed={path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", required=True, help="Root containing train2017/ val2017/ annotations/.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seed", default="sam2_stage1_coco_pilot_v1")
    parser.add_argument("--skip-file-sha256", action="store_true")
    parser.add_argument("--remove-archives", action="store_true", help="Delete zip/tar archives under coco-root after subset copy.")
    parser.add_argument(
        "--remove-extracted-images",
        action="store_true",
        help="Delete coco-root/train2017 and coco-root/val2017 after the pilot subset is copied.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coco_root = Path(args.coco_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    box_rows: list[dict] = []
    subsets = {}
    for split, count in (("train", args.train_count), ("val", args.val_count)):
        rows, boxes, subset = copy_split(
            split=split,
            count=count,
            coco_root=coco_root,
            out_root=out_root,
            seed=args.seed,
            skip_sha256=args.skip_file_sha256,
        )
        manifest_rows.extend(rows)
        box_rows.extend(boxes)
        subsets[split] = subset

    manifest_path = out_root / "manifests" / f"coco_pilot_{args.train_count}train_{args.val_count}val.parquet"
    boxes_path = out_root / "manifests" / "coco_pilot_boxes.jsonl"
    ann_dir = out_root / "annotations"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(manifest_rows).sort_values(["split", "sample_id"]).to_parquet(
        manifest_path, index=False
    )
    with boxes_path.open("w", encoding="utf-8") as f:
        for row in box_rows:
            f.write(json.dumps(row) + "\n")
    for split, subset in subsets.items():
        with (ann_dir / f"instances_{split}2017_pilot.json").open("w", encoding="utf-8") as f:
            json.dump(subset, f)

    if args.remove_archives:
        remove_archives(coco_root)
    if args.remove_extracted_images:
        remove_extracted_images(coco_root)

    print(f"manifest={manifest_path}")
    print(f"boxes={boxes_path}")
    print(f"rows={len(manifest_rows)}")


if __name__ == "__main__":
    main()
