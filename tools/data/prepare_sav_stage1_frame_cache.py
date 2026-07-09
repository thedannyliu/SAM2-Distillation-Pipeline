#!/usr/bin/env python3
"""Prepare storage-aware SA-V frame cache and manifests for Stage 1 distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm


def stable_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reuse-train-manifest", type=Path)
    parser.add_argument("--train-frames-per-video", type=int, default=16)
    parser.add_argument("--val-frames-per-video", type=int, default=8)
    parser.add_argument("--test-frames-per-video", type=int, default=0)
    parser.add_argument("--max-train-videos", type=int, default=0)
    parser.add_argument("--max-val-videos", type=int, default=0)
    parser.add_argument("--max-test-videos", type=int, default=0)
    parser.add_argument("--ann-every", type=int, default=4)
    parser.add_argument("--seed", default="sam2_stage1_sav_vbal16_6fps_v1")
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--use-auto", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    return parser.parse_args()


def detect_video_root(root: Path) -> Path:
    candidates = [root / "videos", root / "train" / "videos", root]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.rglob("*.mp4")):
            return candidate
    raise FileNotFoundError(f"No mp4 files under {root}")


def detect_ann_root(root: Path) -> Path | None:
    candidates = [root / "annotations", root / "train" / "annotations", root]
    for candidate in candidates:
        if candidate.is_dir() and (any(candidate.glob("*_manual.json")) or any(candidate.glob("*_auto.json"))):
            return candidate
    return None


def choose_annotation(ann_root: Path | None, video_id: str, use_auto: bool) -> Path | None:
    if ann_root is None:
        return None
    manual = ann_root / f"{video_id}_manual.json"
    auto = ann_root / f"{video_id}_auto.json"
    if manual.exists():
        return manual
    if use_auto and auto.exists():
        return auto
    return None


def annotation_length(path: Path | None) -> int | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    masklets = payload.get("masklet")
    if isinstance(masklets, list) and masklets:
        return len(masklets)
    return None


def video_frame_count(path: Path) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Frame extraction requires cv2/opencv-python.") from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return count


def select_6fps_indices(total_6fps: int, count: int, seed: str, video_id: str) -> list[int]:
    if count <= 0 or total_6fps <= 0:
        return []
    if count >= total_6fps:
        return list(range(total_6fps))
    selected = []
    for bucket in range(count):
        start = int(bucket * total_6fps / count)
        end = int((bucket + 1) * total_6fps / count)
        end = max(end, start + 1)
        span = end - start
        digest = stable_digest(f"{seed}|{video_id}|bucket={bucket}")
        selected.append(start + (int(digest[:8], 16) % span))
    return sorted(set(selected))


def listed_video_ids(root: Path) -> set[str] | None:
    list_files = sorted(root.glob("*.txt"))
    if not list_files:
        return None
    ids = set()
    for line in list_files[0].read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value:
            ids.add(Path(value).stem)
    return ids


def discover_prepared_split(
    split_name: str,
    root: Path,
    frames_per_video: int,
    max_videos: int,
    seed: str,
    ann_every: int,
) -> list[dict[str, Any]]:
    image_root = root / "JPEGImages_24fps"
    if not image_root.is_dir():
        return []
    annotation_root = root / "Annotations_6fps"
    allowed_ids = listed_video_ids(root)
    tasks = []
    for video_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        video_id = video_dir.name
        if allowed_ids is not None and video_id not in allowed_ids:
            continue
        frames = sorted(
            path
            for path in video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.stem.isdigit()
        )
        aligned = [path for path in frames if int(path.stem) % ann_every == 0]
        if not aligned:
            continue
        selected_positions = select_6fps_indices(len(aligned), frames_per_video, seed, video_id)
        selected_frames = [str(aligned[position]) for position in selected_positions]
        if selected_frames:
            tasks.append(
                {
                    "kind": "prepared",
                    "split": split_name,
                    "video_id": video_id,
                    "video_path": str(video_dir),
                    "annotation_path": str(annotation_root / video_id),
                    "frame_paths": selected_frames,
                }
            )
        if max_videos > 0 and len(tasks) >= max_videos:
            break
    return tasks


def discover_split(
    split_name: str,
    root: Path,
    frames_per_video: int,
    max_videos: int,
    seed: str,
    use_auto: bool,
    ann_every: int,
) -> list[dict[str, Any]]:
    if frames_per_video <= 0:
        return []
    prepared_tasks = discover_prepared_split(split_name, root, frames_per_video, max_videos, seed, ann_every)
    if prepared_tasks:
        return prepared_tasks
    video_root = detect_video_root(root)
    ann_root = detect_ann_root(root)
    tasks = []
    for video in sorted(video_root.rglob("*.mp4")):
        ann_path = choose_annotation(ann_root, video.stem, use_auto)
        total_6fps = annotation_length(ann_path)
        if total_6fps is None:
            frames = video_frame_count(video)
            total_6fps = max(frames // 4, 1)
        indices_6fps = select_6fps_indices(total_6fps, frames_per_video, seed, video.stem)
        if indices_6fps:
            tasks.append(
                {
                    "kind": "raw",
                    "split": split_name,
                    "video_id": video.stem,
                    "video_path": str(video),
                    "annotation_path": str(ann_path) if ann_path else "",
                    "indices_6fps": indices_6fps,
                }
            )
        if max_videos > 0 and len(tasks) >= max_videos:
            break
    return tasks


def prepared_frame_task(task: dict[str, Any], ann_every: int) -> list[dict]:
    rows = []
    for frame_path in task["frame_paths"]:
        path = Path(frame_path)
        idx_24fps = int(path.stem)
        with Image.open(path) as image:
            width, height = image.size
        rows.append(
            {
                "sample_id": f"sav_{task['split']}_{task['video_id']}_{idx_24fps:05d}",
                "source": "sa_v",
                "video_id": task["video_id"],
                "frame_idx_24fps": idx_24fps,
                "frame_idx_6fps": idx_24fps // ann_every,
                "image_path": str(path),
                "height": int(height),
                "width": int(width),
                "split": task["split"],
                "video_path": task["video_path"],
                "annotation_path": task["annotation_path"],
            }
        )
    return rows


def process_task(
    task: dict[str, Any], out_root: str, ann_every: int, jpeg_quality: int, skip_existing: bool
) -> list[dict]:
    if task["kind"] == "prepared":
        return prepared_frame_task(task, ann_every)
    return extract_video_task(task, out_root, ann_every, jpeg_quality, skip_existing)


def extract_video_task(task: dict[str, Any], out_root: str, ann_every: int, jpeg_quality: int, skip_existing: bool) -> list[dict]:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Frame extraction requires cv2/opencv-python.") from exc

    out_dir = Path(out_root) / "JPEGImages" / task["video_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(task["video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {task['video_path']}")

    rows = []
    for idx_6fps in task["indices_6fps"]:
        idx_24fps = int(idx_6fps) * ann_every
        out_path = out_dir / f"{idx_24fps:05d}.jpg"
        if not (skip_existing and out_path.exists()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_24fps)
            ok, frame = cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(out_path, quality=jpeg_quality)
        with Image.open(out_path) as image:
            width, height = image.size
        rows.append(
            {
                "sample_id": f"sav_{task['split']}_{task['video_id']}_{idx_24fps:05d}",
                "source": "sa_v",
                "video_id": task["video_id"],
                "frame_idx_24fps": idx_24fps,
                "frame_idx_6fps": int(idx_6fps),
                "image_path": str(out_path),
                "height": int(height),
                "width": int(width),
                "split": task["split"],
                "video_path": task["video_path"],
                "annotation_path": task["annotation_path"],
            }
        )
    cap.release()
    return rows


def write_manifest(rows: list[dict], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["split", "video_id", "frame_idx_24fps"]).reset_index(drop=True)
    if manifest.suffix == ".parquet":
        temporary = manifest.with_suffix(".tmp.parquet")
        df.to_parquet(temporary, index=False)
    elif manifest.suffix == ".csv":
        temporary = manifest.with_suffix(".tmp.csv")
        df.to_csv(temporary, index=False)
    else:
        raise SystemExit("--manifest must end in .parquet or .csv")
    temporary.replace(manifest)


def read_reused_train_rows(path: Path) -> list[dict]:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    train = df[df["split"] == "train"].copy()
    if train.empty:
        raise RuntimeError(f"No train rows in reusable manifest: {path}")
    return train.to_dict(orient="records")


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    reused_train_rows = []
    tasks = []
    if args.reuse_train_manifest:
        reused_train_rows = read_reused_train_rows(args.reuse_train_manifest)
    else:
        tasks.extend(
            discover_split(
                "train",
                args.train_root,
                args.train_frames_per_video,
                args.max_train_videos,
                args.seed,
                args.use_auto,
                args.ann_every,
            )
        )
    if args.val_root:
        tasks.extend(
            discover_split(
                "val_sav",
                args.val_root,
                args.val_frames_per_video,
                args.max_val_videos,
                args.seed,
                args.use_auto,
                args.ann_every,
            )
        )
    if args.test_root:
        tasks.extend(
            discover_split(
                "test_sav",
                args.test_root,
                args.test_frames_per_video,
                args.max_test_videos,
                args.seed,
                args.use_auto,
                args.ann_every,
            )
        )
    if not tasks and not reused_train_rows:
        raise RuntimeError("No videos selected for frame extraction or reuse.")

    rows = []
    worker_count = max(1, args.num_workers)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                process_task,
                task,
                str(args.out_root),
                args.ann_every,
                args.jpeg_quality,
                args.skip_existing,
            )
            for task in tasks
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="extract videos"):
            rows.extend(future.result())

    rows.extend(reused_train_rows)
    if not rows:
        raise RuntimeError("No frames were extracted.")
    held_out_ids = {
        row["video_id"] for row in rows if row["split"] in {"val_sav", "test_sav"}
    }
    rows = [
        row for row in rows if row["split"] != "train" or row["video_id"] not in held_out_ids
    ]
    write_manifest(rows, args.manifest)
    counts = pd.DataFrame(rows)["split"].value_counts().to_dict()
    summary = {
        "status": "pass",
        "out_root": str(args.out_root),
        "manifest": str(args.manifest),
        "videos": len({(row["split"], row["video_id"]) for row in rows}),
        "frames": len(rows),
        "split_counts": counts,
        "train_root": str(args.train_root),
        "val_root": str(args.val_root) if args.val_root else None,
        "test_root": str(args.test_root) if args.test_root else None,
        "reuse_train_manifest": str(args.reuse_train_manifest) if args.reuse_train_manifest else None,
        "train_frames_per_video": args.train_frames_per_video,
        "val_frames_per_video": args.val_frames_per_video,
        "test_frames_per_video": args.test_frames_per_video,
        "ann_every": args.ann_every,
        "seed": args.seed,
        "num_workers": worker_count,
    }
    (args.out_root / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
