#!/usr/bin/env python3
"""Mine deterministic T4 SA-V hard videos with a frozen base task checkpoint."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sam2_distill.data.sav_task_dataset import resolve_sav_train_annotation_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sav-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--stage-checkpoint", required=True, type=Path)
    parser.add_argument("--sam2-checkpoint", required=True, type=Path)
    parser.add_argument("--student-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--student-model-name",
        default="tiny_vit_21m_512.dist_in22k_ft_in1k",
    )
    parser.add_argument("--seed", type=int, default=250107256)
    parser.add_argument("--max-objects", type=int, default=2)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def decode_rle(rle: Any) -> np.ndarray | None:
    if not isinstance(rle, dict):
        return None
    from pycocotools import mask as mask_utils

    counts = rle.get("counts")
    size = rle.get("size")
    if counts is None or size is None:
        return None
    if isinstance(counts, list):
        rle = mask_utils.frPyObjects(rle, int(size[0]), int(size[1]))
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(bool)


def boundary_f(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(np.uint8)
    target = target.astype(np.uint8)
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    kernel3 = np.ones((3, 3), dtype=np.uint8)
    pred_boundary = cv2.morphologyEx(prediction, cv2.MORPH_GRADIENT, kernel3) > 0
    target_boundary = cv2.morphologyEx(target, cv2.MORPH_GRADIENT, kernel3) > 0
    radius = max(1, int(math.ceil(0.008 * math.hypot(*target.shape))))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    tolerance = ((xx * xx + yy * yy) <= radius * radius).astype(np.uint8)
    pred_dilated = cv2.dilate(pred_boundary.astype(np.uint8), tolerance) > 0
    target_dilated = cv2.dilate(target_boundary.astype(np.uint8), tolerance) > 0
    precision = float((pred_boundary & target_dilated).sum()) / max(
        int(pred_boundary.sum()), 1
    )
    recall = float((target_boundary & pred_dilated).sum()) / max(
        int(target_boundary.sum()), 1
    )
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum()) / float(union)


def load_records(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    frame = pd.read_parquet(
        args.manifest,
        columns=[
            "video_id",
            "frame_idx_24fps",
            "image_path",
            "annotation_path",
            "split",
        ],
    )
    frame = frame[frame["split"] == "train"].sort_values(
        ["video_id", "frame_idx_24fps"]
    )
    records = []
    all_video_ids = []
    for video_id, rows in frame.groupby("video_id", sort=True):
        annotation_values = [
            value
            for value in rows["annotation_path"].tolist()
            if isinstance(value, str) and value.strip()
        ]
        annotation = resolve_sav_train_annotation_path(
            str(video_id),
            annotation_values[0] if annotation_values else None,
            args.sav_root,
        )
        if annotation is None:
            continue
        all_video_ids.append(str(video_id))
        if (len(all_video_ids) - 1) % world_size != rank:
            if args.max_videos > 0 and len(all_video_ids) >= args.max_videos:
                break
            continue
        by_frame_id = {
            int(frame_id): Path(image_path)
            for frame_id, image_path in zip(
                rows["frame_idx_24fps"], rows["image_path"], strict=True
            )
        }
        records.append(
            {
                "video_id": str(video_id),
                "annotation": str(annotation),
                "by_frame_id": by_frame_id,
            }
        )
        if args.max_videos > 0 and len(all_video_ids) >= args.max_videos:
            break
    return records, all_video_ids


def load_usable_frames(record: dict[str, Any]):
    payload = json.loads(Path(record["annotation"]).read_text(encoding="utf-8"))
    masklets = payload.get("masklet") if isinstance(payload, dict) else payload
    if not isinstance(masklets, list):
        raise ValueError("annotation has no masklet list")
    usable = []
    for annotation_index, frame_rles in enumerate(masklets):
        frame_id = annotation_index * 4
        if (
            frame_id in record["by_frame_id"]
            and isinstance(frame_rles, list)
            and None not in frame_rles
        ):
            usable.append((frame_id, record["by_frame_id"][frame_id], frame_rles))
    return usable


def fixed_clip(
    video_id: str,
    usable: list,
    seed: int,
    length: int = 4,
):
    if len(usable) < length:
        return None
    digest = hashlib.sha256(
        f"{seed}:{video_id}:T{length}".encode("utf-8")
    ).digest()
    start = int.from_bytes(digest[:8], "big") % (len(usable) - length + 1)
    return usable[start : start + length]


def mask_bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def score_video(
    predictor,
    record: dict[str, Any],
    usable: list,
    seed: int,
    max_objects: int,
):
    clip = fixed_clip(record["video_id"], usable, seed)
    if clip is None:
        raise ValueError("fewer than four usable frames")
    gt_by_frame = []
    for _, _, rles in clip:
        gt_by_frame.append([decode_rle(rle) for rle in rles])
    selected = [
        index
        for index, mask in enumerate(gt_by_frame[0])
        if mask is not None and mask.any()
    ][:max_objects]
    if not selected:
        raise ValueError("fixed clip starts without a visible object")

    with tempfile.TemporaryDirectory(prefix="sam2_hard_clip_") as temporary:
        video_dir = Path(temporary)
        for index, (_, source, _) in enumerate(clip):
            (video_dir / f"{index:05d}.jpg").symlink_to(source.resolve())
        state = predictor.init_state(video_path=str(video_dir))
        for object_id, object_index in enumerate(selected):
            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=object_id,
                box=mask_bbox(gt_by_frame[0][object_index]),
            )
        predictions = {}
        for frame_index, object_ids, logits in predictor.propagate_in_video(state):
            predictions[frame_index] = {
                int(object_id): logits[position].detach().float().cpu().numpy().squeeze() > 0
                for position, object_id in enumerate(object_ids)
            }
        predictor.reset_state(state)

    js = []
    fs = []
    for frame_index, gt_masks in enumerate(gt_by_frame):
        for object_id, object_index in enumerate(selected):
            target = gt_masks[object_index]
            if target is None:
                continue
            prediction = predictions.get(frame_index, {}).get(object_id)
            if prediction is None:
                prediction = np.zeros_like(target)
            js.append(mask_iou(prediction, target))
            fs.append(boundary_f(prediction, target))
    if not js:
        raise ValueError("fixed clip has no scoreable masks")
    j = float(np.mean(js))
    f = float(np.mean(fs))
    return {"J": j, "F": f, "J&F": (j + f) / 2, "objects": len(selected)}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["video_id"]] = row
    return rows


def write_list(path: Path, values: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    temporary.replace(path)


def input_fingerprint(args: argparse.Namespace, world_size: int) -> dict[str, Any]:
    def file_identity(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    return {
        "schema": "sav_mask_hardness_inputs_v1",
        "manifest": file_identity(args.manifest),
        "stage_checkpoint": file_identity(args.stage_checkpoint),
        "sam2_checkpoint": file_identity(args.sam2_checkpoint),
        "student_checkpoint": file_identity(args.student_checkpoint),
        "seed": args.seed,
        "max_objects": args.max_objects,
        "max_videos": args.max_videos,
        "world_size": world_size,
    }


def merge_results(args: argparse.Namespace, video_ids: list[str], world_size: int):
    rows = {}
    for rank in range(world_size):
        rows.update(read_jsonl(args.out_dir / f"hardness_rank_{rank:02d}.jsonl"))
    expected = set(video_ids)
    missing = sorted(expected.difference(rows))
    if missing:
        raise RuntimeError(f"Hardness shards are incomplete; missing {len(missing)} videos")
    ordered = [rows[video_id] for video_id in video_ids]
    scored = [row for row in ordered if row.get("status") == "pass"]
    scored.sort(key=lambda row: (float(row["J&F"]), row["video_id"]))
    if not scored:
        raise RuntimeError("No SA-V videos received a valid hardness score")
    hard = scored[: math.ceil(len(scored) / 2)]
    eligible_t8 = [row["video_id"] for row in ordered if row["usable_frames"] >= 8]
    if not eligible_t8:
        raise RuntimeError("No SA-V train video has at least eight usable frames")
    eligible_t16 = {row["video_id"] for row in ordered if row["usable_frames"] >= 16}
    hard_t16 = [row["video_id"] for row in hard if row["video_id"] in eligible_t16]
    if not hard_t16:
        raise RuntimeError("No bottom-half hard video has at least 16 usable frames")
    main_steps = math.ceil(len(eligible_t8) / 4)
    refine_steps = max(1, math.ceil(main_steps / 3))
    refine_samples = refine_steps * 4
    hard_t16_budget = [hard_t16[index % len(hard_t16)] for index in range(refine_samples)]
    write_list(args.out_dir / "hard50.txt", [row["video_id"] for row in hard])
    write_list(
        args.out_dir / "hard50_x2.txt",
        [row["video_id"] for row in hard for _ in range(2)],
    )
    write_list(args.out_dir / "eligible_t8.txt", eligible_t8)
    write_list(args.out_dir / "eligible_t16.txt", sorted(eligible_t16))
    write_list(args.out_dir / "hard_t16_budget.txt", hard_t16_budget)
    summary = {
        "status": "complete",
        "schema": "sav_mask_hardness_v1",
        "seed": args.seed,
        "manifest": str(args.manifest),
        "stage_checkpoint": str(args.stage_checkpoint),
        "videos": len(ordered),
        "scored_videos": len(scored),
        "failed_videos": len(ordered) - len(scored),
        "hard50_videos": len(hard),
        "eligible_t8_videos": len(eligible_t8),
        "eligible_t16_videos": len(eligible_t16),
        "a05_optimizer_updates": main_steps,
        "a06_refine_optimizer_updates": refine_steps,
        "a06_refine_samples": refine_samples,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = input_fingerprint(args, world_size)
    fingerprint_path = args.out_dir / "inputs.json"
    if fingerprint_path.is_file() and not args.force:
        existing_fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if existing_fingerprint != fingerprint:
            raise RuntimeError(
                f"Hardness inputs changed under {args.out_dir}; rerun with --force"
            )
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            "nccl", timeout=datetime.timedelta(hours=24)
        )
    if args.force and rank == 0:
        for path in args.out_dir.glob("hardness_rank_*.jsonl"):
            path.unlink()
        for name in (
            "summary.json",
            "hard50.txt",
            "hard50_x2.txt",
            "eligible_t8.txt",
            "eligible_t16.txt",
            "hard_t16_budget.txt",
        ):
            (args.out_dir / name).unlink(missing_ok=True)
    if rank == 0 and (args.force or not fingerprint_path.is_file()):
        temporary = fingerprint_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(fingerprint_path)
    if world_size > 1:
        torch.distributed.barrier()

    records, all_video_ids = load_records(args, rank, world_size)
    shard = records
    shard_path = args.out_dir / f"hardness_rank_{rank:02d}.jsonl"
    completed = read_jsonl(shard_path)
    pending = [record for record in shard if record["video_id"] not in completed]
    predictor = None
    if pending:
        from tools.eval.run_sam2_vos_prompt_dataset import (
            add_import_roots,
            autocast_context,
            build_predictor,
        )

        add_import_roots(args.sam2_root)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        predictor_args = SimpleNamespace(
            model_kind="stage1-student",
            sam2_root=args.sam2_root,
            sam2_cfg=args.sam2_cfg,
            checkpoint=args.stage_checkpoint,
            sam2_checkpoint=args.sam2_checkpoint,
            student_checkpoint=args.student_checkpoint,
            student_model_name=args.student_model_name,
            student_family="tinyvit",
        )
        predictor, _ = build_predictor(predictor_args, device)
        context = autocast_context(device)
    else:
        context = torch.inference_mode()
    with torch.inference_mode(), context:
        for index, record in enumerate(pending, 1):
            row = {
                "video_id": record["video_id"],
                "usable_frames": 0,
                "status": "fail",
            }
            try:
                usable = load_usable_frames(record)
                row["usable_frames"] = len(usable)
                if predictor is None:
                    raise ValueError("fewer than four usable frames")
                row.update(
                    score_video(
                        predictor,
                        record,
                        usable,
                        args.seed,
                        args.max_objects,
                    )
                )
                row["status"] = "pass"
            except Exception as error:  # Keep mining; failures remain auditable.
                row["error"] = f"{type(error).__name__}: {error}"
            append_jsonl(shard_path, row)
            if index % 100 == 0:
                print(
                    f"rank {rank}: {index}/{len(pending)} new videos",
                    flush=True,
                )
    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        merge_results(args, all_video_ids, world_size)
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
