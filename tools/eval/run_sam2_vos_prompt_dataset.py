#!/usr/bin/env python3
"""Run SAM2 VOS with point/box prompts on a generic per-object PNG dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=("sam2", "stage1-student"), default="sam2")
    parser.add_argument("--prompt-kind", choices=("box", "point"), required=True)
    parser.add_argument("--sam2-root", required=True, type=Path)
    parser.add_argument("--sam2-cfg", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sam2-checkpoint", type=Path, help="Full SAM2 checkpoint for stage1-student.")
    parser.add_argument("--tinyvit-checkpoint", type=Path)
    parser.add_argument("--tinyvit-model-name", default="tiny_vit_21m_512.dist_in22k_ft_in1k")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--ann-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--video-list-file", type=Path)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def add_import_roots(sam2_root: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(sam2_root))


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model", "model_state", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return strip_module_prefix(value)
    raise KeyError("checkpoint must contain one of: model, model_state, state_dict")


def patch_stage1_forward_image(predictor, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if args.sam2_checkpoint is None:
        raise SystemExit("--sam2-checkpoint is required for --model-kind stage1-student")

    from sam2_distill.models.stage1_checkpoint import (
        infer_adapter_mode,
        infer_tinyvit_model_name,
        resolve_tinyvit_checkpoint,
    )
    from sam2_distill.models.tinyvit_adapter import TinyViTSAM2Adapter

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    tinyvit_model_name = infer_tinyvit_model_name(state_dict, args.tinyvit_model_name)
    adapter_mode = infer_adapter_mode(checkpoint, state_dict)
    tinyvit_checkpoint = (
        resolve_tinyvit_checkpoint(tinyvit_model_name, args.tinyvit_checkpoint)
        if args.tinyvit_checkpoint is not None
        else None
    )
    student = TinyViTSAM2Adapter(
        model_name=tinyvit_model_name,
        checkpoint_path=str(tinyvit_checkpoint) if tinyvit_checkpoint is not None else None,
        adapter_mode=adapter_mode,
    ).to(device)
    incompatible = student.load_state_dict(state_dict, strict=False)
    student.eval()
    for param in student.parameters():
        param.requires_grad_(False)

    position_encoding = predictor.image_encoder.neck.position_encoding

    @torch.inference_mode()
    def forward_image(self, img_batch: torch.Tensor):
        features = student(img_batch)
        backbone_fpn = [
            features["high_res_s0"].float(),
            features["high_res_s1"].float(),
            features["image_embed"].float(),
        ]
        vision_pos_enc = [position_encoding(feat).float() for feat in backbone_fpn]
        return {
            "vision_features": backbone_fpn[-1],
            "vision_pos_enc": vision_pos_enc,
            "backbone_fpn": backbone_fpn,
        }

    predictor.forward_image = types.MethodType(forward_image, predictor)
    predictor._stage1_student = student
    return {
        "student_checkpoint": str(args.checkpoint),
        "sam2_checkpoint": str(args.sam2_checkpoint),
        "tinyvit_checkpoint": str(tinyvit_checkpoint) if tinyvit_checkpoint is not None else None,
        "requested_tinyvit_checkpoint": str(args.tinyvit_checkpoint) if args.tinyvit_checkpoint is not None else None,
        "tinyvit_model_name": tinyvit_model_name,
        "adapter_mode": adapter_mode,
        "requested_tinyvit_model_name": args.tinyvit_model_name,
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def build_predictor(args: argparse.Namespace, device: torch.device):
    from sam2.build_sam import build_sam2_video_predictor
    from sam2_distill.edgetam.compat import patch_edgetam_perceiver_view

    patch_edgetam_perceiver_view()
    full_checkpoint = args.sam2_checkpoint if args.model_kind == "stage1-student" else args.checkpoint
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=str(full_checkpoint),
        device=str(device),
        apply_postprocessing=False,
        hydra_overrides_extra=["++model.non_overlap_masks=false"],
    )
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad_(False)

    load_summary = {
        "model_kind": args.model_kind,
        "sam2_cfg": args.sam2_cfg,
        "checkpoint": str(args.checkpoint),
    }
    if args.model_kind == "stage1-student":
        load_summary["stage1"] = patch_stage1_forward_image(predictor, args, device)
    return predictor, load_summary


def video_names(
    image_root: Path,
    video_list_file: Path | None,
    max_videos: int,
    num_shards: int,
    shard_index: int,
) -> list[str]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if video_list_file is not None:
        names = [line.strip() for line in video_list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        names = sorted(path.name for path in image_root.iterdir() if path.is_dir())
    names = [name for name in names if (image_root / name).is_dir()]
    names = names[:max_videos] if max_videos > 0 else names
    return names[shard_index::num_shards]


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image) > 0


def mask_bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def mask_point(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    cx = xs.mean()
    cy = ys.mean()
    best = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    points = np.asarray([[xs[best], ys[best]]], dtype=np.float32)
    labels = np.asarray([1], dtype=np.int32)
    return points, labels


def first_mask_per_object(ann_video_dir: Path) -> list[tuple[str, int, Path]]:
    records = []
    for object_dir in sorted(path for path in ann_video_dir.iterdir() if path.is_dir()):
        masks = sorted(object_dir.glob("*.png"))
        if not masks:
            continue
        records.append((object_dir.name, int(masks[0].stem), masks[0]))
    return records


def save_prediction(mask_logits: torch.Tensor, out_path: Path) -> None:
    mask = (mask_logits.detach().cpu().numpy() > 0).astype(np.uint8) * 255
    if mask.ndim == 3:
        mask = mask[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out_path)


def run_video(predictor, args: argparse.Namespace, video: str) -> dict[str, Any]:
    ann_video_dir = args.ann_root / video
    image_video_dir = args.image_root / video
    object_prompts = first_mask_per_object(ann_video_dir)
    if not object_prompts:
        return {"video": video, "status": "skipped", "reason": "no object masks"}

    state = predictor.init_state(video_path=str(image_video_dir))
    prompt_rows = []
    for obj_id, frame_idx, mask_path in object_prompts:
        mask = load_mask(mask_path)
        if args.prompt_kind == "box":
            box = mask_bbox(mask)
            predictor.add_new_points_or_box(state, frame_idx=frame_idx, obj_id=obj_id, box=box)
            prompt_rows.append({"object_id": obj_id, "frame_idx": frame_idx, "prompt": "box", "box": box.tolist()})
        else:
            points, labels = mask_point(mask)
            predictor.add_new_points_or_box(
                state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )
            prompt_rows.append(
                {
                    "object_id": obj_id,
                    "frame_idx": frame_idx,
                    "prompt": "point",
                    "points": points.tolist(),
                    "labels": labels.tolist(),
                }
            )

    pngs = 0
    for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(state):
        for obj_idx, obj_id in enumerate(obj_ids):
            out_path = args.out_dir / video / str(obj_id) / f"{frame_idx:05d}.png"
            save_prediction(video_res_masks[obj_idx], out_path)
            pngs += 1

    return {
        "video": video,
        "status": "pass",
        "objects": len(object_prompts),
        "prompts": prompt_rows,
        "prediction_pngs": pngs,
    }


def main() -> None:
    args = parse_args()
    for path in (args.sam2_root, args.checkpoint, args.image_root, args.ann_root):
        if not path.exists():
            raise FileNotFoundError(path)
    add_import_roots(args.sam2_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    predictor, load_summary = build_predictor(args, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_videos = video_names(
        args.image_root,
        args.video_list_file,
        args.max_videos,
        args.num_shards,
        args.shard_index,
    )
    start = time.perf_counter()
    with torch.inference_mode(), autocast_context(device):
        video_summaries = [run_video(predictor, args, video) for video in selected_videos]
    elapsed = time.perf_counter() - start
    passed = [summary for summary in video_summaries if summary.get("status") == "pass"]
    summary = {
        "status": "pass",
        "model_kind": args.model_kind,
        "prompt_kind": args.prompt_kind,
        "sam2_cfg": args.sam2_cfg,
        "checkpoint": str(args.checkpoint),
        "image_root": str(args.image_root),
        "ann_root": str(args.ann_root),
        "prediction_root": str(args.out_dir),
        "video_names": selected_videos,
        "elapsed_sec": elapsed,
        "videos": len(passed),
        "sec_per_video": elapsed / max(len(passed), 1),
        "num_prediction_pngs": sum(int(row.get("prediction_pngs", 0)) for row in passed),
        "load": load_summary,
        "video_summaries": video_summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
