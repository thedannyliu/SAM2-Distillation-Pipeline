#!/usr/bin/env python3
"""Benchmark promptable image masks on SA-V validation annotations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ObjectRecord:
    video: str
    object_id: str
    frame_stem: str
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-kind",
        choices=("edgetam-trainer", "sam2", "stage1-student", "sam31-stage1-student"),
        required=True,
    )
    parser.add_argument("--prompt-kind", choices=("box", "point"), required=True)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--ann-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, help="Trainer/model config for edgetam-trainer, SAM2 config name for sam2.")
    parser.add_argument("--sam2-checkpoint", type=Path, help="SAM2 checkpoint for prompt encoder/mask decoder with stage1-student.")
    parser.add_argument("--tinyvit-checkpoint", type=Path, help="TinyViT pretrained checkpoint used to instantiate stage1-student.")
    parser.add_argument("--tinyvit-model-name", default="tiny_vit_21m_512.dist_in22k_ft_in1k")
    parser.add_argument("--sam2-root", type=Path, default=Path("/user-volume/repo/facebookresearch-sam2"))
    parser.add_argument("--edgetam-root", type=Path, default=Path("/user-volume/repo/EdgeTAM"))
    parser.add_argument("--sam3-root", type=Path, default=Path("/user-volume/repo/facebookresearch-sam3"))
    parser.add_argument("--sam31-checkpoint", type=Path)
    parser.add_argument("--video-list-file", type=Path)
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all.")
    parser.add_argument("--max-objects", type=int, default=2000, help="0 means all.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--warmup-images", type=int, default=5)
    parser.add_argument("--save-artifacts", type=int, default=0, help="Save this many predicted masks and overlays.")
    parser.add_argument(
        "--save-video-frame-artifacts",
        type=int,
        default=0,
        help="For this many videos, save combined first/middle/last frame overlays and masks.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def add_import_roots(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    for root in (args.sam2_root, args.edgetam_root, args.sam3_root):
        if root.exists():
            sys.path.insert(0, str(root))


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


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


def load_edgetam_predictor(args: argparse.Namespace, device: torch.device):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2_distill.edgetam.compat import patch_edgetam_perceiver_view

    patch_edgetam_perceiver_view()
    cfg = OmegaConf.load(args.config)
    model_cfg = cfg.model if "model" in cfg else cfg.trainer.model
    model_cfg._target_ = "sam2_distill.edgetam.train_model.EdgeTAMTrain"
    if "synthetic_teacher" in model_cfg:
        del model_cfg.synthetic_teacher
    if "teacher_feature_cache_path" in model_cfg:
        del model_cfg.teacher_feature_cache_path
    if "synthetic_teacher_offset" in model_cfg:
        del model_cfg.synthetic_teacher_offset
    model = instantiate(model_cfg, _recursive_=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return SAM2ImagePredictor(model), {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_steps": checkpoint.get("steps"),
        "num_tensors": len(state_dict),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def load_sam2_predictor(args: argparse.Namespace, device: torch.device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(args.config, str(args.checkpoint), device=str(device), mode="eval")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return SAM2ImagePredictor(model), {"checkpoint": str(args.checkpoint), "config": args.config}


def load_stage1_student_predictor(args: argparse.Namespace, device: torch.device):
    if args.sam2_checkpoint is None:
        raise SystemExit("--sam2-checkpoint is required for --model-kind stage1-student")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2_distill.models.stage1_checkpoint import (
        infer_adapter_mode,
        infer_tinyvit_model_name,
        resolve_tinyvit_checkpoint,
    )
    from sam2_distill.models.tinyvit_adapter import TinyViTSAM2Adapter

    model = build_sam2(args.config, str(args.sam2_checkpoint), device=str(device), mode="eval")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

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

    predictor = SAM2ImagePredictor(model)
    predictor._stage1_student = student
    return predictor, {
        "student_checkpoint": str(args.checkpoint),
        "sam2_checkpoint": str(args.sam2_checkpoint),
        "tinyvit_checkpoint": str(tinyvit_checkpoint) if tinyvit_checkpoint is not None else None,
        "requested_tinyvit_checkpoint": str(args.tinyvit_checkpoint) if args.tinyvit_checkpoint is not None else None,
        "tinyvit_model_name": tinyvit_model_name,
        "adapter_mode": adapter_mode,
        "requested_tinyvit_model_name": args.tinyvit_model_name,
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "num_tensors": len(state_dict),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


class SAM31GeometryPredictor:
    """Expose the existing image benchmark predictor interface for SAM3.1."""

    def __init__(self, processor) -> None:
        self.processor = processor
        self.state = None

    def set_image(self, image_np: np.ndarray) -> None:
        image = Image.fromarray(np.asarray(image_np, dtype=np.uint8).copy(), mode="RGB")
        self.state = self.processor.set_image(image)

    def predict(self, *, box, multimask_output=False):
        del multimask_output
        if self.state is None:
            raise RuntimeError("set_image must be called before predict")
        self.processor.reset_all_prompts(self.state)
        height = float(self.state["original_height"])
        width = float(self.state["original_width"])
        x0, y0, x1, y1 = np.asarray(box, dtype=np.float32).reshape(-1)[:4]
        geometric_box = [
            float((x0 + x1) * 0.5 / width),
            float((y0 + y1) * 0.5 / height),
            float((x1 - x0 + 1.0) / width),
            float((y1 - y0 + 1.0) / height),
        ]
        self.processor.add_geometric_prompt(geometric_box, True, self.state)
        masks = self.state["masks"].detach().cpu().numpy()
        scores = self.state["scores"].detach().float().cpu().numpy()
        logits = self.state["masks_logits"].detach().float().cpu().numpy()
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
            logits = logits[:, 0]
        return masks, scores, logits


def load_sam31_stage1_predictor(args: argparse.Namespace, device: torch.device):
    if args.sam31_checkpoint is None:
        raise SystemExit("--sam31-checkpoint is required for sam31-stage1-student")
    if args.prompt_kind != "box":
        raise SystemExit("SAM3.1 Stage 1 image evaluation currently requires box prompts")

    from sam2_distill.models.sam31_stage1_inference import (
        build_sam31_multiplex_predictor,
        patch_multiplex_predictor_trunk,
    )

    multiplex, builder_summary = build_sam31_multiplex_predictor(
        args.sam3_root,
        args.sam31_checkpoint,
        async_loading_frames=False,
    )
    from sam3.model.sam3_image_processor import Sam3Processor

    load_summary = patch_multiplex_predictor_trunk(
        multiplex, args.checkpoint, device
    )
    load_summary.update(builder_summary)
    processor = Sam3Processor(
        multiplex.model.detector,
        resolution=1008,
        device=str(device),
        confidence_threshold=0.0,
    )
    predictor = SAM31GeometryPredictor(processor)
    predictor._multiplex_predictor = multiplex
    load_summary["sam31_checkpoint"] = str(args.sam31_checkpoint)
    return predictor, load_summary


def set_image_with_stage1_student(predictor, image_np: np.ndarray, device: torch.device) -> None:
    image = Image.fromarray(image_np)
    predictor.reset_predictor()
    predictor._orig_hw = [(image.height, image.width)]
    input_image = predictor._transforms(image)
    input_image = input_image[None, ...].to(device)
    with torch.inference_mode(), autocast_context(device):
        features = predictor._stage1_student(input_image)
    predictor._features = {
        "image_embed": features["image_embed"],
        "high_res_feats": [features["high_res_s0"], features["high_res_s1"]],
    }
    predictor._is_image_set = True
    predictor._is_batch = False


def set_image_for_model(predictor, model_kind: str, image_np: np.ndarray, device: torch.device) -> None:
    if model_kind == "stage1-student":
        set_image_with_stage1_student(predictor, image_np, device)
    else:
        predictor.set_image(image_np)


def image_for_mask(image_video_dir: Path, frame_stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = image_video_dir / f"{frame_stem}{suffix}"
        if candidate.exists():
            return candidate
    if frame_stem.isdigit():
        value = int(frame_stem)
        for mapped in (value * 4, value // 4):
            for width in (len(frame_stem), 6, 5, 4):
                mapped_stem = f"{mapped:0{width}d}"
                for suffix in (".jpg", ".jpeg", ".png"):
                    candidate = image_video_dir / f"{mapped_stem}{suffix}"
                    if candidate.exists():
                        return candidate
    return None


def collect_records(args: argparse.Namespace) -> list[ObjectRecord]:
    if not args.image_root.exists():
        raise FileNotFoundError(args.image_root)
    if not args.ann_root.exists():
        raise FileNotFoundError(args.ann_root)

    records: list[ObjectRecord] = []
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.video_list_file is not None:
        videos = [
            line.strip()
            for line in args.video_list_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        videos = sorted(path.name for path in args.ann_root.iterdir() if path.is_dir())
    videos = [video for video in videos if (args.ann_root / video).is_dir()]
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    videos = videos[args.shard_index :: args.num_shards]
    for video in videos:
        image_video_dir = args.image_root / video
        ann_video_dir = args.ann_root / video
        if not image_video_dir.exists():
            continue
        for object_dir in sorted(path for path in ann_video_dir.iterdir() if path.is_dir()):
            for mask_path in sorted(object_dir.glob("*.png")):
                image_path = image_for_mask(image_video_dir, mask_path.stem)
                if image_path is None:
                    continue
                records.append(
                    ObjectRecord(
                        video=video,
                        object_id=object_dir.name,
                        frame_stem=mask_path.stem,
                        image_path=image_path,
                        mask_path=mask_path,
                    )
                )
                if args.max_objects > 0 and len(records) >= args.max_objects:
                    return records
    return records


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image) > 0


def save_mask_and_overlay(image_path: Path, pred_mask: np.ndarray, gt_mask: np.ndarray, out_dir: Path, name: str) -> None:
    mask_dir = out_dir / "masks"
    overlay_dir = out_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred_mask.astype(np.uint8) * 255).save(mask_dir / f"{name}_pred.png")
    Image.fromarray(gt_mask.astype(np.uint8) * 255).save(mask_dir / f"{name}_gt.png")

    with Image.open(image_path) as image:
        base = image.convert("RGBA")
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    color = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    color[gt] = [0, 255, 0, 90]
    color[pred] = [255, 0, 0, 90]
    color[np.logical_and(gt, pred)] = [255, 220, 0, 130]
    overlay = Image.alpha_composite(base, Image.fromarray(color, mode="RGBA"))
    overlay.convert("RGB").save(overlay_dir / f"{name}_overlay.jpg", quality=92)


def save_combined_mask_and_overlay(
    image_path: Path,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    out_dir: Path,
    name: str,
) -> None:
    frame_dir = out_dir / "frame_artifacts"
    mask_dir = frame_dir / "masks"
    overlay_dir = frame_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred_mask.astype(np.uint8) * 255).save(mask_dir / f"{name}_pred.png")
    Image.fromarray(gt_mask.astype(np.uint8) * 255).save(mask_dir / f"{name}_gt.png")

    with Image.open(image_path) as image:
        base = image.convert("RGBA")
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    color = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    color[gt] = [0, 255, 0, 90]
    color[pred] = [255, 0, 0, 90]
    color[np.logical_and(gt, pred)] = [255, 220, 0, 130]
    overlay = Image.alpha_composite(base, Image.fromarray(color, mode="RGBA"))
    overlay.convert("RGB").save(overlay_dir / f"{name}_overlay.jpg", quality=92)


def frame_sort_key(record: ObjectRecord) -> tuple[int, str]:
    return (int(record.frame_stem), record.frame_stem) if record.frame_stem.isdigit() else (10**12, record.frame_stem)


def select_video_frame_artifacts(records: list[ObjectRecord], max_videos: int) -> dict[Path, str]:
    if max_videos <= 0:
        return {}
    by_video: dict[str, dict[str, ObjectRecord]] = defaultdict(dict)
    for record in records:
        by_video[record.video].setdefault(record.frame_stem, record)

    targets: dict[Path, str] = {}
    for video in sorted(by_video)[:max_videos]:
        frames = sorted(by_video[video].values(), key=frame_sort_key)
        if not frames:
            continue
        picks = [
            ("first", frames[0]),
            ("middle", frames[len(frames) // 2]),
            ("last", frames[-1]),
        ]
        for label, record in picks:
            targets[record.image_path] = f"{video}_{label}_{record.frame_stem}"
    return targets


def mask_bbox(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def mask_point(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    cx = xs.mean()
    cy = ys.mean()
    best = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return np.asarray([[xs[best], ys[best]]], dtype=np.float32)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def average_precision(ious: list[float], scores: list[float], threshold: float) -> float:
    if not ious:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray([ious[idx] >= threshold for idx in order], dtype=np.float32)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(len(ious), 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    ap = 0.0
    for recall_threshold in np.linspace(0.0, 1.0, 101):
        valid = precisions[recalls >= recall_threshold]
        ap += float(valid.max()) if valid.size else 0.0
    return ap / 101.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def predict_one(predictor, prompt_kind: str, gt_mask: np.ndarray):
    if prompt_kind == "box":
        box = mask_bbox(gt_mask)
        if box is None:
            return None
        return predictor.predict(box=box, multimask_output=False)
    point = mask_point(gt_mask)
    if point is None:
        return None
    labels = np.asarray([1], dtype=np.int32)
    return predictor.predict(point_coords=point, point_labels=labels, multimask_output=True)


def main() -> None:
    args = parse_args()
    add_import_roots(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    records = collect_records(args)
    if not records:
        raise RuntimeError(f"No benchmarkable object masks under {args.ann_root}")

    if args.model_kind == "edgetam-trainer":
        predictor, load_summary = load_edgetam_predictor(args, device)
    elif args.model_kind == "sam2":
        predictor, load_summary = load_sam2_predictor(args, device)
    elif args.model_kind == "stage1-student":
        predictor, load_summary = load_stage1_student_predictor(args, device)
    else:
        predictor, load_summary = load_sam31_stage1_predictor(args, device)

    by_image: dict[Path, list[ObjectRecord]] = defaultdict(list)
    for record in records:
        by_image[record.image_path].append(record)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    set_image_latencies: list[float] = []
    prompt_latencies: list[float] = []
    saved_artifacts = 0
    saved_frame_artifacts = 0
    frame_artifact_targets = select_video_frame_artifacts(records, args.save_video_frame_artifacts)

    image_items = list(by_image.items())
    with torch.inference_mode():
        for warm_image_path, _ in image_items[: max(args.warmup_images, 0)]:
            with Image.open(warm_image_path) as image:
                set_image_for_model(predictor, args.model_kind, np.asarray(image.convert("RGB")), device)
            sync(device)

        for image_path, image_records in image_items:
            with Image.open(image_path) as image:
                image_np = np.asarray(image.convert("RGB"))
            sync(device)
            start = time.perf_counter()
            set_image_for_model(predictor, args.model_kind, image_np, device)
            sync(device)
            set_image_sec = time.perf_counter() - start
            set_image_latencies.append(set_image_sec)
            target_name = frame_artifact_targets.get(image_path)
            frame_pred_union = np.zeros(image_np.shape[:2], dtype=bool) if target_name is not None else None
            frame_gt_union = np.zeros(image_np.shape[:2], dtype=bool) if target_name is not None else None

            for record in image_records:
                gt_mask = load_mask(record.mask_path)
                sync(device)
                prompt_start = time.perf_counter()
                prediction = predict_one(predictor, args.prompt_kind, gt_mask)
                sync(device)
                prompt_sec = time.perf_counter() - prompt_start
                if prediction is None:
                    continue
                masks, scores, _ = prediction
                scores_np = np.asarray(scores).reshape(-1)
                best_idx = int(np.argmax(scores_np))
                pred_mask = masks[best_idx] if masks.ndim == 3 else masks
                row_iou = iou(pred_mask, gt_mask)
                prompt_latencies.append(prompt_sec)
                rows.append(
                    {
                        "video": record.video,
                        "object_id": record.object_id,
                        "frame_stem": record.frame_stem,
                        "image_path": str(record.image_path),
                        "mask_path": str(record.mask_path),
                        "iou": row_iou,
                        "score": float(scores_np[best_idx]),
                        "num_candidates": int(len(scores_np)),
                        "set_image_seconds": set_image_sec,
                        "prompt_seconds": prompt_sec,
                        "total_object_seconds": set_image_sec + prompt_sec,
                    }
                )
                if saved_artifacts < args.save_artifacts:
                    artifact_name = (
                        f"{saved_artifacts:04d}_{record.video}_{record.object_id}_{record.frame_stem}_{args.prompt_kind}"
                    )
                    save_mask_and_overlay(record.image_path, pred_mask, gt_mask, args.out_dir, artifact_name)
                    saved_artifacts += 1
                if frame_pred_union is not None and frame_gt_union is not None:
                    frame_pred_union |= pred_mask.astype(bool)
                    frame_gt_union |= gt_mask.astype(bool)
            if target_name is not None and frame_pred_union is not None and frame_gt_union is not None:
                save_combined_mask_and_overlay(
                    image_path,
                    frame_pred_union,
                    frame_gt_union,
                    args.out_dir,
                    f"{target_name}_{args.prompt_kind}",
                )
                saved_frame_artifacts += 1

    thresholds = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
    ious = [float(row["iou"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    ap_by_threshold = {f"AP{int(t * 100)}": average_precision(ious, scores, t) for t in thresholds}
    summary = {
        "status": "pass",
        "model_kind": args.model_kind,
        "prompt_kind": args.prompt_kind,
        "config": args.config,
        "checkpoint": str(args.checkpoint),
        "image_root": str(args.image_root),
        "ann_root": str(args.ann_root),
        "out_dir": str(args.out_dir),
        "num_images": len(by_image),
        "num_objects": len(rows),
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "median_IoU": percentile(ious, 50),
        "AP": float(np.mean(list(ap_by_threshold.values()))) if ap_by_threshold else 0.0,
        **ap_by_threshold,
        "latency": {
            "mean_set_image_seconds": float(np.mean(set_image_latencies)) if set_image_latencies else 0.0,
            "p50_set_image_seconds": percentile(set_image_latencies, 50),
            "p95_set_image_seconds": percentile(set_image_latencies, 95),
            "mean_prompt_seconds": float(np.mean(prompt_latencies)) if prompt_latencies else 0.0,
            "p50_prompt_seconds": percentile(prompt_latencies, 50),
            "p95_prompt_seconds": percentile(prompt_latencies, 95),
            "mean_total_object_seconds": float(np.mean([row["total_object_seconds"] for row in rows])) if rows else 0.0,
        },
        "artifacts_saved": saved_artifacts,
        "frame_artifacts_saved": saved_frame_artifacts,
        "load": load_summary,
    }

    csv_path = args.out_dir / "per_object_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["video"])
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
