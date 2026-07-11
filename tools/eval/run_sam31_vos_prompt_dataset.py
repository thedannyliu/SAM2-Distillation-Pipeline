#!/usr/bin/env python3
"""Run official SAM3.1 Object Multiplex memory tracking from GT box prompts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-root", required=True, type=Path)
    parser.add_argument("--sam31-checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--ann-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--video-list-file", type=Path)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def video_names(args: argparse.Namespace) -> list[str]:
    if args.video_list_file and args.video_list_file.is_file():
        names = [
            line.strip()
            for line in args.video_list_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        names = sorted(path.name for path in args.image_root.iterdir() if path.is_dir())
    names = [name for name in names if (args.image_root / name).is_dir()]
    return names[: args.max_videos] if args.max_videos > 0 else names


def frame_paths(video_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in video_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image) > 0


def normalized_xywh(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("empty prompt mask")
    height, width = mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0 / width, y0 / height, (x1 - x0 + 1) / width, (y1 - y0 + 1) / height]


def best_prompt_object(outputs: dict[str, Any]) -> int:
    ids = np.asarray(outputs["out_obj_ids"]).reshape(-1)
    if not len(ids):
        raise RuntimeError("SAM3.1 returned no object for a positive box prompt")
    scores = np.asarray(outputs.get("out_probs", np.ones(len(ids)))).reshape(-1)
    return int(ids[int(np.argmax(scores))])


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask).squeeze().astype(np.uint8) * 255).save(path)


def first_masks(ann_video_dir: Path) -> list[tuple[str, Path]]:
    rows = []
    for object_dir in sorted(path for path in ann_video_dir.iterdir() if path.is_dir()):
        masks = sorted(object_dir.glob("*.png"))
        if masks:
            rows.append((object_dir.name, masks[0]))
    return rows


def run_video(predictor, args: argparse.Namespace, video: str) -> dict[str, Any]:
    image_video_dir = args.image_root / video
    ann_video_dir = args.ann_root / video
    images = frame_paths(image_video_dir)
    stem_to_index = {path.stem: index for index, path in enumerate(images)}
    objects = first_masks(ann_video_dir)
    started = time.perf_counter()
    prediction_pngs = 0

    for gt_object_id, prompt_path in objects:
        if prompt_path.stem not in stem_to_index:
            raise FileNotFoundError(
                f"No image frame for prompt mask {prompt_path}"
            )
        prompt_index = stem_to_index[prompt_path.stem]
        gt_mask = load_mask(prompt_path)
        object_out_dir = args.out_dir / video / gt_object_id
        zero_mask = np.zeros(gt_mask.shape, dtype=bool)
        for annotated_mask in sorted((ann_video_dir / gt_object_id).glob("*.png")):
            save_mask(zero_mask, object_out_dir / annotated_mask.name)
        response = predictor.handle_request(
            {"type": "start_session", "resource_path": str(image_video_dir)}
        )
        session_id = response["session_id"]
        try:
            response = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": prompt_index,
                    "bounding_boxes": [normalized_xywh(gt_mask)],
                    "bounding_box_labels": [1],
                    "output_prob_thresh": 0.5,
                }
            )
            tracked_id = best_prompt_object(response["outputs"])
            for output in predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "both",
                    "start_frame_index": prompt_index,
                    "output_prob_thresh": 0.5,
                }
            ):
                frame_index = int(output["frame_index"])
                if frame_index < 0 or frame_index >= len(images):
                    continue
                ids = np.asarray(output["outputs"]["out_obj_ids"]).reshape(-1)
                matches = np.where(ids == tracked_id)[0]
                if not len(matches):
                    continue
                mask = output["outputs"]["out_binary_masks"][int(matches[0])]
                out_path = object_out_dir / f"{images[frame_index].stem}.png"
                if out_path.exists():
                    save_mask(mask, out_path)
                    prediction_pngs += 1
        finally:
            predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )

    return {
        "video": video,
        "objects": len(objects),
        "prediction_pngs": prediction_pngs,
        "elapsed_sec": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    for path in (
        args.sam3_root,
        args.sam31_checkpoint,
        args.checkpoint,
        args.image_root,
        args.ann_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(args.sam3_root))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Official SAM3.1 multiplex evaluation requires CUDA")

    from sam2_distill.models.sam31_stage1_inference import (
        build_sam31_multiplex_predictor,
        patch_multiplex_predictor_trunk,
    )

    predictor, builder_summary = build_sam31_multiplex_predictor(
        args.sam3_root,
        args.sam31_checkpoint,
        async_loading_frames=False,
    )
    load_summary = patch_multiplex_predictor_trunk(
        predictor, args.checkpoint, device
    )
    load_summary.update(builder_summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_videos = video_names(args)
    started = time.perf_counter()
    rows = [run_video(predictor, args, video) for video in selected_videos]
    elapsed = time.perf_counter() - started
    summary = {
        "status": "pass",
        "model_kind": "sam31-stage1-student",
        "prompt_kind": "box",
        "checkpoint": str(args.checkpoint),
        "sam31_checkpoint": str(args.sam31_checkpoint),
        "image_root": str(args.image_root),
        "ann_root": str(args.ann_root),
        "prediction_root": str(args.out_dir),
        "video_names": selected_videos,
        "videos": len(rows),
        "elapsed_sec": elapsed,
        "sec_per_video": elapsed / max(len(rows), 1),
        "num_prediction_pngs": sum(row["prediction_pngs"] for row in rows),
        "load": load_summary,
        "video_summaries": rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
