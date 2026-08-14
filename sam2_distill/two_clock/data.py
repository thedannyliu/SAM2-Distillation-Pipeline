"""Online 24 FPS SA-V clips with sparse manual-target validity."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image

from sam2_distill.data.sav_task_dataset import (
    _video_number,
    resolve_sav_train_annotation_path,
)
from sam2_distill.two_clock.schedule import (
    fixed_refresh_trajectory,
    training_trajectory,
)


@dataclass
class SAVRawFrame:
    frame_idx: int
    image_path: str
    data: torch.Tensor | None = None
    is_conditioning_only: bool = False


@dataclass
class SAVRawVideo:
    video_name: str
    video_id: int
    frames: list[SAVRawFrame]

    def __len__(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class SampledFramesAndObjects:
    frames: list[SAVRawFrame]
    object_ids: list[int]


@dataclass
class SparseObject:
    object_id: int
    frame_index: int
    segment: torch.Tensor
    gt_valid: bool
    two_clock_refresh: bool
    two_clock_source_raw: int
    two_clock_age: int


@dataclass
class SparseFrame:
    data: Any
    objects: list[SparseObject]
    size: tuple[int, int] | None = None


@dataclass
class SparseVideoDatapoint:
    frames: list[SparseFrame]
    video_id: int
    size: tuple[int, int]


class SAVSparseSegmentLoader:
    """Load manual SA-V masks while preserving missing-target semantics."""

    def __init__(self, annotation_path: str | Path, frames_fps: int = 24) -> None:
        payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
        masklet_name = "masklet" if "masklet" in payload else "masks"
        self.frame_annots = payload[masklet_name]
        if not isinstance(self.frame_annots, list) or not self.frame_annots:
            raise ValueError(f"empty SA-V masklet: {annotation_path}")
        annotation_fps = payload.get("fps", 6)
        if isinstance(annotation_fps, list):
            annotation_fps = annotation_fps[0]
        annotation_fps = int(annotation_fps)
        if annotation_fps <= 0 or frames_fps % annotation_fps:
            raise ValueError(
                f"invalid annotation cadence {annotation_fps} for {frames_fps} FPS"
            )
        self.ann_every = frames_fps // annotation_fps
        self.num_objects = max(len(frame or []) for frame in self.frame_annots)

    def is_annotated(self, raw_frame_idx: int, object_id: int) -> bool:
        if raw_frame_idx < 0 or raw_frame_idx % self.ann_every:
            return False
        annotation_idx = raw_frame_idx // self.ann_every
        if annotation_idx >= len(self.frame_annots):
            return False
        frame = self.frame_annots[annotation_idx]
        return bool(frame is not None and object_id < len(frame) and frame[object_id] is not None)

    def visible_object_ids(self, raw_frame_idx: int) -> list[int]:
        if raw_frame_idx < 0 or raw_frame_idx % self.ann_every:
            return []
        annotation_idx = raw_frame_idx // self.ann_every
        if annotation_idx >= len(self.frame_annots):
            return []
        frame = self.frame_annots[annotation_idx]
        if frame is None:
            return []
        return [object_id for object_id, rle in enumerate(frame) if rle is not None]

    def valid_anchor_frames(self, raw_frame_count: int, clip_length: int) -> list[int]:
        return [
            annotation_idx * self.ann_every
            for annotation_idx in range(len(self.frame_annots))
            if annotation_idx * self.ann_every + clip_length <= raw_frame_count
            and self.visible_object_ids(annotation_idx * self.ann_every)
        ]

    def load(
        self,
        raw_frame_idx: int,
        obj_ids: list[int] | None = None,
    ) -> dict[int, torch.Tensor | None]:
        object_ids = list(range(self.num_objects)) if obj_ids is None else list(obj_ids)
        result = {object_id: None for object_id in object_ids}
        valid_ids = [
            object_id
            for object_id in object_ids
            if self.is_annotated(raw_frame_idx, object_id)
        ]
        if not valid_ids:
            return result

        from pycocotools import mask as mask_utils

        annotation = self.frame_annots[raw_frame_idx // self.ann_every]
        encoded = [annotation[object_id] for object_id in valid_ids]
        decoded = mask_utils.decode(encoded)
        if decoded.ndim == 2:
            decoded = decoded[..., None]
        for index, object_id in enumerate(valid_ids):
            result[object_id] = torch.from_numpy(decoded[..., index]).to(torch.uint8)
        return result


class SAV24FPSRawDataset:
    """Expose one raw MP4 plus manual masklet per usable manifest video."""

    def __init__(
        self,
        manifest: str | Path,
        split: str = "train",
        verify_paths: bool = False,
        max_videos: int = 0,
        sav_root: str | Path | None = None,
        video_ids_file: str | Path | None = None,
    ) -> None:
        manifest = Path(manifest)
        columns = [
            "video_id",
            "frame_idx_24fps",
            "annotation_path",
            "split",
        ]
        if manifest.suffix == ".csv":
            frame = pd.read_csv(manifest, usecols=lambda name: name in columns + ["video_path"])
        else:
            available = pd.read_parquet(manifest).columns
            frame = pd.read_parquet(
                manifest,
                columns=[name for name in columns + ["video_path"] if name in available],
            )
        frame = frame[frame["split"] == split].sort_values(
            ["video_id", "frame_idx_24fps"]
        )
        if frame.empty:
            raise ValueError(f"no rows for split {split!r} in {manifest}")

        requested = None
        if video_ids_file:
            requested = [
                line.strip()
                for line in Path(video_ids_file).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if not requested:
                raise ValueError(f"empty video ID file: {video_ids_file}")
        requested_set = set(requested) if requested is not None else None

        records: dict[str, tuple[str, Path, Path, int]] = {}
        for video_id, rows in frame.groupby("video_id", sort=True):
            video_id = str(video_id)
            if requested_set is not None and video_id not in requested_set:
                continue
            annotation_values = [
                value
                for value in rows["annotation_path"].tolist()
                if isinstance(value, str) and value.strip()
            ]
            annotation = resolve_sav_train_annotation_path(
                video_id,
                annotation_values[0] if annotation_values else None,
                sav_root,
            )
            if annotation is None:
                continue
            video_values = (
                [value for value in rows["video_path"].tolist() if isinstance(value, str) and value.strip()]
                if "video_path" in rows
                else []
            )
            if video_values:
                video_path = Path(video_values[0])
            elif sav_root is not None:
                shard = f"sav_{_video_number(video_id) // 1000:03d}"
                video_path = Path(sav_root) / "sav_train" / shard / f"{video_id}.mp4"
            else:
                continue
            raw_frame_count = int(rows["frame_idx_24fps"].max()) + 4
            if verify_paths and (not video_path.is_file() or not annotation.is_file()):
                raise FileNotFoundError(f"missing raw SA-V input for {video_id}")
            records[video_id] = (video_id, video_path, annotation, raw_frame_count)

        if requested is None:
            self.records = list(records.values())
        else:
            self.records = [records[video_id] for video_id in requested if video_id in records]
        if max_videos > 0:
            self.records = self.records[:max_videos]
        if not self.records:
            raise FileNotFoundError("no usable raw SA-V videos")

    def get_video(self, index: int) -> tuple[SAVRawVideo, SAVSparseSegmentLoader]:
        video_id, video_path, annotation_path, raw_frame_count = self.records[index]
        frames = [
            SAVRawFrame(frame_idx=frame_idx, image_path=str(video_path))
            for frame_idx in range(raw_frame_count)
        ]
        return (
            SAVRawVideo(video_id, _video_number(video_id), frames),
            SAVSparseSegmentLoader(annotation_path),
        )

    def __len__(self) -> int:
        return len(self.records)


class ContiguousSAV24FPSSampler:
    def __init__(
        self,
        num_frames: int = 8,
        max_num_objects: int = 3,
        seed: int = 250107256,
        force_full_refresh: bool = False,
        fixed_refresh_interval: int = 0,
    ) -> None:
        self.num_frames = num_frames
        self.max_num_objects = max_num_objects
        self.seed = seed
        self.force_full_refresh = force_full_refresh
        self.fixed_refresh_interval = fixed_refresh_interval

    def sample(
        self,
        video: SAVRawVideo,
        segment_loader: SAVSparseSegmentLoader,
        epoch: int | None = None,
    ) -> SampledFramesAndObjects:
        epoch = int(epoch or 0)
        anchors = segment_loader.valid_anchor_frames(len(video.frames), self.num_frames)
        if not anchors:
            raise ValueError(f"no valid T{self.num_frames} anchor in {video.video_name}")
        rng = random.Random(f"{self.seed}:{epoch}:{video.video_name}:sample")
        anchor = anchors[rng.randrange(len(anchors))]
        object_ids = segment_loader.visible_object_ids(anchor)
        rng.shuffle(object_ids)
        object_ids = object_ids[: self.max_num_objects]
        frames = video.frames[anchor : anchor + self.num_frames]

        if self.force_full_refresh:
            trajectory = fixed_refresh_trajectory(self.num_frames, 1)
        elif self.fixed_refresh_interval:
            trajectory = fixed_refresh_trajectory(
                self.num_frames, self.fixed_refresh_interval
            )
        else:
            trajectory = training_trajectory(
                self.num_frames,
                seed=self.seed,
                epoch=epoch,
                video_id=video.video_name,
            )
        for index, frame in enumerate(frames):
            frame.two_clock_refresh = trajectory.refresh[index]
            frame.two_clock_source_raw = frames[trajectory.source[index]].frame_idx
            frame.two_clock_age = trajectory.age[index]
        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)


def decode_contiguous_rgb(
    video_path: str | Path,
    frame_indices: list[int],
) -> list[Image.Image]:
    if not frame_indices:
        raise ValueError("frame_indices must be non-empty")
    if frame_indices != list(range(frame_indices[0], frame_indices[0] + len(frame_indices))):
        raise ValueError("two-clock v1 requires contiguous raw frame indices")
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
    images = []
    try:
        for frame_idx in frame_indices:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"could not decode frame {frame_idx} from {video_path}")
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    return images


class SparseSAVVOSDataset:
    """SAM2 VOSDataset-compatible wrapper that keeps sparse-target validity."""

    def __init__(
        self,
        transforms,
        training: bool,
        video_dataset: SAV24FPSRawDataset,
        sampler: ContiguousSAV24FPSSampler,
        multiplier: int = 1,
    ) -> None:
        self._transforms = transforms
        self.training = training
        self.video_dataset = video_dataset
        self.sampler = sampler
        self.repeat_factors = torch.ones(len(video_dataset), dtype=torch.float32) * multiplier
        self.curr_epoch = 0

    def _get_datapoint(self, index: int) -> SparseVideoDatapoint:
        last_error = None
        for retry in range(100):
            try:
                video, loader = self.video_dataset.get_video(index)
                sampled = self.sampler.sample(video, loader, epoch=self.curr_epoch)
                datapoint = self.construct(video, sampled, loader)
                for transform in self._transforms:
                    datapoint = transform(datapoint, epoch=self.curr_epoch)
                return datapoint
            except Exception as error:
                last_error = error
                if not self.training:
                    raise
                logging.warning("two-clock data retry %d for index %s: %s", retry, index, error)
                index = random.randrange(len(self.video_dataset))
        raise RuntimeError("failed to load a two-clock SA-V sample") from last_error

    def construct(
        self,
        video: SAVRawVideo,
        sampled: SampledFramesAndObjects,
        loader: SAVSparseSegmentLoader,
    ) -> SparseVideoDatapoint:
        raw_indices = [frame.frame_idx for frame in sampled.frames]
        images = decode_contiguous_rgb(sampled.frames[0].image_path, raw_indices)
        frames = []
        size = None
        for raw_frame, image in zip(sampled.frames, images):
            width, height = image.size
            size = (height, width)
            segments = loader.load(raw_frame.frame_idx, sampled.object_ids)
            objects = []
            for object_id in sampled.object_ids:
                segment = segments[object_id]
                gt_valid = segment is not None
                if segment is None:
                    segment = torch.zeros(height, width, dtype=torch.uint8)
                objects.append(
                    SparseObject(
                        object_id=object_id,
                        frame_index=raw_frame.frame_idx,
                        segment=segment,
                        gt_valid=gt_valid,
                        two_clock_refresh=bool(raw_frame.two_clock_refresh),
                        two_clock_source_raw=int(raw_frame.two_clock_source_raw),
                        two_clock_age=int(raw_frame.two_clock_age),
                    )
                )
            frames.append(SparseFrame(data=image, objects=objects))
        if size is None:
            raise ValueError("empty sampled clip")
        return SparseVideoDatapoint(frames=frames, video_id=video.video_id, size=size)

    def __getitem__(self, index: int) -> SparseVideoDatapoint:
        return self._get_datapoint(int(index))

    def __len__(self) -> int:
        return len(self.video_dataset)


def collate_two_clock_fn(batch: list[SparseVideoDatapoint], dict_key: str):
    """Collate with metadata columns [video, object, raw, GT, R, source, age]."""
    from training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData

    img_batch = torch.stack(
        [torch.stack([frame.data for frame in video.frames]) for video in batch]
    ).permute(1, 0, 2, 3, 4)
    num_frames = img_batch.shape[0]
    obj_to_frame = [[] for _ in range(num_frames)]
    masks = [[] for _ in range(num_frames)]
    identifiers = [[] for _ in range(num_frames)]
    sizes = [[] for _ in range(num_frames)]
    for video_index, video in enumerate(batch):
        for stage, frame in enumerate(video.frames):
            for object_ in frame.objects:
                obj_to_frame[stage].append(torch.tensor([stage, video_index], dtype=torch.int))
                masks[stage].append(object_.segment.to(torch.bool))
                identifiers[stage].append(
                    torch.tensor(
                        [
                            video.video_id,
                            object_.object_id,
                            object_.frame_index,
                            int(object_.gt_valid),
                            int(object_.two_clock_refresh),
                            object_.two_clock_source_raw,
                            object_.two_clock_age,
                        ],
                        dtype=torch.long,
                    )
                )
                sizes[stage].append(torch.tensor(video.size, dtype=torch.long))
    return BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=torch.stack([torch.stack(values) for values in obj_to_frame]),
        masks=torch.stack([torch.stack(values) for values in masks]),
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=torch.stack(
                [torch.stack(values) for values in identifiers]
            ),
            frame_orig_size=torch.stack([torch.stack(values) for values in sizes]),
            batch_size=[num_frames],
        ),
        dict_key=dict_key,
        batch_size=[num_frames],
    )
