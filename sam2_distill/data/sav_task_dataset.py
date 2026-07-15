"""SA-V task-training adapter backed by the mounted Stage 1 manifest."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd


def _video_number(video_id: str) -> int:
    return int(video_id.rsplit("_", 1)[-1])


def resolve_sav_train_annotation_path(
    video_id: str,
    annotation_value: object,
    sav_root: str | Path | None,
) -> Path | None:
    """Resolve a manual SA-V annotation, including blank mounted manifests."""
    if isinstance(annotation_value, str) and annotation_value.strip():
        candidate = Path(annotation_value.strip())
        if candidate.is_file():
            return candidate
    if sav_root is None:
        return None
    shard = f"sav_{_video_number(video_id) // 1000:03d}"
    candidate = Path(sav_root) / "sav_train" / shard / f"{video_id}_manual.json"
    return candidate if candidate.is_file() else None


class SAVManifestJSONRawDataset:
    """Expose sampled SA-V frames and nested manual JSONs to SAM2 VOSDataset."""

    def __init__(
        self,
        manifest: str,
        split: str = "train",
        verify_paths: bool = True,
        max_videos: int = 0,
        sav_root: str | Path | None = None,
    ):
        from training.dataset.vos_raw_dataset import VOSFrame, VOSVideo
        from training.dataset.vos_segment_loader import JSONSegmentLoader

        self._vos_frame_cls = VOSFrame
        self._vos_video_cls = VOSVideo
        self._segment_loader_cls = JSONSegmentLoader
        frame = pd.read_parquet(
            manifest,
            columns=["video_id", "frame_idx_24fps", "image_path", "annotation_path", "split"],
        )
        frame = frame[frame["split"] == split].copy()
        if frame.empty:
            raise ValueError(f"No rows found for split {split!r} in {manifest}")
        frame = frame.sort_values(["video_id", "frame_idx_24fps"])
        self.records = []
        missing_annotation_video_ids = []
        for video_id, rows in frame.groupby("video_id", sort=True):
            annotation_values = [
                value
                for value in rows["annotation_path"].tolist()
                if isinstance(value, str) and value.strip()
            ]
            annotation_value = annotation_values[0] if annotation_values else None
            annotation = resolve_sav_train_annotation_path(
                str(video_id), annotation_value, sav_root
            )
            if annotation is None:
                missing_annotation_video_ids.append(str(video_id))
                continue
            images = [Path(path) for path in rows["image_path"].astype(str)]
            if verify_paths:
                missing = [str(path) for path in images if not path.is_file()]
                if missing:
                    raise FileNotFoundError(
                        f"Missing SA-V task inputs for {video_id}: {missing[:5]}"
                    )
            self.records.append((str(video_id), annotation, images))
            if max_videos > 0 and len(self.records) >= max_videos:
                break
        self.missing_annotation_video_ids = missing_annotation_video_ids
        if not self.records:
            raise FileNotFoundError(
                "No SA-V task videos have readable manual annotations; "
                f"check SAV_ROOT={sav_root!s}"
            )
        if missing_annotation_video_ids and int(os.environ.get("RANK", "0")) == 0:
            logging.warning(
                "Excluded %d SA-V train videos without readable manual JSON; "
                "examples: %s",
                len(missing_annotation_video_ids),
                missing_annotation_video_ids[:10],
            )

    def get_video(self, index: int):
        video_id, annotation_path, image_paths = self.records[index]
        segment_loader = self._segment_loader_cls(
            video_json_path=str(annotation_path),
            ann_every=4,
            frames_fps=24,
        )
        frames = [
            self._vos_frame_cls(
                int(path.stem),
                image_path=str(path),
            )
            for path in image_paths
            if int(path.stem) % segment_loader.ann_every == 0
            and int(path.stem) // segment_loader.ann_every < len(segment_loader.frame_annots)
            and (
                segment_loader.frame_annots[
                    int(path.stem) // segment_loader.ann_every
                ]
                is not None
            )
            and None
            not in segment_loader.frame_annots[
                int(path.stem) // segment_loader.ann_every
            ]
        ]
        video = self._vos_video_cls(video_id, _video_number(video_id), frames)
        return video, segment_loader

    def __len__(self) -> int:
        return len(self.records)
