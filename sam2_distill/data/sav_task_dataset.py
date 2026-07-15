"""SA-V task-training adapter backed by the mounted Stage 1 manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _video_number(video_id: str) -> int:
    return int(video_id.rsplit("_", 1)[-1])


class SAVManifestJSONRawDataset:
    """Expose sampled SA-V frames and nested manual JSONs to SAM2 VOSDataset."""

    def __init__(
        self,
        manifest: str,
        split: str = "train",
        verify_paths: bool = True,
        max_videos: int = 0,
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
        if max_videos > 0:
            video_ids = frame["video_id"].drop_duplicates().head(max_videos)
            frame = frame[frame["video_id"].isin(video_ids)]
        self.records = []
        for video_id, rows in frame.groupby("video_id", sort=True):
            annotation_paths = rows["annotation_path"].dropna().astype(str).unique()
            if len(annotation_paths) != 1:
                raise ValueError(
                    f"{video_id} must have exactly one annotation path; "
                    f"got {annotation_paths.tolist()}"
                )
            images = [Path(path) for path in rows["image_path"].astype(str)]
            annotation = Path(annotation_paths[0])
            if verify_paths:
                missing = [str(path) for path in [annotation, *images] if not path.is_file()]
                if missing:
                    raise FileNotFoundError(
                        f"Missing SA-V task inputs for {video_id}: {missing[:5]}"
                    )
            self.records.append((str(video_id), annotation, images))

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
