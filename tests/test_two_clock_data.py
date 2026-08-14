from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sam2_distill.two_clock.data import (
    ContiguousSAV24FPSSampler,
    SAVRawFrame,
    SAVRawVideo,
    SAVSparseSegmentLoader,
    decode_contiguous_rgb,
)


def _annotation(path: Path) -> None:
    payload = {
        "fps": 6,
        "masklet": [
            [{"size": [4, 4], "counts": "fake"}, None],
            [None, {"size": [4, 4], "counts": "fake"}],
            [{"size": [4, 4], "counts": "fake"}, {"size": [4, 4], "counts": "fake"}],
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_sparse_loader_preserves_manual_validity(tmp_path: Path) -> None:
    annotation = tmp_path / "manual.json"
    _annotation(annotation)
    loader = SAVSparseSegmentLoader(annotation)
    assert loader.ann_every == 4
    assert loader.is_annotated(0, 0)
    assert not loader.is_annotated(0, 1)
    assert not loader.is_annotated(1, 0)
    assert loader.visible_object_ids(4) == [1]
    assert loader.valid_anchor_frames(raw_frame_count=12, clip_length=8) == [0, 4]


def test_sampler_is_contiguous_deterministic_and_records_raw_time(tmp_path: Path) -> None:
    annotation = tmp_path / "manual.json"
    _annotation(annotation)
    loader = SAVSparseSegmentLoader(annotation)
    video = SAVRawVideo(
        "sav_000123",
        123,
        [SAVRawFrame(index, "/tmp/fake.mp4") for index in range(12)],
    )
    sampler = ContiguousSAV24FPSSampler(num_frames=8, max_num_objects=2)
    first = sampler.sample(video, loader, epoch=2)
    second = sampler.sample(video, loader, epoch=2)
    assert [frame.frame_idx for frame in first.frames] == [
        frame.frame_idx for frame in second.frames
    ]
    assert [frame.frame_idx for frame in first.frames] == list(
        range(first.frames[0].frame_idx, first.frames[0].frame_idx + 8)
    )
    for frame in first.frames:
        assert frame.frame_idx - frame.two_clock_source_raw == frame.two_clock_age
        assert 0 <= frame.two_clock_age <= 5


def test_force_full_refresh_metadata(tmp_path: Path) -> None:
    annotation = tmp_path / "manual.json"
    _annotation(annotation)
    loader = SAVSparseSegmentLoader(annotation)
    video = SAVRawVideo(
        "sav_000123",
        123,
        [SAVRawFrame(index, "/tmp/fake.mp4") for index in range(12)],
    )
    sample = ContiguousSAV24FPSSampler(
        num_frames=8, force_full_refresh=True
    ).sample(video, loader)
    assert all(frame.two_clock_refresh for frame in sample.frames)
    assert all(frame.two_clock_age == 0 for frame in sample.frames)


def test_decode_contiguous_mp4_reads_requested_frames_once(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (16, 16)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 writer unavailable")
    for value in range(8):
        writer.write(np.full((16, 16, 3), value * 20, dtype=np.uint8))
    writer.release()
    images = decode_contiguous_rgb(path, [2, 3, 4])
    assert len(images) == 3
    means = [np.asarray(image).mean() for image in images]
    assert means[0] < means[1] < means[2]
    with pytest.raises(ValueError, match="contiguous"):
        decode_contiguous_rgb(path, [0, 2])
