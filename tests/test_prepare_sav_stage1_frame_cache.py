import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from tools.data.prepare_sav_stage1_frame_cache import (
    choose_annotation,
    detect_ann_root,
    discover_split,
    extract_video_task,
    task_shard_index,
)


def test_sharded_sav_annotation_is_discovered(tmp_path: Path) -> None:
    video_id = "sav_024246"
    annotation = tmp_path / "sav_024" / f"{video_id}_manual.json"
    annotation.parent.mkdir()
    annotation.write_text('{"masklet": []}\n', encoding="utf-8")

    ann_root = detect_ann_root(tmp_path)

    assert ann_root == tmp_path
    assert choose_annotation(ann_root, video_id, use_auto=False) == annotation


def test_manual_annotation_is_preferred_over_auto(tmp_path: Path) -> None:
    video_id = "sav_001234"
    shard = tmp_path / "sav_001"
    shard.mkdir()
    manual = shard / f"{video_id}_manual.json"
    auto = shard / f"{video_id}_auto.json"
    manual.write_text("{}\n", encoding="utf-8")
    auto.write_text("{}\n", encoding="utf-8")

    assert choose_annotation(tmp_path, video_id, use_auto=True) == manual


def test_sharded_release_uses_annotation_length_without_video_decode(
    tmp_path: Path,
) -> None:
    video_id = "sav_024246"
    shard = tmp_path / "sav_024"
    shard.mkdir()
    (shard / f"{video_id}.mp4").touch()
    annotation = shard / f"{video_id}_manual.json"
    annotation.write_text(
        '{"masklet": [[], [], [], [], [], []]}\n',
        encoding="utf-8",
    )

    tasks = discover_split(
        "train",
        tmp_path,
        frames_per_video=1000000,
        max_videos=0,
        seed="test",
        use_auto=False,
        ann_every=4,
    )

    assert len(tasks) == 1
    assert tasks[0]["annotation_path"] == str(annotation)
    assert tasks[0]["indices_6fps"] == list(range(6))


def test_video_tasks_are_deterministically_partitioned() -> None:
    tasks = [
        {"split": "train", "video_id": f"sav_{index:06d}"}
        for index in range(100)
    ]
    shards = [
        {
            task["video_id"]
            for task in tasks
            if task_shard_index(task, 4) == shard_index
        }
        for shard_index in range(4)
    ]

    assert all(shards)
    assert sum(len(shard) for shard in shards) == len(tasks)
    assert set().union(*shards) == {task["video_id"] for task in tasks}


def test_corrupt_cached_jpeg_is_rebuilt_atomically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_id = "sav_038326"
    cached = tmp_path / "JPEGImages" / video_id / "00160.jpg"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"incomplete jpeg")

    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def set(self, _property, _value) -> None:
            pass

        def read(self):
            return True, np.zeros((12, 16, 3), dtype=np.uint8)

        def release(self) -> None:
            pass

    fake_cv2 = SimpleNamespace(
        CAP_PROP_POS_FRAMES=1,
        COLOR_BGR2RGB=2,
        VideoCapture=lambda _path: FakeCapture(),
        cvtColor=lambda frame, _conversion: frame,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    rows = extract_video_task(
        {
            "kind": "raw",
            "split": "train",
            "video_id": video_id,
            "video_path": str(tmp_path / f"{video_id}.mp4"),
            "annotation_path": "",
            "indices_6fps": [40],
        },
        str(tmp_path),
        ann_every=4,
        jpeg_quality=90,
        skip_existing=True,
    )

    with Image.open(cached) as image:
        image.verify()
    assert rows[0]["width"] == 16
    assert rows[0]["height"] == 12
    assert not list(cached.parent.glob(".00160.jpg.*.tmp"))
