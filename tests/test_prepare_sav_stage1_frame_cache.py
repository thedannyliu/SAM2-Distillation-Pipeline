from pathlib import Path

from tools.data.prepare_sav_stage1_frame_cache import (
    choose_annotation,
    detect_ann_root,
    discover_split,
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
