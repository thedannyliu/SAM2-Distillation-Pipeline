import json
import sys

import cv2
import numpy as np
import pytest

from tools.data.prepare_sav_train_shard_benchmark import discover_assets
from tools.data.split_sav_eventsam2 import assign_roles
from tools.eval.merge_vos_rank_summaries import main as merge_main
from tools.experiments.run_eventsam2_coast_screen import main as coast_main, warp_mask


def test_eventsam2_roles_are_exact_disjoint_and_reproducible():
    videos = [f"sav_{index:06d}" for index in range(11)]
    counts = {"route_train": 7, "gate_train": 2, "selection": 1, "calibration": 1}

    first = assign_roles(videos, counts, "eventsam2-test")
    second = assign_roles(list(reversed(videos)), counts, "eventsam2-test")

    assert first == second
    assert {role: len(values) for role, values in first.items()} == counts
    assert len(set().union(*(set(values) for values in first.values()))) == len(videos)
    assert sum(len(set(left) & set(right)) for left in first.values() for right in first.values()) == len(videos)


def test_discover_assets_supports_nested_shards_and_prefers_manual(tmp_path):
    first = tmp_path / "sav_000"
    second = tmp_path / "sav_001"
    first.mkdir()
    second.mkdir()
    (first / "sav_000001.mp4").touch()
    (first / "sav_000001_auto.json").write_text("{}")
    manual = first / "sav_000001_manual.json"
    manual.write_text("{}")
    (second / "sav_001001.mp4").touch()
    (second / "sav_001001_manual.json").write_text("{}")

    videos, annotations = discover_assets(tmp_path, use_auto=True)

    assert set(videos) == {"sav_000001", "sav_001001"}
    assert annotations["sav_000001"] == manual


def test_backward_flow_warp_moves_mask_right():
    mask = np.zeros((8, 8), dtype=bool)
    mask[3, 2] = True
    backward_flow = np.zeros((8, 8, 2), dtype=np.float32)
    backward_flow[..., 0] = -1.0

    warped = warp_mask(mask, backward_flow)

    assert warped[3, 3]
    assert not warped[3, 2]


def test_merge_vos_rank_summaries_reports_single_stream_latency(tmp_path, monkeypatch):
    for rank, (seconds, frames) in enumerate(((10.0, 100), (12.0, 120))):
        payload = {
            "world_size": 2,
            "rank": rank,
            "elapsed_sec": seconds,
            "processed_frames": frames,
            "video_names": [f"video-{rank}"],
            "model_timed_frames": frames,
            "model_frame_latency_sum_ms": frames * 31.0,
            "model_frame_median_ms": 30.0 + rank,
        }
        (tmp_path / f"summary.rank{rank:03d}.json").write_text(json.dumps(payload))
    monkeypatch.setattr(
        "sys.argv", ["merge", "--run-dir", str(tmp_path)]
    )

    merge_main()

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["single_stream_wall_ms_per_frame"] == 100.0
    assert summary["parallel_throughput_ms_per_frame"] == pytest.approx(12.0 / 220 * 1000.0)
    assert summary["single_stream_model_mean_ms"] == 31.0


def test_coast_screen_writes_fixed_and_safe_horizon_results(tmp_path, monkeypatch):
    image_root = tmp_path / "images" / "video"
    gt_root = tmp_path / "gt" / "video" / "001"
    pred_root = tmp_path / "pred" / "video" / "001"
    out_root = tmp_path / "out"
    image_root.mkdir(parents=True)
    gt_root.mkdir(parents=True)
    pred_root.mkdir(parents=True)
    for frame_index in range(6):
        image = np.zeros((48, 48, 3), dtype=np.uint8)
        image[16:28, 8 + frame_index : 20 + frame_index] = 255
        mask = np.zeros((48, 48), dtype=np.uint8)
        mask[16:28, 8 + frame_index : 20 + frame_index] = 255
        cv2.imwrite(str(image_root / f"{frame_index:05d}.jpg"), image)
        cv2.imwrite(str(pred_root / f"{frame_index:05d}.png"), mask)
        if frame_index in {0, 4}:
            cv2.imwrite(str(gt_root / f"{frame_index:05d}.png"), mask)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coast",
            "--image-root",
            str(tmp_path / "images"),
            "--gt-root",
            str(tmp_path / "gt"),
            "--full-pred-root",
            str(tmp_path / "pred"),
            "--out-dir",
            str(out_root),
            "--sentinel-resolution",
            "32",
            "--refresh-intervals",
            "1,2,4",
            "--safe-horizons",
            "4",
            "--full-latency-ms",
            "30",
            "--wandb-mode",
            "disabled",
        ],
    )

    coast_main()

    summary = json.loads((out_root / "summary.json").read_text())
    assert summary["videos"] == 1
    assert {row["refresh_interval"] for row in summary["fixed_policy_table"]} == {1, 2, 4}
    assert summary["safe_horizon_table"][0]["horizon"] == 4
