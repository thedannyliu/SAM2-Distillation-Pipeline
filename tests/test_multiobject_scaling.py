import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from tools.benchmark.benchmark_sam2_multiobject_scaling import (
    aggregate_rows,
    parse_object_counts,
)
from tools.data.audit_vos_object_density import shared_frame_prompts
from tools.data.select_sav_dense_training_videos import (
    frame_object_counts,
    main as select_dense_main,
    repeat_to_length,
)
from tools.train.summarize_mask_finetune_ablations import (
    add_multiobject_latency,
)
from sam2_distill.models.task_finetune import initialize_edgetam_memory_model


def write_mask(path: Path, nonempty: bool) -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    if nonempty:
        mask[2:6, 2:6] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def test_shared_frame_prompts_requires_nonempty_common_frame(tmp_path):
    ann_video = tmp_path / "video"
    write_mask(ann_video / "000" / "00000.png", True)
    write_mask(ann_video / "001" / "00000.png", True)
    write_mask(ann_video / "002" / "00000.png", False)
    write_mask(ann_video / "002" / "00001.png", True)

    shared = shared_frame_prompts(ann_video, min_objects=2)

    assert shared is not None
    assert shared[0] == 0
    assert [object_id for object_id, _ in shared[1]] == ["000", "001"]
    assert shared_frame_prompts(ann_video, min_objects=3) is None


def test_parse_object_counts_requires_one_object_baseline():
    assert parse_object_counts("8,1,4,4") == [1, 4, 8]
    with pytest.raises(argparse.ArgumentTypeError, match="one-object baseline"):
        parse_object_counts("2,4,8")


def test_aggregate_rows_reports_relative_latency():
    rows = [
        {
            "video": "a",
            "object_count": 1,
            "propagation_ms_per_frame": 10.0,
            "propagation_fps": 100.0,
            "end_to_end_fps": 80.0,
            "prompt_sec": 0.1,
            "peak_memory_mb": 1000.0,
        },
        {
            "video": "a",
            "object_count": 2,
            "propagation_ms_per_frame": 11.0,
            "propagation_fps": 90.9,
            "end_to_end_fps": 75.0,
            "prompt_sec": 0.2,
            "peak_memory_mb": 1100.0,
        },
    ]

    aggregate = aggregate_rows(rows, [1, 2])

    assert aggregate[0]["relative_latency_vs_1"] == pytest.approx(1.0)
    assert aggregate[1]["relative_latency_vs_1"] == pytest.approx(1.1)
    assert aggregate[1]["target_pass"] == 1


def test_dense_training_counts_visible_objects():
    payload = {
        "masklet": [
            [{"counts": "a"}, None, {"counts": "b"}],
            [None, None, None],
        ]
    }

    assert frame_object_counts(payload) == [2, 0]


def test_dense_training_repetition_is_deterministic():
    first = repeat_to_length(["a", "b", "c"], target=8, seed=7)
    second = repeat_to_length(["a", "b", "c"], target=8, seed=7)

    assert first == second
    assert len(first) == 8
    assert set(first) == {"a", "b", "c"}


def test_dense_training_cli_writes_repeated_manifest_cohort(
    tmp_path,
    monkeypatch,
):
    annotation = tmp_path / "sav_000001_manual.json"
    annotation.write_text(
        json.dumps({"masklet": [[{} for _ in range(8)] for _ in range(4)]}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(
        [
            {
                "video_id": "sav_000001",
                "annotation_path": str(annotation),
                "split": "train",
            }
        ]
    ).to_parquet(manifest)
    output = tmp_path / "dense8.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_sav_dense_training_videos.py",
            "--manifest",
            str(manifest),
            "--sav-root",
            str(tmp_path),
            "--output-video-ids",
            str(output),
            "--out-csv",
            str(tmp_path / "index.csv"),
            "--out-summary",
            str(tmp_path / "summary.json"),
            "--target-samples",
            "6",
        ],
    )

    select_dense_main()

    assert output.read_text(encoding="utf-8").splitlines() == [
        "sav_000001"
    ] * 6


def test_summary_reads_one_and_eight_object_latency(tmp_path):
    path = tmp_path / "aggregate.csv"
    path.write_text(
        "object_count,median_propagation_fps,relative_latency_vs_1,"
        "median_peak_memory_mb,target_pass\n"
        "1,100.0,1.0,900.0,\n"
        "8,40.0,2.5,1200.0,0\n",
        encoding="utf-8",
    )
    row = {}

    add_multiobject_latency(row, path)

    assert row["latency_n1_fps"] == "100.0"
    assert row["latency_n8_fps"] == "40.0"
    assert row["latency_n8_relative"] == "2.5"
    assert row["latency_gate_pass"] == "0"


def test_two_layer_memory_initializes_from_four_layer_checkpoint(tmp_path):
    source = torch.nn.Module()
    source.memory_attention = torch.nn.Module()
    source.memory_attention.layers = torch.nn.ModuleList(
        [torch.nn.Linear(2, 2, bias=False) for _ in range(4)]
    )
    target = torch.nn.Module()
    target.memory_attention = torch.nn.Module()
    target.memory_attention.layers = torch.nn.ModuleList(
        [torch.nn.Linear(2, 2, bias=False) for _ in range(2)]
    )
    with torch.no_grad():
        for index, layer in enumerate(source.memory_attention.layers):
            layer.weight.fill_(index + 1)
    checkpoint = tmp_path / "mem4.pt"
    torch.save({"model": source.state_dict()}, checkpoint)

    initialize_edgetam_memory_model(
        target,
        previous_task_checkpoint=str(checkpoint),
        memory_initializer="current",
    )

    assert torch.equal(
        target.memory_attention.layers[0].weight,
        source.memory_attention.layers[0].weight,
    )
    assert torch.equal(
        target.memory_attention.layers[1].weight,
        source.memory_attention.layers[1].weight,
    )
