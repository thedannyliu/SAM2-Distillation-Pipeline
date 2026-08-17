import numpy as np
import pytest
import torch
from PIL import Image

from tools.experiments.analyze_sam21l_embedding_drift import (
    adjacent_embedding_metrics,
    extract_video_rows,
    stratified_video_selection,
)


def test_adjacent_embedding_metrics_identical_features():
    feature = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]])

    cosine, mse, nmse = adjacent_embedding_metrics(feature, feature.clone())

    assert cosine.item() == pytest.approx(1.0)
    assert mse.item() == pytest.approx(0.0)
    assert nmse.item() == pytest.approx(0.0)


def test_adjacent_embedding_metrics_uses_symmetric_energy_normalization():
    previous = torch.tensor([[[[1.0, 1.0]]]])
    current = torch.tensor([[[[2.0, 2.0]]]])

    cosine, mse, nmse = adjacent_embedding_metrics(current, previous)

    assert cosine.item() == pytest.approx(1.0)
    assert mse.item() == pytest.approx(1.0)
    assert nmse.item() == pytest.approx(0.4)


def test_stratified_selection_is_deterministic_and_covers_four_length_strata():
    candidates = [(f"video_{index:02d}", 10 + index) for index in range(40)]

    first = stratified_video_selection(candidates, num_videos=20, seed=7)
    second = stratified_video_selection(candidates, num_videos=20, seed=7)

    assert first == second
    assert len({row["video"] for row in first}) == 20
    assert {row["length_stratum"] for row in first} == {1, 2, 3, 4}
    assert all(
        sum(row["length_stratum"] == stratum for row in first) == 5
        for stratum in (1, 2, 3, 4)
    )
    lengths_by_stratum = {
        stratum: [row["frame_count"] for row in first if row["length_stratum"] == stratum]
        for stratum in (1, 2, 3, 4)
    }
    assert max(lengths_by_stratum[1]) < min(lengths_by_stratum[2])
    assert max(lengths_by_stratum[2]) < min(lengths_by_stratum[3])
    assert max(lengths_by_stratum[3]) < min(lengths_by_stratum[4])


def test_extract_video_rows_keeps_all_cross_batch_transitions(tmp_path):
    paths = []
    for index in range(5):
        path = tmp_path / f"{index:06d}.png"
        Image.fromarray(np.full((2, 2, 3), index + 1, dtype=np.uint8)).save(path)
        paths.append(path)

    class FakePredictor:
        def set_image_batch(self, images):
            self._features = {
                "image_embed": torch.stack(
                    [torch.full((2, 2, 2), float(image[0, 0, 0])) for image in images]
                )
            }

    rows = extract_video_rows(
        FakePredictor(),
        "video",
        paths,
        batch_size=2,
        fps=24.0,
        device=torch.device("cpu"),
        amp_dtype="none",
    )

    assert len(rows) == 4
    assert [row["transition_index"] for row in rows] == [1, 2, 3, 4]
    assert [(row["previous_frame"], row["current_frame"]) for row in rows] == [
        ("000000.png", "000001.png"),
        ("000001.png", "000002.png"),
        ("000002.png", "000003.png"),
        ("000003.png", "000004.png"),
    ]
