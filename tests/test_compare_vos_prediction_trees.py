from __future__ import annotations

import numpy as np
from PIL import Image

from tools.eval.compare_vos_prediction_trees import compare


def _mask(path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 5), value, dtype=np.uint8)).save(path)


def test_compare_prediction_trees_checks_decoded_pixels(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _mask(left / "video/object/000.png", 1)
    _mask(right / "video/object/000.png", 1)
    assert compare(left, right)["identical"] is True

    _mask(right / "video/object/000.png", 2)
    result = compare(left, right)
    assert result["identical"] is False
    assert result["pixel_mismatch_count"] == 1
