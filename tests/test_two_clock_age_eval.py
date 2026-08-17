from __future__ import annotations

import numpy as np

from tools.eval.evaluate_two_clock_age import _single_frame_metrics


class _UnusedEvaluator:
    def __init__(self) -> None:
        raise AssertionError("empty-mask metrics must not instantiate the evaluator")


def test_both_empty_masks_are_perfect() -> None:
    empty = np.zeros((4, 4), dtype=bool)
    assert _single_frame_metrics(_UnusedEvaluator, empty, empty) == (100.0, 100.0)


def test_one_empty_mask_is_zero() -> None:
    empty = np.zeros((4, 4), dtype=bool)
    present = empty.copy()
    present[1:3, 1:3] = True
    assert _single_frame_metrics(_UnusedEvaluator, empty, present) == (0.0, 0.0)
    assert _single_frame_metrics(_UnusedEvaluator, present, empty) == (0.0, 0.0)
