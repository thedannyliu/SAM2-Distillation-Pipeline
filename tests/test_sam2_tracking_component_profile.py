import pytest

from tools.benchmark.profile_sam2_tracking_components import summarize_rows


def test_component_summary_reports_encoder_share_of_tracking_step():
    rows = [
        {
            "tracking_step_ms": 30.0,
            "image_encoder_ms": 18.0,
            "image_feature_path_ms": 20.0,
        },
        {
            "tracking_step_ms": 40.0,
            "image_encoder_ms": 22.0,
            "image_feature_path_ms": 24.0,
        },
    ]

    summary = summarize_rows(rows)

    assert summary["tracking_step_mean_ms"] == 35.0
    assert summary["image_encoder_mean_ms"] == 20.0
    assert summary["image_encoder_share_of_tracking"] == pytest.approx(40.0 / 70.0)
    assert summary["image_feature_path_share_of_tracking"] == pytest.approx(44.0 / 70.0)
    assert summary["non_image_tracking_mean_ms"] == 15.0
