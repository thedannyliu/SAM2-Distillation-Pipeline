import pytest

from tools.benchmark.profile_sam2_tracking_components import summarize_rows


def test_component_summary_reports_additive_percentages_without_double_counting():
    rows = []
    for tracking_ms, encoder_ms in ((30.0, 18.0), (40.0, 22.0)):
        row = {
            "tracking_step_ms": tracking_ms,
            "image_encoder_ms": encoder_ms,
            "prompt_encoder_ms": 1.0,
            "memory_attention_ms": 4.0,
            "mask_decoder_ms": 3.0,
            "memory_encoder_ms": 2.0,
            "object_pointer_projection_ms": 0.5,
            "object_pointer_temporal_projection_ms": 0.5,
            "image_trunk_ms": encoder_ms - 2.0,
            "image_neck_ms": 2.0,
            "mask_decoder_transformer_ms": 2.0,
            "memory_mask_downsampler_ms": 0.5,
            "memory_pixel_projection_ms": 0.25,
            "memory_fuser_ms": 1.0,
            "memory_output_projection_ms": 0.25,
            "image_feature_path_ms": encoder_ms + 2.0,
        }
        row["residual_ms"] = tracking_ms - sum(
            row[f"{component}_ms"]
            for component in (
                "image_encoder",
                "prompt_encoder",
                "memory_attention",
                "mask_decoder",
                "memory_encoder",
                "object_pointer_projection",
                "object_pointer_temporal_projection",
            )
        )
        rows.append(row)

    summary = summarize_rows(rows)

    assert summary["tracking_step_mean_ms"] == 35.0
    assert summary["component_percent_total"] == pytest.approx(100.0)
    components = {row["component"]: row for row in summary["component_breakdown"]}
    assert components["image_encoder"]["mean_ms"] == 20.0
    assert components["image_encoder"]["percent_of_tracking"] == pytest.approx(
        100.0 * 40.0 / 70.0
    )
    nested = {row["component"]: row for row in summary["nested_component_breakdown"]}
    assert nested["image_neck"]["percent_of_parent"] == pytest.approx(10.0)
    assert summary["image_feature_path_percent_of_tracking"] == pytest.approx(
        100.0 * 44.0 / 70.0
    )
