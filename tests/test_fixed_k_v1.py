from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from sam2_distill.two_clock.schedule import fixed_refresh_trajectory
from tools.eval.merge_vos_rank_summaries import main as merge_rank_summaries
from tools.eval.run_two_clock_eval30 import Target, _prediction_complete
from tools.eval.summarize_two_clock_phase_sweep import summarize
from tools.train.run_sam2_task_training import (
    fraction_checkpoint_steps,
    trainable_parameter_names,
)


def test_k20_trajectory_is_causal_and_reaches_age_19() -> None:
    trajectory = fixed_refresh_trajectory(21, 20, max_age=19)
    assert trajectory.source == (0,) * 20 + (20,)
    assert trajectory.age == tuple(range(20)) + (0,)
    assert trajectory.encoder_calls == 2


def test_fraction_checkpoint_steps_round_up_to_completed_updates() -> None:
    assert fraction_checkpoint_steps("0.10,0.25,0.50", 101) == {
        11: 0.10,
        26: 0.25,
        51: 0.50,
    }
    with pytest.raises(ValueError, match="inside"):
        fraction_checkpoint_steps("1.0", 100)


def test_optimizer_allowlist_excludes_frozen_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    for parameter in model[1].parameters():
        parameter.requires_grad_(False)

    assert trainable_parameter_names(model) == {"0.weight", "0.bias"}


def _write_unit(
    root: Path,
    model: str,
    interval: int,
    phase: int,
    values: dict[str, float],
) -> None:
    unit = root / model / f"R{interval}" / f"phase_{phase}"
    (unit / "pred").mkdir(parents=True)
    per_video = [
        {"video": video, "age": phase, "count": 2, "J": value, "F": value, "J&F": value}
        for video, value in values.items()
    ]
    (unit / "age_metrics.json").write_text(
        json.dumps({"status": "pass", "per_video": per_video})
    )
    mean = sum(values.values()) / len(values)
    (unit / "sav_eval.json").write_text(
        json.dumps({"status": "pass", "metrics": {"J": mean, "F": mean, "J&F": mean}})
    )
    (unit / "pred/summary.json").write_text(
        json.dumps(
            {
                "two_clock_refresh_rate": 1 / interval,
                "model_frame_mean_ms": 10.0,
                "model_frame_p95_ms": 12.0,
                "wall_ms_per_frame": 20.0,
            }
        )
    )


def test_phase_sweep_uses_per_video_phase_neutral_macro(tmp_path: Path) -> None:
    _write_unit(tmp_path, "O0", 1, 0, {"v1": 10.0, "v2": 30.0})
    for phase in range(4):
        _write_unit(tmp_path, "O1", 4, phase, {"v1": 20.0, "v2": 40.0})
        _write_unit(tmp_path, "A", 4, phase, {"v1": 25.0, "v2": 45.0})

    payload = summarize(
        tmp_path,
        models=("O0", "O1", "A"),
        bootstrap_samples=100,
    )
    rows = {row["model"]: row for row in payload["models"]}
    assert rows["O0"]["J&F"] == pytest.approx(20.0)
    assert rows["O1"]["J&F"] == pytest.approx(30.0)
    assert rows["A"]["J&F"] == pytest.approx(35.0)
    assert payload["paired_bootstrap"]["A"]["O1"]["delta_J&F"] == pytest.approx(5.0)


def test_phase_sweep_supports_fixed_k_model_intervals(tmp_path: Path) -> None:
    _write_unit(tmp_path, "O0", 1, 0, {"v1": 10.0, "v2": 30.0})
    for phase in range(8):
        _write_unit(tmp_path, "O1", 8, phase, {"v1": 20.0, "v2": 40.0})
        _write_unit(tmp_path, "A8", 8, phase, {"v1": 25.0, "v2": 45.0})

    payload = summarize(
        tmp_path,
        models=("O0", "O1", "A8"),
        intervals={"O0": 1, "O1": 8, "A8": 8},
        bootstrap_samples=100,
    )
    rows = {row["model"]: row for row in payload["models"]}
    assert rows["A8"]["interval"] == 8
    assert rows["A8"]["J&F"] == pytest.approx(35.0)
    assert payload["paired_bootstrap"]["A8"]["O1"]["delta_J&F"] == pytest.approx(5.0)


def test_rank_merge_preserves_fixed_phase_resume_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for rank in range(2):
        payload = {
            "world_size": 2,
            "rank": rank,
            "video_names": [f"video{rank}"],
            "processed_frames": 10,
            "elapsed_sec": 2.0,
            "model_timed_frames": 8,
            "model_frame_latency_sum_ms": 80.0,
            "model_frame_median_ms": 10.0,
            "model_frame_p95_ms": 12.0,
            "num_prediction_pngs": 5,
            "model_kind": "two-clock",
            "build": {
                "experiment": "A",
                "refresh_interval": 8,
                "checkpoint": "/runs/A8/checkpoints/last.pt",
            },
            "two_clock_encoder_calls": 2,
            "two_clock_tracking_frames": 16,
            "two_clock_refresh_phase_mode": "fixed",
            "two_clock_refresh_phase": 3,
            "two_clock_refresh_phase_seed": 250107256,
            "offload_video_to_cpu": True,
        }
        (tmp_path / f"summary.rank{rank:03d}.json").write_text(
            json.dumps(payload)
        )

    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["merge", "--run-dir", str(tmp_path), "--out", str(output)],
    )
    merge_rank_summaries()
    merged = json.loads(output.read_text())
    assert merged["build"]["checkpoint"] == "/runs/A8/checkpoints/last.pt"
    assert merged["two_clock_refresh_phase"] == 3
    assert merged["video_names"] == ["video0", "video1"]
    assert merged["num_prediction_pngs"] == 10


def test_company_runner_has_six_fixed_k_targets_and_sequential_eval() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts/company/76_run_sam21l_fixed_k_v1.sh"
    ).read_text(encoding="utf-8")
    for target in ("A1", "A4", "A8", "A12", "A16", "A20"):
        assert f"{target})" in runner
    assert "TASK_FRACTION_CHECKPOINTS=" in runner
    assert "run_two_clock_eval30.py" in runner
    assert 'if [[ "${gpu_count}" -ne 1 ]]' in runner
    assert 'run_dir="${run_root}/smoke/${git_sha}/${target}"' in runner
    assert "run_fixed_k_eval30.py" in runner
    assert "eval-fixed) eval_fixed" in runner


def test_eval_resume_rejects_wrong_phase_or_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    target = Target("A", "two-clock", "A", 4, checkpoint, None)
    unit = tmp_path / "unit"
    pred = unit / "pred/video/object"
    pred.mkdir(parents=True)
    (pred / "000000.png").touch()
    summary = {
        "status": "pass",
        "model_kind": "two-clock",
        "two_clock_refresh_phase_mode": "fixed",
        "two_clock_refresh_phase": 2,
        "video_names": ["video"],
        "num_prediction_pngs": 1,
        "build": {
            "checkpoint": str(checkpoint),
            "experiment": "A",
            "refresh_interval": 4,
        },
    }
    (unit / "pred/summary.json").write_text(json.dumps(summary))
    assert _prediction_complete(unit, target, 2, ["video"])
    assert not _prediction_complete(unit, target, 1, ["video"])
    summary["build"]["checkpoint"] = str(tmp_path / "other.pt")
    (unit / "pred/summary.json").write_text(json.dumps(summary))
    assert not _prediction_complete(unit, target, 2, ["video"])
