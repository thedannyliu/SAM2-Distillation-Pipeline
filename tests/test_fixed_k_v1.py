from __future__ import annotations

import json
from pathlib import Path

import pytest

from sam2_distill.two_clock.schedule import fixed_refresh_trajectory
from tools.eval.summarize_two_clock_phase_sweep import summarize
from tools.train.run_sam2_task_training import fraction_checkpoint_steps


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
