from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sam2_distill.two_clock.verification import (
    validate_clip_trace,
    validate_optimizer_audit,
)
from tools.train.verify_two_clock_experiment import (
    legacy_artifacts,
    parse_settings,
    same_identity,
    validate_smoke_artifacts,
)


def test_evaluation_transition_keeps_training_inputs_immutable() -> None:
    source = Path("tools/train/verify_two_clock_experiment.py").read_text(
        encoding="utf-8"
    )
    transition = source.split("def check_evaluation_transition(", maxsplit=1)[1]
    transition = transition.split("def issue_remote_stamp(", maxsplit=1)[0]
    assert '"contract", "training_config", "inputs", "sam2_repository"' in transition
    assert '"merge-base"' in transition
    assert '"--is-ancestor"' in transition
    assert "validate_optimizer_audit(optimizer)" in transition
    assert '"LEGACY_EVALUATION_PROVISIONAL"' in transition
    assert '"optimizer_membership_audit"' in transition


def _trace(experiment: str = "E") -> dict:
    actions = [True, False, False, False, True, False, False, True]
    sources = [100, 100, 100, 100, 104, 104, 104, 107]
    frames = []
    for stage, (refresh, source) in enumerate(zip(actions, sources)):
        raw = 100 + stage
        frames.append(
            {
                "stage": stage,
                "raw_frame": raw,
                "action": "REFRESH" if refresh else "REUSE",
                "source_frame": source,
                "age": raw - source,
                "encoder_called": refresh,
                "spatial_write": refresh if experiment in {"D", "E"} else True,
                "memory_write_feature": "raw",
                "pointer_write": True,
            }
        )
    return {
        "experiment": experiment,
        "encoder_calls": sum(actions),
        "teacher_frozen": True,
        "teacher_registered": False,
        "frames": frames,
    }


def test_validate_clip_trace_accepts_registered_schedule() -> None:
    validate_clip_trace(_trace())


def test_validate_clip_trace_rejects_encoder_mismatch() -> None:
    trace = _trace()
    trace["encoder_calls"] += 1
    with pytest.raises(ValueError, match="encoder-call"):
        validate_clip_trace(trace)


def test_validate_clip_trace_rejects_stale_spatial_write() -> None:
    trace = _trace()
    trace["frames"][1]["spatial_write"] = True
    with pytest.raises(ValueError, match="spatial memory"):
        validate_clip_trace(trace)


def test_validate_clip_trace_rejects_missing_pointer() -> None:
    trace = _trace()
    trace["frames"][2]["pointer_write"] = False
    with pytest.raises(ValueError, match="object pointer"):
        validate_clip_trace(trace)


def test_validate_clip_trace_rejects_registered_teacher() -> None:
    trace = _trace()
    trace["teacher_registered"] = True
    with pytest.raises(ValueError, match="module graph"):
        validate_clip_trace(trace)


def test_non_safe_write_variant_writes_every_frame() -> None:
    validate_clip_trace(_trace("C"))
    trace = _trace("C")
    trace["frames"][1]["spatial_write"] = False
    with pytest.raises(ValueError, match="skipped spatial memory"):
        validate_clip_trace(trace)


def test_validate_optimizer_audit_rejects_missing_parameters() -> None:
    audit = {
        "status": "pass",
        "missing_trainable_parameters": [],
        "unexpected_optimizer_parameters": 0,
        "teacher": {
            "parameters": 10,
            "frozen": True,
            "registered_with_student": False,
            "optimizer_parameters": 0,
        },
    }
    validate_optimizer_audit(audit)
    broken = deepcopy(audit)
    broken["status"] = "fail"
    broken["missing_trainable_parameters"] = ["age_conditioner.embedding.weight"]
    with pytest.raises(ValueError, match="status"):
        validate_optimizer_audit(broken)


def test_identity_comparison_rejects_changed_input() -> None:
    expected = {"repository": {"commit": "abc"}, "config": "one"}
    actual = {"repository": {"commit": "def"}, "config": "one"}
    with pytest.raises(RuntimeError, match="does not match"):
        same_identity(expected, actual)


def test_runtime_settings_reject_duplicate_keys() -> None:
    assert parse_settings(["epochs=5", "video_ids_file="]) == {
        "epochs": "5",
        "video_ids_file": "",
    }
    with pytest.raises(ValueError, match="duplicate"):
        parse_settings(["epochs=5", "epochs=1"])


def test_legacy_run_detection_finds_unstamped_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("legacy\n", encoding="utf-8")
    assert legacy_artifacts(tmp_path) == [checkpoint]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _fake_smoke(root: Path, world_size: int = 4) -> None:
    for path in (
        root / "checkpoints/checkpoint.pt",
        root / "checkpoints/last.pt",
        root / "resolved_config.yaml",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    _write_json(root / "training_status.json", {"status": "complete"})
    _write_json(
        root / "gradient_diagnostics.json",
        {"status": "pass", "nonfinite_steps": 0},
    )
    _write_json(
        root / "optimizer_audit.json",
        {
            "status": "pass",
            "missing_trainable_parameters": [],
            "unexpected_optimizer_parameters": 0,
            "teacher": {
                "parameters": 10,
                "frozen": True,
                "registered_with_student": False,
                "optimizer_parameters": 0,
            },
        },
    )
    for rank in range(world_size):
        _write_json(
            root / f"capacity_rank{rank}.json",
            {"world_size": world_size, "per_gpu_batch": 1},
        )
        _write_json(
            root / f"mechanism_gradients_rank{rank}.json",
            {
                "status": "pass",
                "groups": {
                    "age_conditioner": {
                        "steps_with_nonzero_gradient": int(rank == 0)
                    },
                    "memory_time_conditioner": {
                        "steps_with_nonzero_gradient": int(rank == 1)
                    },
                },
            },
        )
        (root / f"flight_recorder_rank{rank}.jsonl").write_text(
            json.dumps(_trace()) + "\n", encoding="utf-8"
        )


def test_smoke_artifact_gate_accepts_complete_evidence(tmp_path: Path) -> None:
    _fake_smoke(tmp_path)
    evidence = validate_smoke_artifacts(tmp_path, 4)
    assert evidence["flight_traces"] == 4


def test_smoke_artifact_gate_catches_optimizer_mutation(tmp_path: Path) -> None:
    _fake_smoke(tmp_path)
    optimizer = json.loads(
        (tmp_path / "optimizer_audit.json").read_text(encoding="utf-8")
    )
    optimizer["status"] = "fail"
    optimizer["missing_trainable_parameters"] = ["age_conditioner.embedding.weight"]
    _write_json(tmp_path / "optimizer_audit.json", optimizer)
    with pytest.raises(ValueError, match="status"):
        validate_smoke_artifacts(tmp_path, 4)
