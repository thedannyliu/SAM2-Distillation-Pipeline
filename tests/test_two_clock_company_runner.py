from __future__ import annotations

from pathlib import Path


def test_company_runner_exports_sam2_root_to_training_process() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (
        repo_root / "scripts/company/75_run_sam21l_two_clock_reuse_v1.sh"
    ).read_text(encoding="utf-8")
    launch = runner.split('echo "Run directory: ${run_dir}"', maxsplit=1)[1]
    launch = launch.split("torchrun --standalone", maxsplit=1)[0]
    assert 'SAM2_TRAINING_ROOT="${sam2_root}"' in launch


def test_training_entrypoint_suppresses_verbose_module_match_listing() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    entrypoint = (
        repo_root / "tools/train/run_sam2_task_training.py"
    ).read_text(encoding="utf-8")
    assert "def quiet_module_cls_pattern_match(" in entrypoint
    assert (
        "optimizer_module.unix_module_cls_pattern_to_parameter_names = ("
        in entrypoint
    )


def test_formal_training_records_verified_launch_before_torchrun() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (
        repo_root / "scripts/company/75_run_sam21l_two_clock_reuse_v1.sh"
    ).read_text(encoding="utf-8")
    train = runner.split("train_target()", maxsplit=1)[1]
    train = train.split("verify_smoke()", maxsplit=1)[0]
    assert train.index("    record_launch \\") < train.index("torchrun --standalone")
    assert "TASK_TWO_CLOCK_FLIGHT_RECORDER=" in train


def test_verification_smoke_is_namespaced_by_git_sha() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (
        repo_root / "scripts/company/75_run_sam21l_two_clock_reuse_v1.sh"
    ).read_text(encoding="utf-8")
    assert '"${verification_dir}/smoke/${git_sha}/$1"' in runner
    assert 'if [[ "${target}" != "E" ]]' in runner


def test_raw_cache_and_conditioned_read_wiring_remain_separate() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model = (repo_root / "sam2_distill/two_clock/model.py").read_text(
        encoding="utf-8"
    )
    track_step = model.split("def _two_clock_track_step(", maxsplit=1)[1]
    track_step = track_step.split(
        "def _prepare_memory_conditioned_features(", maxsplit=1
    )[0]
    assert "read_vision_feats[-1:]" in track_step
    assert "raw_vision_feats,\n            feat_sizes" in track_step


def test_selection_and_curves_recheck_formal_launch_identity() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (
        repo_root / "scripts/company/75_run_sam21l_two_clock_reuse_v1.sh"
    ).read_text(encoding="utf-8")
    assert runner.count('check_launch "${experiment}" "${run_dir}"') == 2
    run_action = runner.split("    run)", maxsplit=1)[1].split("      ;;", maxsplit=1)[0]
    assert 'postrun_audit "${target}"' in run_action


def test_validation_uses_paired_balanced_refresh_phases() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (
        repo_root / "scripts/company/75_run_sam21l_two_clock_reuse_v1.sh"
    ).read_text(encoding="utf-8")
    evaluate = runner.split("evaluate_checkpoint()", maxsplit=1)[1]
    evaluate = evaluate.split("select_checkpoint()", maxsplit=1)[0]
    assert evaluate.count("--refresh-phase-mode balanced") == 2
    assert 'age.get("protocol") == "balanced_phase_v1"' in evaluate
    assert "full_sav_val_balanced_phase_R4_J&F" in runner
    val_action = runner.split("    val)", maxsplit=1)[1].split(
        "      ;;", maxsplit=1
    )[0]
    assert 'select_checkpoint "${target}"' in val_action
    assert 'curves "${target}"' in val_action
    assert 'train_target "${target}"' not in val_action
