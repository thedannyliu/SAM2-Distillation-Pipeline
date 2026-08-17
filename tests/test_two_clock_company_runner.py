from __future__ import annotations

from pathlib import Path

from tools.eval.run_edgetam_vos_dataset import (
    configure_video_storage,
    normalize_sam2_config,
)


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


def test_selection_and_curves_verify_completed_training_for_evaluation() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (
        repo_root / "scripts/company/75_run_sam21l_two_clock_reuse_v1.sh"
    ).read_text(encoding="utf-8")
    assert runner.count('check_eval "${experiment}" "${run_dir}"') == 4
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
    assert "Resume completed inference; compute only missing age metrics" in evaluate


def test_age_evaluator_adds_repository_import_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    evaluator = (repo_root / "tools/eval/evaluate_two_clock_age.py").read_text(
        encoding="utf-8"
    )
    assert 'REPO_ROOT = Path(__file__).resolve().parents[2]' in evaluator
    assert 'sys.path.insert(0, str(REPO_ROOT))' in evaluator


def test_report_requires_complete_jf_and_latency_matrix() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (
        repo_root / "scripts/company/75_run_sam21l_two_clock_reuse_v1.sh"
    ).read_text(encoding="utf-8")
    report = runner.split("  report()", maxsplit=1)[1].split(
        "  status()", maxsplit=1
    )[0]
    assert "summarize_two_clock_validation.py" in report
    assert "--require-complete" in report


def test_vos_eval_offloads_video_frames_but_keeps_explicit_override() -> None:
    class Predictor:
        def __init__(self):
            self.calls = []

        def init_state(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return kwargs

    predictor = Predictor()
    configure_video_storage(predictor, offload_video_to_cpu=True)

    default = predictor.init_state(video_path="video")
    explicit = predictor.init_state(video_path="video", offload_video_to_cpu=False)

    assert default["offload_video_to_cpu"] is True
    assert explicit["offload_video_to_cpu"] is False


def test_official_builder_uses_hydra_package_relative_sam2_config() -> None:
    root = Path("/repo/facebookresearch-sam2")
    absolute = root / "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    assert normalize_sam2_config(root, str(absolute)) == (
        "configs/sam2.1/sam2.1_hiera_l.yaml"
    )


def test_partial_report_does_not_require_complete_matrix() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (
        repo_root / "scripts/company/75_run_sam21l_two_clock_reuse_v1.sh"
    ).read_text(encoding="utf-8")
    partial = runner.split("  report_partial()", maxsplit=1)[1].split(
        "  status()", maxsplit=1
    )[0]
    assert "summarize_two_clock_validation.py" in partial
    assert "--require-complete" not in partial


def test_quick_val_is_fixed_ten_video_epoch_five_r4_screen() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (
        repo_root / "scripts/company/75_run_sam21l_two_clock_reuse_v1.sh"
    ).read_text(encoding="utf-8")
    assert 'local count=10' in runner
    assert 'gt_10_seed250107256/Annotations_6fps' in runner
    assert 'local experiment="$1" epoch="${QUICK_EPOCH:-5}"' in runner
    quick = runner.split("  quick_val()", maxsplit=1)[1].split(
        "  quick_controls()", maxsplit=1
    )[0]
    assert '4 "${run_dir}/quick_val/epoch_${epoch}/R4"' in quick


def test_screen50_jobs_are_single_gpu_and_use_distinct_control_predictors() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (
        repo_root / "scripts/company/75_run_sam21l_two_clock_reuse_v1.sh"
    ).read_text(encoding="utf-8")
    screen = runner.split("  require_single_gpu()", maxsplit=1)[1].split(
        "  select_checkpoint()", maxsplit=1
    )[0]
    assert 'make_hash_cohort 50 screen50' in runner
    assert 'if [[ "${gpu_count}" -ne 1 ]]' in screen
    assert '"${gt_view}" official' in screen
    assert '"${gt_view}" two-clock' in screen
    assert '"${gt_view}" official-reuse' in screen


def test_evaluation_resume_validates_predictor_identity() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (
        repo_root / "scripts/company/75_run_sam21l_two_clock_reuse_v1.sh"
    ).read_text(encoding="utf-8")
    evaluate = runner.split("evaluate_checkpoint()", maxsplit=1)[1].split(
        "make_hash_cohort()", maxsplit=1
    )[0]
    assert 'timing.get("model_kind") == sys.argv[6]' in evaluate
    assert 'all(row.get("model_kind") == model_kind for row in ranks)' in evaluate


def test_official_reuse_predictor_restores_official_tracking_and_memory_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    predictor = (repo_root / "sam2_distill/two_clock/predictor.py").read_text(
        encoding="utf-8"
    )
    official_reuse = predictor.split(
        "class OfficialReuseVideoPredictor", maxsplit=1
    )[1].split("def _build_video_predictor", maxsplit=1)[0]
    assert "track_step = SAM2VideoPredictor.track_step" in official_reuse
    assert "SAM2VideoPredictor._prepare_memory_conditioned_features" in official_reuse
