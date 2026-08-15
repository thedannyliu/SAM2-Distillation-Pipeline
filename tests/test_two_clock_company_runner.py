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
