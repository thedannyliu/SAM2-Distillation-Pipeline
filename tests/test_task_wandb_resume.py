import json
import sys
from types import SimpleNamespace

from tools.train.run_sam2_task_training import init_wandb


def test_init_wandb_allows_local_id_missing_remotely(monkeypatch, tmp_path):
    calls = []
    fake_run = SimpleNamespace(
        id="local-only-id",
        url="https://wandb.invalid/local-only-id",
        entity="test-entity",
    )
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: calls.append(kwargs) or fake_run,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("TASK_RUN_DIR", str(tmp_path / "task"))
    (tmp_path / "wandb_run.json").write_text(
        json.dumps({"run_id": "local-only-id"}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        wandb_project="test-project",
        wandb_name="test-run",
        wandb_dir=tmp_path,
    )

    run = init_wandb(args)

    assert run is fake_run
    assert calls[0]["id"] == "local-only-id"
    assert calls[0]["resume"] == "allow"
