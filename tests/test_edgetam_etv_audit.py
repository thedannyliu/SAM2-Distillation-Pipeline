import json
import sys

import torch
import yaml

from tools.edgetam.audit_etv_run import main


def test_etv_audit_matches_official_temporal_provenance(
    tmp_path,
    monkeypatch,
    capsys,
):
    checkpoint = tmp_path / "edgetam.pt"
    torch.save(
        {
            "model": {
                "memory_attention.weight": torch.ones(2, 2),
                "memory_encoder.weight": torch.ones(2, 2),
                "spatial_perceiver.latents": torch.ones(2, 2),
                "obj_ptr_proj.weight": torch.ones(2, 2),
                "maskmem_tpos_enc": torch.ones(2, 2),
                "image_encoder.weight": torch.ones(3, 3),
            }
        },
        checkpoint,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "initialization_summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "memory_initializer": "official_temporal",
                "tensor_provenance": {"official_edgetam": 5},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "training_model_summary.json").write_text(
        json.dumps(
            {
                "trainable_tensors": 5,
                "trainable_parameters": 20,
                "train_dataset_samples": 50337,
            }
        ),
        encoding="utf-8",
    )
    resolved = {
        "trainer": {
            "model": {
                "freeze_batchnorm": True,
                "memory_attention": {"num_layers": 2},
                "spatial_perceiver": {
                    "num_latents": 256,
                    "num_latents_2d": 256,
                },
            },
            "optim": {"gradient_clip": {"max_norm": 0.1}},
            "loss": {
                "all": {
                    "lambda_mem": 1,
                    "lambda_mask_logits": 1,
                    "task_loss": {
                        "weight_dict": {"loss_mask": 20, "loss_dice": 1}
                    },
                }
            },
        }
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved), encoding="utf-8"
    )
    (run_dir / "loss_outliers.jsonl").write_text(
        json.dumps(
            {
                "global_step": 7,
                "epoch": 0,
                "num_frames": 4,
                "present_object_frames": 6,
                "object_identifiers": [[1, 2]],
                "mask_areas": [[[0, 16], [4, 0]]],
                "losses": {"train/loss_total": 25.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_etv_run.py",
            "--run-dir",
            str(run_dir),
            "--edgetam-checkpoint",
            str(checkpoint),
            "--out-json",
            str(output),
        ],
    )

    main()

    capsys.readouterr()
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    assert summary["edgetam_checkpoint"]["official_temporal_tensors"] == 5
    assert summary["edgetam_checkpoint"]["official_temporal_parameters"] == 20
    assert summary["resolved_contract"]["gradient_clip_max_norm"] == 0.1
    assert summary["loss_outliers"]["count"] == 1
    assert summary["loss_outliers"]["positive_mask_area"]["min"] == 4
    assert summary["loss_outliers"]["positive_mask_area"]["max"] == 16
