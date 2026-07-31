import sys
from types import SimpleNamespace

from tools.train.run_sam2_task_training import apply_mask_ablation_overrides


class _Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _namespace(value):
    if isinstance(value, dict):
        return _Config(
            {key: _namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def test_obj_ptr_kd_exposes_student_and_teacher_pointer_outputs(
    monkeypatch,
) -> None:
    config = _namespace(
        {
            "trainer": {
                "model": {},
                "seed_value": 0,
                "data": {
                    "train": {
                        "batch_sizes": [1],
                        "datasets": [
                            {
                                "video_dataset": {},
                                "sampler": {"max_num_objects": 2},
                            }
                        ],
                    }
                },
                "loss": {
                    "all": {
                        "weight_dict": {
                            "loss_mask": 20.0,
                            "loss_dice": 1.0,
                        }
                    }
                },
                "optim": {
                    "options": {
                        "weight_decay": [{"scheduler": {"value": 0.05}}]
                    }
                },
            }
        }
    )
    fake_omegaconf = SimpleNamespace(
        OmegaConf=SimpleNamespace(create=_namespace)
    )
    monkeypatch.setitem(sys.modules, "omegaconf", fake_omegaconf)
    monkeypatch.setenv("TASK_MASK_ABLATION_V2", "1")
    monkeypatch.setenv("TASK_LAMBDA_OBJ_PTR", "0.25")
    monkeypatch.setenv("TASK_TEACHER_MODEL_CONFIG", "/teacher/config.yaml")
    monkeypatch.setenv("TASK_TEACHER_CHECKPOINT", "/teacher/checkpoint.pt")

    apply_mask_ablation_overrides(config)

    assert config.trainer.model.expose_obj_ptr_for_distillation is True
    assert (
        config.trainer.model._target_
        == "sam2_distill.edgetam.train_model.EdgeTAMTrainWithTeacher"
    )
    assert config.trainer.loss.all.lambda_obj_ptr == 0.25
