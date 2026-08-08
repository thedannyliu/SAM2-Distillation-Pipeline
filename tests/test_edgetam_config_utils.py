from sam2_distill.edgetam.config_utils import (
    TEACHER_ONLY_MODEL_KEYS,
    strip_teacher_only_model_config,
)


def test_strip_teacher_only_model_config_removes_paired_prompt_flag():
    config = {
        "_target_": "sam2_distill.edgetam.train_model.EdgeTAMTrainWithTeacher",
        "pair_teacher_student_prompts": True,
        "teacher_model_config": "/teacher.yaml",
        "teacher_checkpoint": "/teacher.pt",
        "image_size": 1024,
    }

    strip_teacher_only_model_config(config)

    assert not set(TEACHER_ONLY_MODEL_KEYS).intersection(config)
    assert config == {
        "_target_": "sam2_distill.edgetam.train_model.EdgeTAMTrainWithTeacher",
        "image_size": 1024,
    }
