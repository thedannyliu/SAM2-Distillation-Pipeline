from tools.experiments.sam2_full_data_50 import BY_NAME, EXPERIMENTS, validate


def test_full_data_matrix_has_ten_balanced_long_queues():
    result = validate()

    assert result["status"] == "pass"
    assert result["experiments"] == 50
    assert result["node_counts"] == {node: 5 for node in range(1, 11)}


def test_full_data_matrix_covers_accuracy_multiplex_and_edgetam():
    assert BY_NAME["FD01_tv21_t4_decmem_5ep"].env["FD_DATA_COHORT"] == "all"
    assert BY_NAME["FD06_tv11_t4_decmem_5ep"].env["FD_BASE_PROFILE"] == "tv11"
    assert BY_NAME["FD20_sharedkv_r32_ptr8_8ep"].env[
        "TASK_OBJECT_RESIDUAL_RANK"
    ] == "32"
    assert BY_NAME["FD35_sharedkv_t16_r16_ptr8_8ep"].env[
        "TASK_NUM_FRAMES"
    ] == "16"
    assert BY_NAME["FD49_edgetam2_temporal_logits2_8ep"].env[
        "TASK_MEMORY_TOPOLOGY"
    ] == "edgetam_hybrid2"


def test_every_full_data_experiment_has_a_unique_question_and_variant_pair():
    pairs = {(item.question, item.variant) for item in EXPERIMENTS}

    assert len(pairs) == 50
