import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from tools.benchmark.benchmark_sam2_multiobject_scaling import (
    aggregate_rows,
    binary_mask_metrics,
    parse_object_counts,
)
from tools.data.audit_vos_object_density import (
    main as audit_density_main,
    shared_frame_prompts,
)
from tools.data.select_sav_dense_training_videos import (
    frame_object_counts,
    main as select_dense_main,
    repeat_to_length,
)
from tools.train.summarize_mask_finetune_ablations import (
    add_multiobject_latency,
)
from sam2_distill.models.task_finetune import (
    initialize_edgetam_memory_model,
    initialize_object_slot_model,
)
from sam2_distill.models.sam2_object_buckets import (
    SAM2ObjectBucketAdapter,
    merge_object_output_dicts,
    split_bucket_output,
)
from sam2_distill.models.sam2_object_slots import (
    LearnedObjectSlotDecoder,
    ObjectSlotModelMixin,
    SharedSlotMemoryAttention,
)


def write_mask(path: Path, nonempty: bool) -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    if nonempty:
        mask[2:6, 2:6] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def test_shared_frame_prompts_requires_nonempty_common_frame(tmp_path):
    ann_video = tmp_path / "video"
    write_mask(ann_video / "000" / "00000.png", True)
    write_mask(ann_video / "001" / "00000.png", True)
    write_mask(ann_video / "002" / "00000.png", False)
    write_mask(ann_video / "002" / "00001.png", True)

    shared = shared_frame_prompts(ann_video, min_objects=2)

    assert shared is not None
    assert shared[0] == 0
    assert [object_id for object_id, _ in shared[1]] == ["000", "001"]
    assert shared_frame_prompts(ann_video, min_objects=3) is None


def test_density_audit_records_fixed_prompt_objects(tmp_path, monkeypatch):
    image_root = tmp_path / "images"
    ann_root = tmp_path / "annotations"
    video = "video"
    write_mask(image_root / video / "00000.png", True)
    write_mask(ann_root / video / "000" / "00000.png", True)
    write_mask(ann_root / video / "001" / "00000.png", True)
    video_list = tmp_path / "videos.txt"
    video_list.write_text(f"{video}\n", encoding="utf-8")
    out_dir = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_vos_object_density.py",
            "--image-root",
            str(image_root),
            "--ann-root",
            str(ann_root),
            "--video-list-file",
            str(video_list),
            "--out-dir",
            str(out_dir),
            "--min-shared-objects",
            "2",
        ],
    )

    audit_density_main()

    prompts = pd.read_csv(out_dir / "cohort_prompts.csv")
    assert prompts.loc[0, "prompt_frame"] == 0
    assert json.loads(prompts.loc[0, "object_ids"]) == ["000", "001"]


def test_parse_object_counts_requires_one_object_baseline():
    assert parse_object_counts("8,1,4,4") == [1, 4, 8]
    with pytest.raises(argparse.ArgumentTypeError, match="one-object baseline"):
        parse_object_counts("2,4,8")


def test_aggregate_rows_reports_relative_latency():
    rows = [
        {
            "video": "a",
            "object_count": 1,
            "propagation_ms_per_frame": 10.0,
            "propagation_fps": 100.0,
            "end_to_end_fps": 80.0,
            "prompt_sec": 0.1,
            "peak_memory_mb": 1000.0,
        },
        {
            "video": "a",
            "object_count": 2,
            "propagation_ms_per_frame": 11.0,
            "propagation_fps": 90.9,
            "end_to_end_fps": 75.0,
            "prompt_sec": 0.2,
            "peak_memory_mb": 1100.0,
        },
    ]

    aggregate = aggregate_rows(rows, [1, 2])

    assert aggregate[0]["relative_latency_vs_1"] == pytest.approx(1.0)
    assert aggregate[1]["relative_latency_vs_1"] == pytest.approx(1.1)
    assert aggregate[1]["target_pass"] == 1


def test_dense_training_counts_visible_objects():
    payload = {
        "masklet": [
            [{"counts": "a"}, None, {"counts": "b"}],
            [None, None, None],
        ]
    }

    assert frame_object_counts(payload) == [2, 0]


def test_dense_training_repetition_is_deterministic():
    first = repeat_to_length(["a", "b", "c"], target=8, seed=7)
    second = repeat_to_length(["a", "b", "c"], target=8, seed=7)

    assert first == second
    assert len(first) == 8
    assert set(first) == {"a", "b", "c"}


def test_dense_training_cli_writes_repeated_manifest_cohort(
    tmp_path,
    monkeypatch,
):
    annotation = tmp_path / "sav_000001_manual.json"
    annotation.write_text(
        json.dumps({"masklet": [[{} for _ in range(8)] for _ in range(4)]}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(
        [
            {
                "video_id": "sav_000001",
                "annotation_path": str(annotation),
                "split": "train",
            }
        ]
    ).to_parquet(manifest)
    output = tmp_path / "dense8.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_sav_dense_training_videos.py",
            "--manifest",
            str(manifest),
            "--sav-root",
            str(tmp_path),
            "--output-video-ids",
            str(output),
            "--out-csv",
            str(tmp_path / "index.csv"),
            "--out-summary",
            str(tmp_path / "summary.json"),
            "--target-samples",
            "6",
        ],
    )

    select_dense_main()

    assert output.read_text(encoding="utf-8").splitlines() == [
        "sav_000001"
    ] * 6


def test_summary_reads_one_and_eight_object_latency(tmp_path):
    path = tmp_path / "aggregate.csv"
    path.write_text(
        "object_count,median_propagation_fps,relative_latency_vs_1,"
        "median_peak_memory_mb,target_pass\n"
        "1,100.0,1.0,900.0,\n"
        "8,40.0,2.5,1200.0,0\n",
        encoding="utf-8",
    )
    row = {}

    add_multiobject_latency(row, path)

    assert row["latency_n1_fps"] == "100.0"
    assert row["latency_n8_fps"] == "40.0"
    assert row["latency_n8_relative"] == "2.5"
    assert row["latency_gate_pass"] == "0"


def test_two_layer_memory_initializes_from_four_layer_checkpoint(tmp_path):
    source = torch.nn.Module()
    source.memory_attention = torch.nn.Module()
    source.memory_attention.layers = torch.nn.ModuleList(
        [torch.nn.Linear(2, 2, bias=False) for _ in range(4)]
    )
    target = torch.nn.Module()
    target.memory_attention = torch.nn.Module()
    target.memory_attention.layers = torch.nn.ModuleList(
        [torch.nn.Linear(2, 2, bias=False) for _ in range(2)]
    )
    with torch.no_grad():
        for index, layer in enumerate(source.memory_attention.layers):
            layer.weight.fill_(index + 1)
    checkpoint = tmp_path / "mem4.pt"
    torch.save({"model": source.state_dict()}, checkpoint)

    initialize_edgetam_memory_model(
        target,
        previous_task_checkpoint=str(checkpoint),
        memory_initializer="current",
    )

    assert torch.equal(
        target.memory_attention.layers[0].weight,
        source.memory_attention.layers[0].weight,
    )
    assert torch.equal(
        target.memory_attention.layers[1].weight,
        source.memory_attention.layers[1].weight,
    )


class FakeMemoryAttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(
        self,
        *,
        tgt,
        memory,
        pos,
        query_pos,
        num_k_exclude_rope,
    ):
        self.batch_sizes.append(tgt.shape[0])
        return tgt + memory.mean(dim=1, keepdim=True)


class RoPEAttentionv2:
    pass


class FakeV2MemoryAttentionLayer(FakeMemoryAttentionLayer):
    def __init__(self):
        super().__init__()
        self.cross_attn_image = RoPEAttentionv2()
        self.rope_k_repeats = []

    def forward(self, *, rope_k_repeat, **kwargs):
        self.rope_k_repeats.append(rope_k_repeat)
        return super().forward(**kwargs)


def test_shared_slot_memory_attention_reduces_object_batch():
    module = SharedSlotMemoryAttention(
        d_model=4,
        pos_enc_at_input=False,
        layer=FakeMemoryAttentionLayer(),
        num_layers=1,
        batch_first=True,
        slot_count=4,
        min_objects=4,
        memory_dim=4,
    )
    curr = torch.randn(3, 8, 4)
    memory = torch.randn(5, 8, 4)

    output = module(curr, memory)
    output.sum().backward()

    assert output.shape == curr.shape
    assert module.layers[0].batch_sizes == [2]
    assert torch.equal(output[:, 0], output[:, 1])
    assert torch.equal(output[:, 4], output[:, 7])
    assert module.slot_memory_scale.grad is not None


def test_shared_slot_memory_attention_forwards_edgetam_spatial_memory_count():
    module = SharedSlotMemoryAttention(
        d_model=4,
        pos_enc_at_input=False,
        layer=FakeV2MemoryAttentionLayer(),
        num_layers=1,
        batch_first=True,
        slot_count=4,
        min_objects=4,
        memory_dim=4,
    )

    output = module(
        torch.randn(3, 8, 4),
        torch.randn(5, 8, 4),
        num_spatial_mem=3,
    )

    assert output.shape == (3, 8, 4)
    assert module.layers[0].rope_k_repeats == [3]


def test_shared_slot_memory_attention_keeps_small_batch_legacy_path():
    module = SharedSlotMemoryAttention(
        d_model=4,
        pos_enc_at_input=False,
        layer=FakeMemoryAttentionLayer(),
        num_layers=1,
        batch_first=True,
        slot_count=4,
        min_objects=4,
        memory_dim=4,
    )

    output = module(torch.randn(3, 2, 4), torch.randn(5, 2, 4))

    assert output.shape == (3, 2, 4)
    assert module.layers[0].batch_sizes == [2]


class FakeMaskTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, image, image_pe, tokens):
        self.batch_sizes.append(image.shape[0])
        return tokens, image.flatten(2).transpose(1, 2)


class FakeMaskDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 4
        self.pred_obj_scores = True
        self.use_high_res_features = False
        self.obj_score_token = torch.nn.Embedding(1, hidden_dim)
        self.iou_token = torch.nn.Embedding(1, hidden_dim)
        self.mask_tokens = torch.nn.Embedding(4, hidden_dim)
        self.transformer = FakeMaskTransformer()
        self.output_upscaling = torch.nn.Identity()
        self.output_hypernetworks_mlps = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim)]
        )
        self.iou_prediction_head = torch.nn.Linear(hidden_dim, 4)
        self.pred_obj_score_head = torch.nn.Linear(hidden_dim, 1)


def test_learned_slot_decoder_runs_one_transformer_per_bucket():
    slot_decoder = LearnedObjectSlotDecoder(
        slot_count=4,
        hidden_dim=4,
        min_objects=4,
    )
    decoder = FakeMaskDecoder()
    image_embeddings = torch.randn(8, 4, 2, 2, requires_grad=True)

    masks, ious, tokens, scores = slot_decoder(
        decoder,
        image_embeddings,
        torch.randn(1, 4, 2, 2),
        None,
    )
    masks.sum().backward()

    assert decoder.transformer.batch_sizes == [2]
    assert masks.shape == (8, 1, 2, 2)
    assert ious.shape == (8, 1)
    assert tokens.shape == (8, 1, 4)
    assert scores.shape == (8, 1)
    assert slot_decoder.slot_token_embed.grad is not None
    assert slot_decoder.slot_spatial_scale.grad is not None


class FrozenSAMHead(torch.nn.Module):
    def _forward_sam_heads(self, backbone_features, *args, **kwargs):
        batch = backbone_features.shape[0]
        masks = torch.ones(batch, 1, 2, 2)
        scores = torch.ones(batch, 1)
        pointers = torch.ones(batch, 4)
        return masks, masks, scores, masks, masks, pointers, scores


class FrozenObjectSlotModel(ObjectSlotModelMixin, FrozenSAMHead):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 4
        self._init_object_slots(
            object_slot_count=4,
            object_slot_min_objects=4,
        )


def test_prompt_fallback_keeps_frozen_slot_training_loss_differentiable():
    model = FrozenObjectSlotModel().train()

    outputs = model._forward_sam_heads(
        torch.ones(2, 4, 2, 2),
        point_inputs={
            "point_coords": torch.ones(2, 1, 2),
            "point_labels": torch.ones(2, 1),
        },
    )
    outputs[3].sum().backward()

    assert all(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad) == 0
        for parameter in model.object_slot_decoder.parameters()
    )


class RaisingSlotDecoder(torch.nn.Module):
    min_objects = 4

    def forward(self, *args, **kwargs):
        raise RuntimeError("slot decoder called")


class FakePromptEncoder:
    @staticmethod
    def get_dense_pe():
        return torch.ones(1, 4, 2, 2)


def test_slot_training_uses_small_object_batches_but_eval_does_not():
    model = FrozenObjectSlotModel()
    model.object_slot_decoder = RaisingSlotDecoder()
    model.sam_mask_decoder = torch.nn.Identity()
    model.sam_prompt_encoder = FakePromptEncoder()
    features = torch.ones(2, 4, 2, 2)

    model.eval()
    model._forward_sam_heads(features)

    model.train()
    with pytest.raises(RuntimeError, match="slot decoder called"):
        model._forward_sam_heads(features)


def test_object_slot_initializer_copies_base_and_keeps_new_parameters(tmp_path):
    source = torch.nn.Module()
    source.projection = torch.nn.Linear(2, 2, bias=False)
    target = torch.nn.Module()
    target.projection = torch.nn.Linear(2, 2, bias=False)
    target.object_slot_decoder = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        source.projection.weight.fill_(3)
        target.projection.weight.zero_()
        target.object_slot_decoder.weight.fill_(7)
    checkpoint = tmp_path / "selected.pt"
    torch.save({"model": source.state_dict()}, checkpoint)

    initialize_object_slot_model(target, str(checkpoint))

    assert torch.equal(target.projection.weight, source.projection.weight)
    assert torch.all(target.object_slot_decoder.weight == 7)


def compact_output(value: float) -> dict:
    tensor = torch.tensor([[value]])
    return {
        "maskmem_features": tensor[:, :, None, None],
        "maskmem_pos_enc": [tensor[:, :, None, None]],
        "pred_masks": tensor[:, :, None, None],
        "obj_ptr": tensor,
        "object_score_logits": tensor,
    }


def object_output(value: float) -> dict:
    return {
        "cond_frame_outputs": {0: compact_output(value)},
        "non_cond_frame_outputs": {},
    }


def test_bucket_output_merge_and_split_preserve_object_order():
    merged = merge_object_output_dicts([object_output(2.0), object_output(5.0)])

    assert merged["cond_frame_outputs"][0]["obj_ptr"].tolist() == [[2.0], [5.0]]
    split = split_bucket_output(merged["cond_frame_outputs"][0], batch_size=2)
    assert split[0]["pred_masks"].item() == 2.0
    assert split[1]["pred_masks"].item() == 5.0


class FakeBucketPredictor:
    non_overlap_masks_for_mem_enc = False

    def __init__(self):
        self.batch_sizes = []
        self.output_dict_ids = []
        self.legacy_calls = 0

    def propagate_in_video_preflight(self, inference_state):
        return None

    def propagate_in_video(self, inference_state, **kwargs):
        self.legacy_calls += 1
        masks = torch.cat(
            [
                output["cond_frame_outputs"][0]["pred_masks"]
                for output in inference_state["output_dict_per_obj"].values()
            ]
        )
        yield 0, inference_state["obj_ids"], masks

    def _run_single_frame_inference(
        self,
        *,
        output_dict,
        frame_idx,
        batch_size,
        **kwargs,
    ):
        self.batch_sizes.append(batch_size)
        self.output_dict_ids.append(id(output_dict))
        values = output_dict["cond_frame_outputs"][0]["obj_ptr"] + frame_idx
        output = compact_output(0.0)
        output = {
            key: (
                [values[:, :, None, None]]
                if key == "maskmem_pos_enc"
                else values[:, :, None, None]
                if key in {"maskmem_features", "pred_masks"}
                else values
            )
            for key in output
        }
        return output, output["pred_masks"]

    def _get_orig_video_res_output(self, inference_state, masks):
        return masks, masks


def test_bucket_adapter_runs_one_tracker_call_per_bucket():
    predictor = FakeBucketPredictor()
    adapter = SAM2ObjectBucketAdapter(predictor, bucket_size=4)
    state = {
        "obj_ids": ["a", "b", "c", "d", "e"],
        "num_frames": 3,
        "device": torch.device("cpu"),
        "output_dict_per_obj": {
            index: object_output(float(index)) for index in range(5)
        },
        "frames_tracked_per_obj": {index: {} for index in range(5)},
    }

    frames = list(adapter.propagate_in_video(state))

    assert predictor.batch_sizes == [4, 1, 4, 1]
    assert predictor.output_dict_ids[0] == predictor.output_dict_ids[2]
    assert predictor.output_dict_ids[1] == predictor.output_dict_ids[3]
    assert predictor.output_dict_ids[0] != predictor.output_dict_ids[1]
    assert [frame_idx for frame_idx, _, _ in frames] == [0, 1, 2]
    assert frames[-1][2][:, 0, 0, 0].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert (
        state["output_dict_per_obj"][4]["non_cond_frame_outputs"][2]["obj_ptr"].item()
        == 6.0
    )


def test_bucket_adapter_supports_edgetam_state_without_tracking_bookkeeping():
    predictor = FakeBucketPredictor()
    adapter = SAM2ObjectBucketAdapter(predictor, bucket_size=4)
    state = {
        "obj_ids": ["a", "b", "c", "d"],
        "num_frames": 2,
        "device": torch.device("cpu"),
        "output_dict_per_obj": {
            index: object_output(float(index)) for index in range(4)
        },
    }

    frames = list(adapter.propagate_in_video(state))

    assert predictor.batch_sizes == [4]
    assert [frame_idx for frame_idx, _, _ in frames] == [0, 1]
    assert frames[-1][2][:, 0, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_bucket_adapter_uses_legacy_fast_path_below_threshold():
    predictor = FakeBucketPredictor()
    adapter = SAM2ObjectBucketAdapter(
        predictor,
        bucket_size=4,
        min_bucket_objects=4,
    )
    state = {
        "obj_ids": ["a", "b"],
        "num_frames": 1,
        "device": torch.device("cpu"),
        "output_dict_per_obj": {
            index: object_output(float(index)) for index in range(2)
        },
        "frames_tracked_per_obj": {index: {} for index in range(2)},
    }

    frames = list(adapter.propagate_in_video(state))

    assert predictor.legacy_calls == 1
    assert predictor.batch_sizes == []
    assert frames[0][1] == ["a", "b"]


def test_bucket_adapter_falls_back_for_unsynchronized_prompt_histories():
    predictor = FakeBucketPredictor()
    adapter = SAM2ObjectBucketAdapter(
        predictor,
        bucket_size=4,
        min_bucket_objects=2,
    )
    outputs = {
        index: object_output(float(index)) for index in range(2)
    }
    outputs[1]["cond_frame_outputs"][1] = compact_output(1.0)
    state = {
        "obj_ids": ["a", "b"],
        "num_frames": 1,
        "device": torch.device("cpu"),
        "output_dict_per_obj": outputs,
        "frames_tracked_per_obj": {index: {} for index in range(2)},
    }

    frames = list(adapter.propagate_in_video(state))

    assert predictor.legacy_calls == 1
    assert predictor.batch_sizes == []
    assert frames[0][1] == ["a", "b"]
    assert adapter.execution_stats == {
        "bucket_sessions": 0,
        "legacy_small_sessions": 0,
        "legacy_unsynchronized_sessions": 1,
    }


def test_binary_mask_metrics_exposes_small_boundary_difference():
    reference = torch.zeros((2, 1, 100, 100), dtype=torch.bool)
    reference[:, :, 10:90, 10:90] = True
    candidate = reference.clone()
    candidate[0, 0, 10, 10] = False

    metrics = binary_mask_metrics(reference, candidate)

    assert metrics["mismatched_pixels"] == 1
    assert metrics["total_pixels"] == 20000
    assert min(metrics["mask_ious"]) == pytest.approx(6399 / 6400)
    assert max(metrics["mismatch_fractions"]) == pytest.approx(0.0001)


def test_bucket_adapter_rejects_unsynchronized_history():
    first = object_output(1.0)
    second = object_output(2.0)
    second["non_cond_frame_outputs"][1] = compact_output(3.0)

    with pytest.raises(RuntimeError, match="synchronized object histories"):
        merge_object_output_dicts([first, second])


@pytest.mark.parametrize(
    ("variant", "expected_losses"),
    [
        ("MX1_slot4_decoder_kd_3ep", "1/0/0/1/0"),
        ("MX2_slot8_decoder_kd_3ep", "1/0/0/1/0"),
        ("MX3_slot4_sharedkv_kd_3ep", "1/0/0.25/1/0"),
        ("MX4_slot8_sharedkv_kd_3ep", "1/0/0.25/1/0"),
        ("MX5_slot8_decoder_t8_logits2_5ep", "1/0/0/2/0"),
        ("MX6_slot8_sharedkv_t8_mem1_5ep", "1/0/1/2/0"),
        ("MX7_slot8_sharedkv_t8_mem4_5ep", "1/0/4/2/0"),
        ("MX8_slot8_sharedkv_t8_mem1_logits4_5ep", "1/0/1/4/0"),
    ],
)
def test_object_slot_variants_do_not_require_missing_teacher_pointer(
    variant,
    expected_losses,
):
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            "bash",
            "scripts/company/49_run_edgetam_memory_ablation.sh",
            "describe",
            variant,
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        f"Loss task/image/memory/logits/obj: {expected_losses}"
        in result.stdout
    )
