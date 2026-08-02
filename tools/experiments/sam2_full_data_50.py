#!/usr/bin/env python3
"""Define the 50-run full SA-V SAM2 experiment matrix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field


DEFAULTS = {
    "FD_BASE_PROFILE": "mx5",
    "FD_DATA_COHORT": "dense8",
    "FD_LATENCY_MODE": "bucket",
    "TASK_EPOCHS": "8",
    "TASK_NUM_FRAMES": "8",
    "TASK_TRAIN_BATCH_SIZE": "1",
    "TASK_MAX_NUM_OBJECTS": "8",
    "TASK_TRAINABLE_MODE": "object_slot_shared_kv",
    "TASK_MEMORY_TOPOLOGY": "standard4",
    "TASK_MEMORY_LAYERS": "4",
    "TASK_MEMORY_INITIALIZER": "current",
    "TASK_MEMORY_LAYOUT": "legacy",
    "TASK_OBJECT_SLOT_MODE": "shared_kv",
    "TASK_OBJECT_SLOT_COUNT": "8",
    "TASK_OBJECT_SLOT_MIN_OBJECTS": "4",
    "TASK_OBJECT_RESIDUAL_RANK": "16",
    "TASK_OBJECT_POINTER_RESIDUAL_RANK": "8",
    "TASK_OBJECT_RESIDUAL_TEMPORAL_POOL": "mean",
    "TASK_OBJECT_RESIDUAL_TEMPORAL_DECAY": "0.5",
    "TASK_HEAD_LR": "1.0e-5",
    "TASK_HEAD_LR_END": "1.0e-6",
    "TASK_MEMORY_LR": "3.0e-5",
    "TASK_MEMORY_LR_END": "3.0e-6",
    "TASK_MEMORY_AUX_LR": "1.0e-5",
    "TASK_MEMORY_AUX_LR_END": "1.0e-6",
    "TASK_PERCEIVER_LR": "1.0e-5",
    "TASK_PERCEIVER_LR_END": "1.0e-6",
    "TASK_ENCODER_LR": "0",
    "TASK_ENCODER_LR_END": "0",
    "TASK_LAMBDA_TASK": "1",
    "TASK_LAMBDA_IMG": "0",
    "TASK_LAMBDA_MEM": "1",
    "TASK_LAMBDA_MASK_LOGITS": "2",
    "TASK_LAMBDA_OBJ_PTR": "0.25",
    "TASK_PROB_USE_POINT": "0.5",
    "TASK_PROB_USE_BOX": "0.5",
    "TASK_PROB_SAMPLE_GT": "0.1",
    "TASK_NUM_FRAMES_TO_CORRECT": "1",
    "TASK_RANDOM_CORRECTION_FRAMES": "false",
    "TASK_NUM_CORRECTION_POINTS": "0",
    "TASK_NUM_GLOBAL_LATENTS": "0",
    "TASK_NUM_2D_LATENTS": "0",
}


@dataclass(frozen=True)
class Experiment:
    variant: str
    node: int
    question: str
    hypothesis: str
    overrides: dict[str, str] = field(default_factory=dict)

    @property
    def env(self) -> dict[str, str]:
        values = dict(DEFAULTS)
        values.update(self.overrides)
        return values


def exp(
    number: int,
    name: str,
    node: int,
    question: str,
    hypothesis: str,
    **overrides: object,
) -> Experiment:
    return Experiment(
        variant=f"FD{number:02d}_{name}",
        node=node,
        question=question,
        hypothesis=hypothesis,
        overrides={key: str(value) for key, value in overrides.items()},
    )


def standard_anchor(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "FD_BASE_PROFILE": "tv21",
        "FD_DATA_COHORT": "all",
        "FD_LATENCY_MODE": "legacy",
        "TASK_TRAINABLE_MODE": "mask_decoder_memory",
        "TASK_OBJECT_SLOT_MODE": "none",
        "TASK_OBJECT_SLOT_COUNT": 0,
        "TASK_OBJECT_RESIDUAL_RANK": 0,
        "TASK_OBJECT_POINTER_RESIDUAL_RANK": 0,
        "TASK_LAMBDA_MEM": 0,
        "TASK_LAMBDA_MASK_LOGITS": 0,
        "TASK_LAMBDA_OBJ_PTR": 0,
        "TASK_MEMORY_LR": "5.0e-7",
        "TASK_MEMORY_LR_END": "5.0e-8",
        "TASK_MEMORY_AUX_LR": "2.5e-7",
        "TASK_MEMORY_AUX_LR_END": "2.5e-8",
        "TASK_HEAD_LR": "2.5e-7",
        "TASK_HEAD_LR_END": "2.5e-8",
    }
    values.update(overrides)
    return values


EXPERIMENTS = [
    # Node 1: TV21 full-data accuracy ceiling.
    exp(1, "tv21_t4_decmem_5ep", 1, "TV21 accuracy", "T4 control", **standard_anchor(TASK_EPOCHS=5, TASK_NUM_FRAMES=4)),
    exp(2, "tv21_t8_decmem_5ep", 1, "TV21 accuracy", "Longer clips improve memory", **standard_anchor(TASK_EPOCHS=5, TASK_NUM_FRAMES=8)),
    exp(3, "tv21_t8_logits1_8ep", 1, "TV21 accuracy", "Logit KD prevents drift", **standard_anchor(TASK_EPOCHS=8, TASK_NUM_FRAMES=8, TASK_LAMBDA_MASK_LOGITS=1)),
    exp(4, "tv21_t8_joint_logits1_8ep", 1, "TV21 accuracy", "Low-LR encoder adaptation lifts the ceiling", **standard_anchor(TASK_EPOCHS=8, TASK_NUM_FRAMES=8, TASK_TRAINABLE_MODE="image_encoder_mask_decoder_memory", TASK_ENCODER_LR="5.0e-8", TASK_ENCODER_LR_END="5.0e-9", TASK_LAMBDA_MASK_LOGITS=1)),
    exp(5, "tv21_t12_joint_mem025_logits2_8ep", 1, "TV21 accuracy", "T12 plus behavior KD improves long tracking", **standard_anchor(TASK_EPOCHS=8, TASK_NUM_FRAMES=12, TASK_TRAINABLE_MODE="image_encoder_mask_decoder_memory", TASK_ENCODER_LR="5.0e-8", TASK_ENCODER_LR_END="5.0e-9", TASK_LAMBDA_MEM=0.25, TASK_LAMBDA_MASK_LOGITS=2)),

    # Node 2: recover TV11 to the 95% quality Pareto boundary.
    exp(6, "tv11_t4_decmem_5ep", 2, "TV11 Pareto", "Full-data T4 control", **standard_anchor(FD_BASE_PROFILE="tv11", TASK_EPOCHS=5, TASK_NUM_FRAMES=4)),
    exp(7, "tv11_t8_decmem_5ep", 2, "TV11 Pareto", "T8 closes the remaining quality gap", **standard_anchor(FD_BASE_PROFILE="tv11", TASK_EPOCHS=5, TASK_NUM_FRAMES=8)),
    exp(8, "tv11_t8_logits1_8ep", 2, "TV11 Pareto", "TV21 logit KD recovers boundary detail", **standard_anchor(FD_BASE_PROFILE="tv11", TASK_EPOCHS=8, TASK_NUM_FRAMES=8, TASK_LAMBDA_MASK_LOGITS=1)),
    exp(9, "tv11_t8_joint_logits1_8ep", 2, "TV11 Pareto", "Very-low-LR joint tuning reaches 95%", **standard_anchor(FD_BASE_PROFILE="tv11", TASK_EPOCHS=8, TASK_NUM_FRAMES=8, TASK_TRAINABLE_MODE="image_encoder_mask_decoder_memory", TASK_ENCODER_LR="7.5e-8", TASK_ENCODER_LR_END="7.5e-9", TASK_LAMBDA_MASK_LOGITS=1)),
    exp(10, "tv11_t12_joint_mem025_logits2_8ep", 2, "TV11 Pareto", "Long clips and behavior KD improve TV11 memory", **standard_anchor(FD_BASE_PROFILE="tv11", TASK_EPOCHS=8, TASK_NUM_FRAMES=12, TASK_TRAINABLE_MODE="image_encoder_mask_decoder_memory", TASK_ENCODER_LR="7.5e-8", TASK_ENCODER_LR_END="7.5e-9", TASK_LAMBDA_MEM=0.25, TASK_LAMBDA_MASK_LOGITS=2)),

    # Node 3: isolate cohort diversity from temporal-window density.
    exp(11, "sharedkv_all_t8_r16_8ep", 3, "Data scaling", "All videos maximize identity diversity", FD_DATA_COHORT="all"),
    exp(12, "sharedkv_dense4_t8_r16_8ep", 3, "Data scaling", "Dense-4 balances diversity and multiplex supervision", FD_DATA_COHORT="dense4"),
    exp(13, "sharedkv_dense8_t8_r16_8ep", 3, "Data scaling", "Dense-8 maximizes strict multiplex supervision"),
    exp(14, "sharedkv_dense4_t12_r16_8ep", 3, "Data scaling", "Dense-4 benefits from longer temporal windows", FD_DATA_COHORT="dense4", TASK_NUM_FRAMES=12),
    exp(15, "sharedkv_dense8_t12_r16_8ep", 3, "Data scaling", "Dense-8 T12 is the strict long-context anchor", TASK_NUM_FRAMES=12),

    # Node 4: low-rank identity path capacity.
    *[
        exp(15 + index, f"sharedkv_r{rank}_ptr8_8ep", 4, "Residual rank", f"Spatial residual rank {rank}", TASK_OBJECT_RESIDUAL_RANK=rank)
        for index, rank in enumerate((2, 4, 8, 16, 32), 1)
    ],

    # Node 5: memory/logit KD balance at the strongest practical rank.
    exp(21, "sharedkv_r16_mem025_logits2_8ep", 5, "KD balance", "Weak memory KD preserves task plasticity", TASK_LAMBDA_MEM=0.25),
    exp(22, "sharedkv_r16_mem050_logits2_8ep", 5, "KD balance", "Moderate memory KD balances identity", TASK_LAMBDA_MEM=0.5),
    exp(23, "sharedkv_r16_mem100_logits2_8ep", 5, "KD balance", "Current memory-KD anchor", TASK_LAMBDA_MEM=1),
    exp(24, "sharedkv_r16_mem200_logits2_8ep", 5, "KD balance", "Strong memory KD preserves temporal state", TASK_LAMBDA_MEM=2),
    exp(25, "sharedkv_r16_mem100_logits4_8ep", 5, "KD balance", "Stronger mask KD recovers J&F", TASK_LAMBDA_MEM=1, TASK_LAMBDA_MASK_LOGITS=4),

    # Node 6: pointer rank and temporal pooling.
    exp(26, "sharedkv_r16_ptr0_mean_8ep", 6, "Pointer identity", "Spatial residual alone is sufficient", TASK_OBJECT_POINTER_RESIDUAL_RANK=0, TASK_LAMBDA_OBJ_PTR=0),
    exp(27, "sharedkv_r16_ptr4_mean_obj010_8ep", 6, "Pointer identity", "A rank-4 pointer path is enough", TASK_OBJECT_POINTER_RESIDUAL_RANK=4, TASK_LAMBDA_OBJ_PTR=0.1),
    exp(28, "sharedkv_r16_ptr8_mean_obj025_8ep", 6, "Pointer identity", "Rank-8 mean-pool anchor", TASK_OBJECT_POINTER_RESIDUAL_RANK=8, TASK_LAMBDA_OBJ_PTR=0.25),
    exp(29, "sharedkv_r16_ptr8_latest_obj025_8ep", 6, "Pointer identity", "Latest memory avoids stale identity", TASK_OBJECT_RESIDUAL_TEMPORAL_POOL="latest"),
    exp(30, "sharedkv_r16_ptr8_recency050_obj025_8ep", 6, "Pointer identity", "Recency weighting balances stability and change", TASK_OBJECT_RESIDUAL_TEMPORAL_POOL="recency", TASK_OBJECT_RESIDUAL_TEMPORAL_DECAY=0.5),

    # Node 7: temporal context length.
    *[
        exp(30 + index, f"sharedkv_t{frames}_r16_ptr8_8ep", 7, "Temporal horizon", f"Train on T={frames}", TASK_NUM_FRAMES=frames)
        for index, frames in enumerate((4, 6, 8, 12, 16), 1)
    ],

    # Node 8: prompt and correction curriculum.
    exp(36, "sharedkv_pointheavy_8ep", 8, "Prompt curriculum", "Point-heavy training improves interactive tracking", TASK_PROB_USE_POINT=0.75, TASK_PROB_USE_BOX=0.25),
    exp(37, "sharedkv_mixedprompt_8ep", 8, "Prompt curriculum", "Balanced point/box control"),
    exp(38, "sharedkv_boxheavy_8ep", 8, "Prompt curriculum", "Box-heavy initialization stabilizes identity", TASK_PROB_USE_POINT=0.25, TASK_PROB_USE_BOX=0.75),
    exp(39, "sharedkv_correct2x3_8ep", 8, "Prompt curriculum", "Two random corrected frames improve recovery", TASK_NUM_FRAMES_TO_CORRECT=2, TASK_RANDOM_CORRECTION_FRAMES="true", TASK_NUM_CORRECTION_POINTS=3),
    exp(40, "sharedkv_correct3x7_8ep", 8, "Prompt curriculum", "Aggressive corrections improve hard sequences", TASK_NUM_FRAMES_TO_CORRECT=3, TASK_RANDOM_CORRECTION_FRAMES="true", TASK_NUM_CORRECTION_POINTS=7),

    # Node 9: deployment routing threshold and slot capacity.
    exp(41, "sharedkv_slot4_min4_r16_8ep", 9, "Routing", "Four-slot buckets minimize N=4 cost", TASK_OBJECT_SLOT_COUNT=4),
    exp(42, "sharedkv_slot6_min4_r16_8ep", 9, "Routing", "Six slots are the capacity knee", TASK_OBJECT_SLOT_COUNT=6),
    exp(43, "sharedkv_slot8_min2_r16_8ep", 9, "Routing", "Shared path becomes useful at N=2", TASK_OBJECT_SLOT_MIN_OBJECTS=2),
    exp(44, "sharedkv_slot8_min3_r16_8ep", 9, "Routing", "N=3 routing improves crossover", TASK_OBJECT_SLOT_MIN_OBJECTS=3),
    exp(45, "sharedkv_slot8_min4_gt025_r16_8ep", 9, "Routing", "GT sampling stabilizes the N>=4 route", TASK_PROB_SAMPLE_GT=0.25),

    # Node 10: memory-depth and true EdgeTAM temporal modules.
    exp(46, "mem4_t8_decmem_8ep", 10, "Memory architecture", "Four-layer full-data memory anchor", **standard_anchor(TASK_EPOCHS=8, TASK_NUM_FRAMES=8)),
    exp(47, "mem2_t8_decmem_8ep", 10, "Memory architecture", "Two standard layers retain quality", **standard_anchor(TASK_EPOCHS=8, TASK_NUM_FRAMES=8, TASK_MEMORY_TOPOLOGY="standard2", TASK_MEMORY_LAYERS=2)),
    exp(48, "mem2_t8_joint_logits2_8ep", 10, "Memory architecture", "Joint tuning recovers two-layer loss", **standard_anchor(TASK_EPOCHS=8, TASK_NUM_FRAMES=8, TASK_MEMORY_TOPOLOGY="standard2", TASK_MEMORY_LAYERS=2, TASK_TRAINABLE_MODE="image_encoder_mask_decoder_memory", TASK_ENCODER_LR="5.0e-8", TASK_ENCODER_LR_END="5.0e-9", TASK_LAMBDA_MASK_LOGITS=2)),
    exp(49, "edgetam2_temporal_logits2_8ep", 10, "Memory architecture", "Full data trains official EdgeTAM temporal modules", **standard_anchor(TASK_EPOCHS=8, TASK_NUM_FRAMES=8, TASK_MEMORY_TOPOLOGY="edgetam_hybrid2", TASK_MEMORY_LAYERS=2, TASK_MEMORY_INITIALIZER="official_pair", TASK_MEMORY_LAYOUT="official", TASK_TRAINABLE_MODE="memory_perceiver_full", TASK_MEMORY_LR="3.0e-6", TASK_MEMORY_LR_END="3.0e-7", TASK_MEMORY_AUX_LR="1.0e-6", TASK_MEMORY_AUX_LR_END="1.0e-7", TASK_PERCEIVER_LR="1.0e-5", TASK_PERCEIVER_LR_END="1.0e-6", TASK_LAMBDA_MASK_LOGITS=2, TASK_NUM_GLOBAL_LATENTS=256, TASK_NUM_2D_LATENTS=256)),
    exp(50, "edgetam2_joint_img_logits2_8ep", 10, "Memory architecture", "Joint encoder-temporal KD recovers EdgeTAM quality", **standard_anchor(TASK_EPOCHS=8, TASK_NUM_FRAMES=8, TASK_MEMORY_TOPOLOGY="edgetam_hybrid2", TASK_MEMORY_LAYERS=2, TASK_MEMORY_INITIALIZER="official_pair", TASK_MEMORY_LAYOUT="official", TASK_TRAINABLE_MODE="image_encoder_memory_perceiver", TASK_ENCODER_LR="5.0e-8", TASK_ENCODER_LR_END="5.0e-9", TASK_MEMORY_LR="3.0e-6", TASK_MEMORY_LR_END="3.0e-7", TASK_MEMORY_AUX_LR="1.0e-6", TASK_MEMORY_AUX_LR_END="1.0e-7", TASK_PERCEIVER_LR="1.0e-5", TASK_PERCEIVER_LR_END="1.0e-6", TASK_LAMBDA_IMG=1, TASK_LAMBDA_MASK_LOGITS=2, TASK_NUM_GLOBAL_LATENTS=256, TASK_NUM_2D_LATENTS=256)),
]


BY_NAME = {experiment.variant: experiment for experiment in EXPERIMENTS}


def validate() -> dict[str, object]:
    errors = []
    if len(EXPERIMENTS) != 50:
        errors.append(f"expected 50 experiments, found {len(EXPERIMENTS)}")
    if len(BY_NAME) != len(EXPERIMENTS):
        errors.append("experiment names are not unique")
    node_counts = Counter(experiment.node for experiment in EXPERIMENTS)
    if node_counts != Counter({node: 5 for node in range(1, 11)}):
        errors.append(f"queues are not balanced: {dict(node_counts)}")
    for experiment in EXPERIMENTS:
        env = experiment.env
        if env["TASK_OBJECT_SLOT_MODE"] == "shared_kv" and int(
            env["TASK_OBJECT_RESIDUAL_RANK"]
        ) <= 0:
            errors.append(f"{experiment.variant}: shared_kv needs a residual")
        if env["FD_DATA_COHORT"] not in {"all", "dense4", "dense8"}:
            errors.append(f"{experiment.variant}: invalid cohort")
        if int(env["TASK_EPOCHS"]) < 5:
            errors.append(f"{experiment.variant}: not a long training run")
    return {
        "status": "pass" if not errors else "fail",
        "experiments": len(EXPERIMENTS),
        "node_counts": dict(sorted(node_counts.items())),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "queue", "env", "describe", "validate"))
    parser.add_argument("target", nargs="?")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "list":
        print("\n".join(experiment.variant for experiment in EXPERIMENTS))
        return
    if args.action == "validate":
        result = validate()
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["status"] == "pass" else 1)
    if args.action == "queue":
        if args.target is None or not args.target.isdigit():
            raise SystemExit("queue target must be an integer from 1 to 10")
        node = int(args.target)
        selected = [item.variant for item in EXPERIMENTS if item.node == node]
        if len(selected) != 5:
            raise SystemExit(f"unknown or unbalanced node queue: {node}")
        print(" ".join(selected))
        return
    if args.target not in BY_NAME:
        raise SystemExit(f"unknown full-data experiment: {args.target}")
    experiment = BY_NAME[args.target]
    if args.action == "env":
        for key, value in sorted(experiment.env.items()):
            print(f"{key}\t{value}")
        return
    print(f"{experiment.variant}\tnode={experiment.node}")
    print(f"question: {experiment.question}")
    print(f"hypothesis: {experiment.hypothesis}")
    for key, value in sorted(experiment.env.items()):
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
