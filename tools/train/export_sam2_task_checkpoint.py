#!/usr/bin/env python3
"""Export an official SAM2 Trainer checkpoint for existing Stage 1 benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sam2_distill.models.task_finetune import export_task_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--trainable-mode", required=True)
    parser.add_argument("--source-stage1-checkpoint", required=True)
    parser.add_argument(
        "--student-family",
        choices=("tinyvit", "repvit"),
        default="tinyvit",
    )
    parser.add_argument(
        "--model-name",
        default="tiny_vit_21m_512.dist_in22k_ft_in1k",
    )
    parser.add_argument(
        "--adapter-mode",
        choices=("projection", "residual_dwconv"),
        default="projection",
    )
    args = parser.parse_args()
    summary = export_task_checkpoint(
        trainer_checkpoint=args.trainer_checkpoint,
        output_path=args.output,
        stage_name=args.stage_name,
        trainable_mode=args.trainable_mode,
        source_stage1_checkpoint=args.source_stage1_checkpoint,
        student_family=args.student_family,
        model_name=args.model_name,
        adapter_mode=args.adapter_mode,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
