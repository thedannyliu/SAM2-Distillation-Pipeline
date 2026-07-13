# Company Stage 1 Run Recovery

Use the mounted complete SA-V release and shared run roots to audit all expected
SAM2.1 and SAM3.1 Stage 1 experiments before assigning GPUs:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline

SAV_ROOT=/mnt/data/danny-dataset/SA-V \
RUNS_ROOTS=/group-volume/danny-dataset/sam2_distill/runs:/mnt/data/danny-dataset/sam2_distill/runs \
scripts/company/32_audit_stage1_run_progress.sh
```

The audit is read-only. It reads checkpoint metadata and local W&B run IDs but
does not contact W&B or start GPU work. It covers the 15 registered SAM2.1
experiments and 9 registered SAM3.1 experiments, then discovers additional
checkpoint-bearing run directories for manual review.

`complete` requires all of the following:

1. The registered five-epoch target step is reached.
2. A readable `best.pt` selected by validation exists.
3. That `best.pt` has box-prompt image mIoU/AP and memory VOS J&F results over
   every video in the mounted `sav_val.txt`.
4. The same full evaluation exists for every video in `sav_test.txt`.

Feature-loss validation during training is not a substitute for downstream
full SA-V validation. A checkpoint with finished training but missing full
evaluation is reported as `needs_full_eval`.

Status meanings:

| status | meaning |
| --- | --- |
| `complete` | training, best checkpoint, full val, and full test all complete |
| `needs_full_eval` | training and best checkpoint complete; full val/test missing or stale |
| `needs_final_validation` | target steps reached but no readable `best.pt` |
| `resumable` | `last.pt` contains model, optimizer, and the original W&B run ID |
| `resumable_missing_wandb_id` | weights can resume, but continuing the same W&B run is not currently guaranteed |
| `missing` | no checkpoint exists at the registered path |
| `invalid` | checkpoint exists but is unreadable or lacks required resume state |
| `unregistered_*` | discovered legacy run, with the suffix describing its recovery state |

Reports are written to:

```text
/user-volume/stage1_run_progress_${HOSTNAME}/stage1_run_progress.csv
/user-volume/stage1_run_progress_${HOSTNAME}/stage1_run_progress.json
```

To audit progress and produce one comparison table for all SAM2.1, SAM3.1,
and discovered legacy runs, use:

```bash
SAV_ROOT=/mnt/data/danny-dataset/SA-V \
RUNS_ROOTS=/group-volume/danny-dataset/sam2_distill/runs:/mnt/data/danny-dataset/sam2_distill/runs \
scripts/company/35_report_stage1_experiment_metrics.sh
```

This command is read-only and does not require a GPU. It writes:

```text
/user-volume/stage1_experiment_report_${HOSTNAME}/experiment_key_metrics.csv
/user-volume/stage1_experiment_report_${HOSTNAME}/incomplete_runs.csv
/user-volume/stage1_experiment_report_${HOSTNAME}/experiment_report.md
/user-volume/stage1_experiment_report_${HOSTNAME}/experiment_report.json
```

`experiment_key_metrics.csv` has one row per run and split. Image columns
include mIoU, AP, set-image latency, prompt latency, and total latency per
object. Video columns include J&F, J, F, elapsed time, and seconds per video.
`evaluation_complete=true` means the metrics use the current `best.pt` and
cover every video in that split. Partial or stale metrics remain visible but
are not marked complete. `incomplete_runs.csv` records training/checkpoint,
full-validation, and full-test gaps plus the next recovery action.

To measure deployable SAM2.1 hybrid weight sizes without counting optimizer
state from `best.pt`, provide one checkpoint for each TinyViT variant:

```bash
TV21_CHECKPOINT=/path/to/tv21m/checkpoints/best.pt \
TV11_CHECKPOINT=/path/to/tv11m/checkpoints/best.pt \
TV5_CHECKPOINT=/path/to/tv5m/checkpoints/best.pt \
scripts/company/36_measure_sam2_hybrid_sizes.sh
```

The report separates the TinyViT image encoder from the shared SAM2.1
non-image modules and records theoretical FP32/FP16 tensor storage plus actual
serialized inference bundle sizes. By default, pure FP32 and FP16 bundles are
written under `/user-volume/sam2_hybrid_sizes_${HOSTNAME}`. Set
`EXPORT_DTYPES=fp16` to export only FP16, or `EXPORT_DTYPES=` to produce the
CSV/JSON estimates without writing bundles. Architecture is inferred from the
projection shape; a checkpoint labelled TV11M or TV5M but containing TV21M
weights causes the command to fail after writing the diagnostic report.

Do not relaunch queues until the CSV has been used to assign each incomplete
run to exactly one node. Resume must reuse the same run directory and
`checkpoints/last.pt`; the trainer then reads `wandb_run_id` from the checkpoint
and resumes the same W&B run.

## Recovery execution contract

Prepare the mounted manifest once before starting the recovery lanes:

```bash
SAV_ROOT=/mnt/data/danny-dataset/SA-V \
scripts/company/33_prepare_mounted_sav_stage1_manifest.sh
```

`scripts/company/34_run_stage1_recovery_lane.sh` is intentionally strict. For
every registered experiment assigned to the lane it performs these steps in
order:

1. Resume `checkpoints/last.pt` in the original run and W&B run, or start the
   missing experiment.
2. Require the target step and validation-selected `checkpoints/best.pt`.
3. Evaluate that exact `best.pt` on all 155 `sav_val` videos.
4. Evaluate that exact `best.pt` on all 150 `sav_test` videos.

Both downstream evaluations use GT box prompts. Image mode runs frame by frame
and records mIoU, AP50:95, encoder/prompt latency, and total per-object latency.
Video mode includes the model's memory tracker and records J&F, J, F, whole
elapsed time, and seconds per video. Results are stored under:

```text
<run>/sav_val_box_benchmark/metrics.csv
<run>/sav_test_box_benchmark/metrics.csv
```

SAM2.1 students use the existing SAM2 prompt decoder and memory pipeline.
SAM3.1 students replace the official Object Multiplex detector vision trunk.
Because the official SAM3.1 semantic box API resets state for every box prompt,
the VOS evaluator uses one memory-tracking session per GT object and maps the
selected output back to the GT object ID. This preserves box-prompt semantics
and supports objects that first appear after frame zero, at the cost of higher
latency than a shared multi-object session.

Before a formal SAM3.1 lane, the company SAM3 checkout must expose the current
multiplex API:

```bash
PYTHONPATH=/user-volume/repo/facebookresearch-sam3 python - <<'PY'
from sam3.model_builder import build_sam3_multiplex_video_predictor
print("SAM3.1 multiplex API: PASS")
PY
```

Set `DRY_RUN=1` to inspect assignments without training or evaluation. Set
`FULL_EVAL=0` only for debugging; such a run will remain incomplete in the
progress audit.
