# All-Experiment Report

`scripts/company/45_report_all_experiments.sh` builds one row per training run
or fine-tuning stage across every available runs root. It includes registered
experiments that have not started and recursively discovers older or
unregistered runs from checkpoints and standard SA-V benchmark directories.

The report covers the registered SAM2/SAM3.1 Stage 1 queues, RepViT Stage 1,
three-stage task fine-tuning v1/v2, and mask fine-tuning ablations v1/v2.
Additional ablation versions are included automatically when their run
directories contain checkpoints or `sav_val_box_benchmark`/
`sav_test_box_benchmark` metrics.

One CSV contains training progress, checkpoint state, W&B identity, image
segmentation metrics and latency, video tracking metrics and latency, status,
and the next recovery action. The terminal prints a compact tab-separated
`RESULT` row for every experiment so the output can be pasted into a review.

Status is conservative. `complete` requires a completed training target, a
final checkpoint, passing image and VOS rows for both `sav_val` and `sav_test`,
metrics newer than the evaluated training checkpoint, and no pending
`.full_eval_required` marker. Unknown legacy training targets remain
`training_state_unknown` even when partial artifacts exist.

Example:

```bash
RUNS_ROOTS=/danny-dataset/sam2_distill/runs:/group-volume/danny-dataset/sam2_distill/runs \
REPORT_DIR=/user-volume/all_experiment_report \
scripts/company/45_report_all_experiments.sh
```

The default output is `${REPORT_DIR}/all_experiments.csv`. Re-running the
command atomically replaces the CSV and never starts training or evaluation.
