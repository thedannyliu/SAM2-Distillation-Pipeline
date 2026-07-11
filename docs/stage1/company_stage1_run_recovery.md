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

Do not relaunch queues until the CSV has been used to assign each incomplete
run to exactly one node. Resume must reuse the same run directory and
`checkpoints/last.pt`; the trainer then reads `wandb_run_id` from the checkpoint
and resumes the same W&B run.
