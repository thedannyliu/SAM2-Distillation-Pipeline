# SAM2 mask end-to-end fine-tuning ablation v2

This suite adds 12 controlled runs to the six v1 runs. Every production run is
one resumable foreground pipeline:

`train (W&B + TensorBoard) -> full SA-V val -> full SA-V test -> summary CSV`

Model selection uses **SA-V val J&F only**. Val mIoU and AP are guardrails: a
candidate fails the guardrail if either drops by more than 0.005 relative to its
named control. Test metrics are recorded for reporting and never used to rank or
select a run.

## Common control

Unless a row says otherwise, all runs start from the same completed v2 Stage-1
task checkpoint, train for one epoch with T4 and global batch 4 on 4 H100s,
freeze BatchNorm and the prompt encoder, and use:

- encoder LR `3e-7 -> 3e-8`;
- all other trainable parameters LR `1e-6 -> 1e-7`;
- cosine decay, BF16, AdamW, and gradient clipping at 0.1;
- an initial box plus one error-correction click;
- seed `250107256`, at most two objects per clip.

## Question 1: which training signal and trainable scope work best?

| ID | Change from A00 | Research signal |
|---|---|---|
| A00 | Full E2E T4 control | Reference for all direct one-factor comparisons |
| A01 | Initial box only; zero correction clicks | Isolate the value of error correction |
| A02 | 50% mask, 25% point, 25% box; up to 2 correction frames, 7 clicks, 10% GT clicks | Test the official SAM2 interactive-prompt recipe |
| A03 | Freeze encoder; train decoder and temporal/memory modules | Test whether decoder+memory adaptation is sufficient |
| A04 | Train only temporal/memory modules | Isolate pure temporal adaptation |
| A10 | A01 plus online SAM2.1-L image-feature KD, lambda 0.5 | Test whether a frozen teacher protects image representation |
| A11 | A10 plus memory-feature KD, lambda 0.25 | Test whether teacher temporal features add value |

For A02, the implementation's conditional probabilities are
`P(point-or-box)=0.5` and `P(box | point-or-box)=0.5`, yielding the intended
unconditional 50% mask / 25% point / 25% box mix.

## Question 2: which temporal/data/optimization recipe works best?

| ID | Change from A00 | Research signal |
|---|---|---|
| A05 | T8 on videos with at least 8 usable cached frames | Measure temporal-context benefit |
| A06 | Re-run A05, then T16 bottom-half hard refinement with frozen encoder, half LR, and one-third of A05 update budget | Test the SAM2-style long-clip hard refinement recipe |
| A07 | Linear LR warmup from 0.1x over the first 5% of progress | Measure early optimization stability |
| A08 | 2 clips/GPU, global batch 8, same LR and one epoch | Test batch size at fixed data exposure; optimizer updates are approximately halved |
| A09 | Bottom 50% base-error videos, repeated twice | Test hard-example concentration at approximately matched samples/updates |

## Hardness mining and storage

Hardness uses the frozen common base checkpoint. Each train video gets one
deterministic hash-selected T4 clip, an initial box, and a per-video J, F, and
J&F score. The lowest-J&F half forms A09. The same pass writes T8/T16 eligibility
lists and an exactly budgeted A06 refinement list.

Only JSONL scores, small text lists, logs, and checkpoints are retained. Online
teachers are deliberately excluded from DDP state and checkpoints. Evaluation
prediction PNGs are deleted after scoring. No teacher-feature cache or dataset
copy is created, keeping the suite comfortably inside the 450 GB budget; all
artifacts remain under `/danny-dataset/sam2_distill/runs`.

## Company execution

Run from the company checkout and container. Commands stay in the foreground;
re-running the same command resumes the same checkpoint, W&B run ID,
TensorBoard directory, and output directory.

First validate a two-update control smoke run:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
mkdir -p /danny-dataset/sam2_distill/runs/sam2_mask_finetune_ablation_v2
GPUS=0,1,2,3 scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh smoke 2>&1 | tee /danny-dataset/sam2_distill/runs/sam2_mask_finetune_ablation_v2/smoke.log
```

Mine common base-error scores once. A00-A04, A07-A08, and A10-A11 do not depend
on this pass and may run concurrently; A05, A06, and A09 require it.

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
mkdir -p /danny-dataset/sam2_distill/runs/sam2_mask_finetune_ablation_v2
GPUS=0,1,2,3 scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh prepare-hardness 2>&1 | tee /danny-dataset/sam2_distill/runs/sam2_mask_finetune_ablation_v2/prepare-hardness.log
```

Launch one of the following on each 4-H100 node. Do not combine them with shell
backgrounding; each terminal keeps its own live output.

```bash
GPUS=0,1,2,3 scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh run A00_e2e_t4_box1_control
GPUS=0,1,2,3 scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh run A01_e2e_t4_box0
GPUS=0,1,2,3 scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh run A02_e2e_t4_official_prompt
GPUS=0,1,2,3 scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh run A03_decmem_t4
GPUS=0,1,2,3 scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh run A04_memory_t4
GPUS=0,1,2,3 scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh run A05_e2e_t8
GPUS=0,1,2,3 scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh run A06_e2e_t8_s4_t16_hard
GPUS=0,1,2,3 scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh run A07_e2e_t4_warmup5
GPUS=0,1,2,3 scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh run A08_e2e_t4_gb8
GPUS=0,1,2,3 scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh run A09_e2e_t4_hard50x2
GPUS=0,1,2,3 scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh run A10_e2e_t4_box0_imgkd
GPUS=0,1,2,3 scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh run A11_e2e_t4_box0_imgmemkd
```

Each completed job atomically upserts the central table. Rebuild it at any time,
including any finished v1 rows discoverable under `MASK_ABLATION_V1_ROOT`:

```bash
scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh summarize
```

The central result is
`/danny-dataset/sam2_distill/runs/sam2_mask_finetune_ablation_v2/summary.csv`.
Each variant also has `summary.csv`, `experiment.json`, `resolved_config.yaml`,
the W&B run descriptor, TensorBoard logs, trainer checkpoint, exported
`stage.pt`, and full val/test metric CSVs.

## Recovery and failure isolation

- Training failure: repeat `run VARIANT`; the trainer resumes its checkpoint.
- Eval failure after training: run `eval VARIANT`; it does not retrain.
- W&B network failure: set `WANDB_MODE=offline`, repeat the command, then sync
  the retained offline run later.
- A10/A11 OOM: do not change the experiment silently. Record the failure first;
  then test a bounded smoke with the same config before deciding whether teacher
  activation checkpointing or a separate image is required.
- Hardness interruption: repeat `prepare-hardness`; each rank skips video IDs
  already present in its JSONL shard.
- If the base checkpoint, manifest, GPU world size, or mining seed intentionally
  changes, set `HARDNESS_FORCE=1` for one `prepare-hardness` rerun; stale scores
  are otherwise rejected.
