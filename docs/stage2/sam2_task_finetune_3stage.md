# SAM2 Task-Loss Progressive Fine-Tuning

## Question

Can the Stage 1 TinyViT-21M encoder recover prompt segmentation and video
tracking quality when SAM2.1-L modules are progressively unfrozen, without the
instability seen when the whole model is optimized from the first step?

## Data

- Train candidates: all 50,453 SA-V train videos represented by the mounted
  manifest, using the existing 16 deterministic 6-fps-aligned frames per
  video. The input audit resolves mounted manual JSONs and excludes only the
  release videos that have no readable manual annotation.
- Validation: complete mounted `sav_val` split.
- Test: complete mounted `sav_test` split.
- Prompt: ground-truth box prompt for both image segmentation and video
  tracking evaluation.

"All SA-V" in this run means every train candidate with a readable manual
annotation. It does not decode every raw 24-fps MP4 frame; training uses the
released sampled frame set so the run is reproducible and fits the current
storage layout.

The mounted Stage 1 manifest may contain blank legacy `annotation_path`
values. Task training reconstructs them from
`SAV_ROOT/sav_train/<shard>/<video>_manual.json`; it never treats a blank path
as the current directory. The audit reports the exact included and excluded
video counts before any GPU process starts.

## Experiment Matrix

| Stage | Research question | Trainable modules | Frames/clip | Epochs | Encoder LR | Other LR |
|---|---|---|---:|---:|---:|---:|
| 1 | Can task loss adapt the distilled representation alone? | TinyViT + projection | 2 | 2 | 1e-6 to 1e-7 | n/a |
| 2 | Does decoder co-adaptation improve prompt masks? | Stage 1 + mask decoder | 2 | 2 | 5e-7 to 5e-8 | 2e-6 to 2e-7 |
| 3 | Does memory co-adaptation improve VOS? | Stage 2 + memory and tracking modules | 4 | 1 | 3e-7 to 3e-8 | 1e-6 to 1e-7 |

The prompt encoder remains frozen in every stage. TinyViT BatchNorm statistics
and affine parameters remain frozen because the per-GPU video batch is one.
Frozen decoder/memory modules run in evaluation mode. Training uses the
official SAM2 multi-step focal, dice, IoU, and object-presence losses, bf16,
AdamW, cosine LR schedules, and global gradient clipping at 0.1.
Each training clip starts from a box and uses one official error-driven
correction point; full validation and test remain box-only.

## Results

| Stage | Train status | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F | Image latency | VOS latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pending | | | | | | | | |
| 2 | pending | | | | | | | | |
| 3 | pending | | | | | | | | |

## Artifacts And Metrics

Each stage writes:

- `checkpoints/checkpoint.pt`: optimizer-aware official SAM2 checkpoint used
  for exact continuation.
- `checkpoints/stage.pt`: complete inference state plus the Stage 1-compatible
  encoder state.
- `tensorboard/`, `logs/`, and `wandb/wandb_run.json` in the same stage folder.
- Full `sav_val` and `sav_test` image metrics: mIoU, AP, and latency.
- Full `sav_val` and `sav_test` video metrics: J&F and latency.

Do not judge convergence from adjacent step losses. Video/object composition
changes each step, so the decision signal is epoch-average task loss plus the
full validation metrics. Compare Stage 2 primarily on mIoU/AP and Stage 3
primarily on J&F; reject a stage if its target metric regresses materially.

## Company Command

Run `scripts/company/39_run_sam2_task_finetune_3stage.sh all` on one 4xH100
node. `all` first audits paths and runs an actual 8-video distributed smoke
train before starting the formal stages. Re-running the same command resumes
the stage checkpoint and W&B run, and skips completed evaluation summaries.
The runner replaces upstream full-model and optimizer parameter-set dumps with
a compact parameter-count summary. It also disables upstream environment dumps
and logs task losses/LR directly to W&B instead of using TensorBoard patching;
warnings and tracebacks remain visible.
