# RepViT to SAM2.1-L Stage 1 Distillation

This experiment asks whether a mobile CNN can preserve SAM2.1-L promptable
segmentation and memory-tracking quality more efficiently than the existing
TinyViT students.

| student | ImageNet initialization | parameters | role |
| --- | --- | ---: | --- |
| RepViT-M0.9 | `repvit_m0_9.dist_450e_in1k` | 5.5M | small-model challenge |
| RepViT-M2.3 | `repvit_m2_3.dist_450e_in1k` | 23.7M | mobile-architecture control |

M2.3 is not smaller than TinyViT-21M by parameter count. Its value is testing
the RepViT architecture and deployment profile at a similar scale.

Prepare the pinned Hugging Face checkpoints on the company cluster:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only origin edgetam-tinyvit-pipeline

HF_XET_HIGH_PERFORMANCE=1 \
scripts/company/37_download_repvit_pretrained.sh
```

The command writes checkpoints, Hugging Face configs, SHA256 files, and model
contract summaries under:

```text
/group-volume/danny-dataset/sam2_distill/checkpoints/repvit
```

The planned training contract matches the TinyViT Stage 1 comparison: frozen
SAM2.1-L teacher, projection-only RepViT student interface, MSE + cosine + L1,
five epochs on the mounted SA-V manifest, full feature-loss validation for
`best.pt`, then full `sav_val` and `sav_test` box-prompt image and memory-VOS
evaluation. Image metrics are mIoU, AP50:95, and latency; VOS metrics are J&F
and latency. RepViT runs use a separate W&B project.

The adapter uses RepViT strides 4 and 8 for SAM2 high-resolution features. It
uses the final stride-32 RepViT stage for `image_embed`, followed by a 1x1
projection and bilinear resize to `[256, 64, 64]`. This keeps every RepViT stage
on the distillation gradient path.

| experiment | batch/GPU | global batch | LR | loss weights |
| --- | ---: | ---: | ---: | --- |
| `repvit_m09_proj_sam21l_msehr_cos025_l1010` | 8 | 32 | 1e-4 | MSE 1, HR-MSE 1, cosine .25, Smooth-L1 .10 |
| `repvit_m23_proj_sam21l_msehr_cos025_l1010` | 4 | 16 | 5e-5 | MSE 1, HR-MSE 1, cosine .25, Smooth-L1 .10 |

Run both experiments sequentially on one 4xH100 company node. The script stays
in the foreground, resumes from each run's `last.pt`, reuses its W&B run ID,
keeps only `best.pt` and `last.pt`, and then evaluates `best.pt` on full
`sav_val` and `sav_test`:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only origin edgetam-tinyvit-pipeline

mkdir -p /user-volume/repvit_logs
LOG="/user-volume/repvit_logs/repvit_stage1_$(date +%Y%m%d_%H%M%S).log"

GPUS=0,1,2,3 \
FULL_EVAL_GPUS=0,1,2,3 \
WANDB_MODE=online \
scripts/company/38_run_repvit_sam21l_stage1.sh all \
2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}
echo "RepViT pipeline status: $STATUS"
echo "Log: $LOG"
```

Metrics are written below each run in `sav_val_box_benchmark/metrics.csv` and
`sav_test_box_benchmark/metrics.csv`. Aggregate CSV files are written at the
RepViT run-root level.
