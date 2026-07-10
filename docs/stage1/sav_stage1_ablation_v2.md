# SA-V Stage 1 Ablation V2

This runbook is for the corrected TinyViT Stage 1 ablation matrix on raw SA-V.
It avoids full 24fps extraction and stores only selected 6fps-aligned frames.

## Prepare SA-V Frames

Default output is under `/group-volume/danny-dataset` and uses 64 CPU workers:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline

DATA_ROOT=/group-volume/danny-dataset \
SAV_ROOT=/mnt/data/danny-dataset/SA-V \
TRAIN_ROOT=/mnt/data/danny-dataset/SA-V/sav_train \
VAL_ROOT=/mnt/data/danny-dataset/SA-V/sav_val \
TEST_ROOT=/mnt/data/danny-dataset/SA-V/sav_test \
CACHE_NAME=stage1_vbal16_6fps \
TRAIN_FRAMES_PER_VIDEO=16 \
VAL_FRAMES_PER_VIDEO=8 \
NUM_WORKERS=64 \
scripts/company/18_prepare_sav_stage1_frame_cache.sh
```

Outputs:

```text
/group-volume/danny-dataset/sam2_distill/data/sav_v2/frame_cache/stage1_vbal16_6fps/
/group-volume/danny-dataset/sam2_distill/manifests/stage1_vbal16_6fps.parquet
```

The selected frame index is always annotation-aligned:

```text
frame_idx_24fps = frame_idx_6fps * 4
```

This is intended for image encoder distillation and future video-memory
distillation. Do not expand all SA-V train videos to full 24fps JPEGs.

Raw train data is read from MP4 plus JSON annotations. Official validation
data is read directly from `sav_val/JPEGImages_24fps`; it is not copied into
the frame cache. Validation video IDs are removed from the train split to
prevent video-level leakage.

To repair a manifest produced before official prepared validation data was
supported, reuse the existing train rows and rebuild only validation:

```bash
MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps.parquet \
REUSE_TRAIN_MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps.parquet \
DATA_ROOT=/group-volume/danny-dataset \
SAV_ROOT=/mnt/data/danny-dataset/SA-V \
NUM_WORKERS=64 \
scripts/company/18_prepare_sav_stage1_frame_cache.sh
```

## Run Ablations

Use the preset launcher. It sets model name, checkpoint, adapter mode, loss
weights, teacher, W&B name, run directory, and converts `EPOCHS` to `MAX_STEPS`
from the manifest size.

The queue wrappers below run three experiments sequentially per node. They use
non-overlapping run directories, keep W&B enabled, and save only `last.pt` and
`best.pt`:

```bash
# 8 GPU node
GPUS=0,1,2,3,4,5,6,7 scripts/company/20_queue_sav_stage1_ablation_8gpu.sh

# 4 GPU nodes
GPUS=0,1,2,3 scripts/company/21_queue_sav_stage1_ablation_4gpu_size.sh
GPUS=0,1,2,3 scripts/company/22_queue_sav_stage1_ablation_4gpu_loss.sh
GPUS=0,1,2,3 scripts/company/23_queue_sav_stage1_ablation_4gpu_adapter_teacher.sh
```

After each experiment, the launcher evaluates `best.pt` on the official SA-V
test split with box prompts. Image mode records prompted mIoU, AP, and encoder/
prompt latency. Video mode initializes objects with boxes, runs SAM2 memory
tracking, and records J&F, J, F, and whole-video latency. Per-run results are
stored under `sav_test_box_benchmark/metrics.csv`; all nodes safely upsert into
`runs/sav_stage1_ablation_v2/sav_test_metrics.csv`. VOS prediction PNGs are
deleted after official evaluation to limit storage use.

Training metrics remain logged every 10 steps to W&B and TensorBoard, while
terminal loss output is reduced to every 300 steps.

Queue contents:

| script | experiments |
| --- | --- |
| `20_queue_sav_stage1_ablation_8gpu.sh` | `tv21_proj_sam21l_msehr`, `tv21_proj_sam21l_msehr_cos025`, `tv21_adapter_sam21l_msehr` |
| `21_queue_sav_stage1_ablation_4gpu_size.sh` | `tv11_proj_sam21l_msehr`, `tv5_proj_sam21l_msehr`, `tv11_proj_sam21l_msehr_cos025` |
| `22_queue_sav_stage1_ablation_4gpu_loss.sh` | `tv5_proj_sam21l_msehr_cos025`, `tv21_proj_sam21l_image_only`, `tv21_proj_sam21l_hr025` |
| `23_queue_sav_stage1_ablation_4gpu_adapter_teacher.sh` | `tv21_proj_sam21l_msehr_l1_025`, `tv21_adapter_sam21l_msehr_cos025`, `tv21_proj_sam21bplus_msehr` |

Example 8-GPU run:

```bash
EXPERIMENT=tv21_proj_sam21l_msehr \
GPUS=0,1,2,3,4,5,6,7 \
EPOCHS=5 \
NUM_WORKERS=16 \
scripts/company/19_run_sav_stage1_ablation.sh
```

Example 4-GPU run:

```bash
EXPERIMENT=tv11_proj_sam21l_msehr \
GPUS=0,1,2,3 \
EPOCHS=5 \
NUM_WORKERS=16 \
scripts/company/19_run_sav_stage1_ablation.sh
```

Priority presets:

```text
tv21_proj_sam21l_msehr
tv21_proj_sam21l_msehr_cos025
tv21_adapter_sam21l_msehr
tv21_proj_sam21bplus_msehr
tv11_proj_sam21l_msehr
tv5_proj_sam21l_msehr
tv11_proj_sam21l_msehr_cos025
tv5_proj_sam21l_msehr_cos025
tv21_proj_sam21l_image_only
tv21_proj_sam21l_hr025
tv21_proj_sam21l_msehr_l1_025
tv21_proj_sam21l_msehr_cos1
tv21_adapter_sam21l_msehr_cos025
tv11_adapter_sam21l_msehr
tv5_adapter_sam21l_msehr
tv11_proj_sam21bplus_msehr
tv5_proj_sam21bplus_msehr
tv21_proj_sam21l_msehr_seed2
tv21_proj_sam21l_msehr_vbal64
```

## Reliability Checks

Before full runs:

```bash
python - <<'PY'
import pandas as pd
p="/group-volume/danny-dataset/sam2_distill/manifests/stage1_vbal16_6fps.parquet"
df=pd.read_parquet(p)
print(df["split"].value_counts())
print("all aligned:", bool((df["frame_idx_24fps"] % 4 == 0).all()))
print(df.head())
PY
```

The trainer now fails early if a TV5M/TV11M/TV21M run instantiates the wrong
TinyViT architecture. Expected `projections.image_embed.weight.shape[1]`:

```text
TV21M: 384
TV11M: 256
TV5M: 160
```

## Notes

- `projection` is the original 1x1 projection head.
- `residual_dwconv` adds a BN-free residual adapter after projection.
- SAM3/SAM3.1 is not included in this formal matrix until model code and
  feature mapping are integrated.
