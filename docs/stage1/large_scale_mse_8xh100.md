# Stage 1 Large-Scale MSE Distillation On 8xH100

This run distills SAM2.1-Large image features into TinyViT-21M with MSE losses. Run it on the company cluster, not on PACE.

## 0. Defaults

```bash
export SAM2D_REPO=/user-volume/repo/SAM2-Distillation-Pipeline
export SAM2_UPSTREAM=/user-volume/repo/facebookresearch-sam2
export SAM2D_ROOT=/group-volume/danny-dataset/sam2_distill
export IMAGE_ROOT=/group-volume/danny-dataset/SA-1B/images
export SAM2D_ENV=/user-volume/env/sam2_stage1_torch24
export GPUS=0,1,2,3,4,5,6,7
export WANDB_PROJECT=sam2-distill-stage1
```

Default output layout:

```text
/group-volume/danny-dataset/sam2_distill/
  checkpoints/
    sam2.1/sam2.1_hiera_large.pt
    tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
  manifests/sa1b_1pct_v1.parquet
  cache/stage1_teacher/sam2p1_large_sa1b_1pct_v1/
  runs/stage1_mse_sa1b_1pct_8xh100/
    checkpoints/
    tensorboard/
    wandb_run.json
```

The default data sample remains deterministic SA-1B 1%:

```bash
export SAMPLE_PERCENT=1
export SEED=sam2_stage1_sa1b_1pct_v1
```

For a bigger speed run, choose a new manifest/cache/run name explicitly:

```bash
export SAMPLE_PERCENT=5
export SEED=sam2_stage1_sa1b_5pct_v1
export MANIFEST=$SAM2D_ROOT/manifests/sa1b_5pct_v1.parquet
export CACHE_ROOT=$SAM2D_ROOT/cache/stage1_teacher/sam2p1_large_sa1b_5pct_v1
export RUN_DIR=$SAM2D_ROOT/runs/stage1_mse_sa1b_5pct_8xh100
```

## 1. Pull Code And Prepare Env

```bash
cd /user-volume/repo
git clone https://github.com/thedannyliu/SAM2-Distillation-Pipeline.git || true
git clone https://github.com/facebookresearch/sam2.git facebookresearch-sam2 || true
cd $SAM2D_REPO
git pull origin main

bash scripts/company/00_setup_env.sh \
  --venv $SAM2D_ENV \
  --sam2-upstream $SAM2_UPSTREAM
source $SAM2D_ENV/bin/activate
```

## 2. Verify Checkpoints

Expected files:

```bash
ls -lh \
  $SAM2D_ROOT/checkpoints/sam2.1/sam2.1_hiera_large.pt \
  $SAM2D_ROOT/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
```

If your files are still directly under `/group-volume/danny-dataset`:

```bash
mkdir -p $SAM2D_ROOT/checkpoints/sam2.1
mkdir -p $SAM2D_ROOT/checkpoints/tinyvit

cp /group-volume/danny-dataset/SAM2.1_hiera_large.pt \
  $SAM2D_ROOT/checkpoints/sam2.1/sam2.1_hiera_large.pt

cp /group-volume/danny-dataset/model.safetensors \
  $SAM2D_ROOT/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
```

## 3. Verify W&B And TensorBoard

W&B:

```bash
wandb login
wandb status

python - <<'PY'
import wandb
run = wandb.init(project="sam2-distill-smoke", name="8xh100-wandb-smoke")
wandb.log({"ok": 1})
run.finish()
PY
```

TensorBoard:

```bash
mkdir -p $SAM2D_ROOT/logs/tensorboard-smoke

python - <<'PY'
from torch.utils.tensorboard import SummaryWriter
import os
root = os.environ["SAM2D_ROOT"]
logdir = f"{root}/logs/tensorboard-smoke"
writer = SummaryWriter(logdir)
writer.add_scalar("smoke/ok", 1, 0)
writer.close()
print(logdir)
PY

find $SAM2D_ROOT/logs/tensorboard-smoke -type f -name 'events.out.tfevents.*' -ls
```

## 4. Build Image Manifest

For speed, the wrapper skips per-file sha256 by default. It still records image path and dimensions.

```bash
cd $SAM2D_REPO
source $SAM2D_ENV/bin/activate

export SKIP_FILE_SHA256=1
bash scripts/company/05_run_stage1_large_mse_8xh100.sh manifest
```

Check split counts:

```bash
python - <<'PY'
import os
import pandas as pd
manifest = os.environ.get("MANIFEST", "/group-volume/danny-dataset/sam2_distill/manifests/sa1b_1pct_v1.parquet")
df = pd.read_parquet(manifest)
print(df["split"].value_counts())
print(df.head())
PY
```

The train split is `train`; validation split is `val_sa1b`.

## 5. Plan And Cache Teacher Embeddings

Estimate shard count:

```bash
bash scripts/company/05_run_stage1_large_mse_8xh100.sh plan-cache
```

Cache SAM2.1-Large teacher features on 8 GPUs:

```bash
export GPUS=0,1,2,3,4,5,6,7
export SHARD_SIZE=512
export CACHE_BATCH_SIZE=8
export CACHE_NUM_WORKERS=8

bash scripts/company/05_run_stage1_large_mse_8xh100.sh cache
```

Each GPU process writes different zarr shards. Completed shards get `.done`; active shards get `.lock`.

Inspect cache:

```bash
python tools/cache/inspect_teacher_cache.py \
  --cache-root $CACHE_ROOT \
  --check-values
```

If `$CACHE_ROOT` was not exported, use the default:

```bash
python tools/cache/inspect_teacher_cache.py \
  --cache-root $SAM2D_ROOT/cache/stage1_teacher/sam2p1_large_sa1b_1pct_v1 \
  --check-values
```

## 6. Train TinyViT With MSE On 8xH100

This run uses:

```text
lambda_mse = 1.0
lambda_l1  = 0.0
lambda_cos = 0.0
lambda_hr  = 1.0
amp_dtype  = bf16
```

So the objective is MSE on final image embedding plus MSE on high-resolution features. To train only the final image embedding, set `--lambda-hr 0.0` when calling `tools/train/train_stage1.py` directly.

Start training:

```bash
export GPUS=0,1,2,3,4,5,6,7
export BATCH_SIZE=8
export NUM_WORKERS=12
export MAX_STEPS=100000
export EVAL_EVERY=1000
export SAVE_EVERY=5000
export WANDB_PROJECT=sam2-distill-stage1
export WANDB_NAME=stage1-mse-sa1b-1pct-8xh100

bash scripts/company/05_run_stage1_large_mse_8xh100.sh train
```

Outputs:

```text
$RUN_DIR/checkpoints/last.pt
$RUN_DIR/tensorboard/
$RUN_DIR/wandb_run.json
```

Default run dir:

```text
/group-volume/danny-dataset/sam2_distill/runs/stage1_mse_sa1b_1pct_8xh100
```

Monitor:

```bash
tensorboard --logdir $SAM2D_ROOT/runs --host 0.0.0.0 --port 6006
```

W&B should show:

```text
loss_stage1_total
loss_image_mse
loss_high_res_mse
train/sec_per_step
val/loss_stage1_total
```

## 7. Resume

```bash
export RUN_DIR=/group-volume/danny-dataset/sam2_distill/runs/stage1_mse_sa1b_1pct_8xh100
export WANDB_RUN_ID=$(python - <<'PY'
import json, os
print(json.load(open(f"{os.environ['RUN_DIR']}/wandb_run.json"))["run_id"])
PY
)
export WANDB_RESUME=allow

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node 8 \
  tools/train/train_stage1.py \
    --manifest $SAM2D_ROOT/manifests/sa1b_1pct_v1.parquet \
    --train-split train \
    --val-split val_sa1b \
    --cache-root $SAM2D_ROOT/cache/stage1_teacher/sam2p1_large_sa1b_1pct_v1 \
    --tinyvit-checkpoint $SAM2D_ROOT/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors \
    --out-dir $RUN_DIR \
    --resume $RUN_DIR/checkpoints/last.pt \
    --batch-size 8 \
    --num-workers 12 \
    --max-steps 100000 \
    --lambda-mse 1.0 \
    --lambda-l1 0.0 \
    --lambda-cos 0.0 \
    --lambda-hr 1.0 \
    --amp-dtype bf16
```

## 8. GPU Utilization Knobs

If H100 utilization is low:

```bash
export BATCH_SIZE=12      # if memory allows
export NUM_WORKERS=16
export CACHE_BATCH_SIZE=12
export CACHE_NUM_WORKERS=12
```

If data loading is the bottleneck, keep cache and images on `/group-volume/danny-dataset` and avoid reading from home or user-volume.

For multiple nodes/jobs, split cache generation with:

```bash
python tools/cache/plan_cache_shards.py \
  --manifest $SAM2D_ROOT/manifests/sa1b_1pct_v1.parquet \
  --shard-size 512 \
  --num-jobs 4
```

Then run separate cache jobs with `--shard-ids` using `scripts/company/03_cache_teacher_embeddings.sh`.

## 9. One Command

After env, checkpoints, and image root are ready:

```bash
cd $SAM2D_REPO
source $SAM2D_ENV/bin/activate

export SAM2D_ROOT=/group-volume/danny-dataset/sam2_distill
export IMAGE_ROOT=/group-volume/danny-dataset/SA-1B/images
export GPUS=0,1,2,3,4,5,6,7
export WANDB_PROJECT=sam2-distill-stage1
export WANDB_NAME=stage1-mse-sa1b-1pct-8xh100

bash scripts/company/05_run_stage1_large_mse_8xh100.sh all
```

For the first large run, prefer separate `manifest`, `cache`, and `train` steps so failures are isolated.
