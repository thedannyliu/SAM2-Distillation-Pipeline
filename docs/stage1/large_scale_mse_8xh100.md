# Stage 1 Large-Scale MSE Distillation On 8xH100

This run distills SAM2.1-Large image features into TinyViT-21M with MSE losses. Run it on the company cluster, not on PACE.

## 0. Defaults

```bash
export SAM2D_REPO=/user-volume/repo/SAM2-Distillation-Pipeline
export SAM2_UPSTREAM=/user-volume/repo/facebookresearch-sam2
export SAM2D_ROOT=/danny-dataset/sam2_distill
export SA1B_ROOT=/danny-dataset/SA-1B
export IMAGE_ROOT=/danny-dataset/SA-1B/images_3pct
export FINAL_WEIGHT_ROOT=/group-volume/sam2_distill/final_weights
export GPUS=0,1,2,3,4,5,6,7
export WANDB_PROJECT=sam2-distill-stage1
```

Storage policy:

```text
/danny-dataset
  Large data lake for SA-1B, manifests, teacher embedding cache, TensorBoard/W&B logs,
  runs, best.pt, last.pt, and step checkpoints.

/group-volume
  Shared small volume. Keep only final selected/exported weights here.
  Do not put SA-1B images, teacher cache, or intermediate training runs here.
```

Default output layout:

```text
/danny-dataset/sam2_distill/
  checkpoints/
    sam2.1/sam2.1_hiera_large.pt
    tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
  manifests/sa1b_3pct_v1.parquet
  cache/stage1_teacher/sam2p1_large_sa1b_3pct_v1/
  runs/stage1_mse_sa1b_3pct_8xh100/
    checkpoints/
      best.pt
      last.pt
    tensorboard/
    wandb_run.json

/danny-dataset/SA-1B/
  sa1b_links.txt
  manifests/
    sa1b_download_selected_3pct_hash.tsv
    sa1b_download_selected_3pct_hash.json
  images_3pct/

/group-volume/sam2_distill/final_weights/
  stage1_mse_sa1b_3pct_8xh100_best.pt
```

The default data sample is a deterministic SA-1B 3% downloaded shard subset. The training manifest then uses all downloaded images and splits them 90/10 into train/validation:

```bash
export SA1B_DOWNLOAD_PERCENT=3
export SA1B_SELECTION_MODE=hash
export SAMPLE_PERCENT=100
export VAL_FRACTION=0.1
export SEED=sam2_stage1_sa1b_3pct_v1
```

Do not set `SAMPLE_PERCENT=3` after downloading only 3%; that would train on 3% of the 3% subset. Leave `SAMPLE_PERCENT=100` unless you intentionally want a smaller smoke run.

For a different data size, choose matching dataset/manifest/cache/run names explicitly:

```bash
export SA1B_DOWNLOAD_PERCENT=5
export IMAGE_ROOT=/danny-dataset/SA-1B/images_5pct
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
  --sam2-upstream $SAM2_UPSTREAM
```

Do not activate a venv in the company container; use the container Python directly so the preinstalled PyTorch package stays visible.

## 2. Verify Checkpoints

Your model files are already prepared; this step only verifies paths.

Expected files:

```bash
ls -lh \
  $SAM2D_ROOT/checkpoints/sam2.1/sam2.1_hiera_large.pt \
  $SAM2D_ROOT/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
```

If your files are still directly under `/danny-dataset`:

```bash
mkdir -p $SAM2D_ROOT/checkpoints/sam2.1
mkdir -p $SAM2D_ROOT/checkpoints/tinyvit

cp /danny-dataset/SAM2.1_hiera_large.pt \
  $SAM2D_ROOT/checkpoints/sam2.1/sam2.1_hiera_large.pt

cp /danny-dataset/model.safetensors \
  $SAM2D_ROOT/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
```

## 3. Download Deterministic SA-1B 3% Dataset

Get the official SA-1B download link list after accepting the dataset terms from Meta's Segment Anything dataset page. SA-1B is listed as research-use data with 11M images and 1.1B masks, and the official download entry is on Meta's dataset page:

```text
https://ai.meta.com/datasets/segment-anything/
https://ai.meta.com/datasets/segment-anything-downloads/
```

The URL you get from Meta is usually a `.txt` file with this format:

```text
file_name	cdn_link
sa_000020.tar	https://scontent.xx.fbcdn.net/...
sa_000021.tar	https://scontent.xx.fbcdn.net/...
```

The current script can download that link-list file directly. Because fbcdn URLs include expiration parameters, paste the current URL from Meta/HF into `SA1B_LINK_URL` and run the downloader soon after obtaining it:

```bash
cd $SAM2D_REPO

export SA1B_ROOT=/danny-dataset/SA-1B
export SA1B_LINK_FILE=$SA1B_ROOT/sa1b_links.txt
export SA1B_LINK_URL='<paste-current-sa1b-link-list-txt-url-here>'
export REFRESH_LINK_FILE=1
export SA1B_DOWNLOAD_PERCENT=3
export SA1B_SELECTION_MODE=hash
export IMAGE_ROOT=$SA1B_ROOT/images_3pct
export SA1B_DOWNLOAD_WORKERS=8
export KEEP_ARCHIVES=0
export EXTRACT_ANNOTATIONS=0

DRY_RUN=1 bash scripts/company/02_download_sa1b_subset.sh
```

If the dry-run prints a valid shard count, run the actual download:

```bash
bash scripts/company/02_download_sa1b_subset.sh
```

Alternatively, save the URL list manually here:

```bash
mkdir -p /danny-dataset/SA-1B
$EDITOR /danny-dataset/SA-1B/sa1b_links.txt
```

The link file can contain either:

```text
sa_000000.tar https://...
```

or:

```text
https://.../sa_000000.tar
```

If you have an equivalent HF-authorized mirror, write those shard URLs into the same file format. The downloader does not care whether the URL came from Meta or HF; reproducibility comes from preserving the exact `sa1b_links.txt` plus the generated selected-shard TSV/JSON.

Only archive rows ending in `.tar`, `.tar.gz`, `.tgz`, or `.zip` are selected for download. Auxiliary rows such as `sa_images_ids.txt` are ignored by this Stage 1 image downloader.

If `sa1b_links.txt` already exists and you do not want to re-download it, unset `SA1B_LINK_URL` or use `REFRESH_LINK_FILE=0`, then dry-run the deterministic 3% shard selection:

```bash
cd $SAM2D_REPO

export SA1B_ROOT=/danny-dataset/SA-1B
export SA1B_LINK_FILE=$SA1B_ROOT/sa1b_links.txt
unset SA1B_LINK_URL
export REFRESH_LINK_FILE=0
export SA1B_DOWNLOAD_PERCENT=3
export SA1B_SELECTION_MODE=hash
export IMAGE_ROOT=$SA1B_ROOT/images_3pct
export SA1B_DOWNLOAD_WORKERS=8
export KEEP_ARCHIVES=0
export EXTRACT_ANNOTATIONS=0

DRY_RUN=1 bash scripts/company/02_download_sa1b_subset.sh
```

Run the actual download/extract:

```bash
bash scripts/company/02_download_sa1b_subset.sh
```

Default cleanup behavior:

```text
KEEP_ARCHIVES=0
```

So downloaded `.tar`, `.tar.gz`, `.tgz`, `.zip`, and partial archive files are removed after successful extraction. The raw archive directory is removed if it becomes empty. The only non-image files intentionally retained are:

```text
$SA1B_ROOT/sa1b_links.txt
$SA1B_ROOT/manifests/sa1b_download_selected_3pct_hash.tsv
$SA1B_ROOT/manifests/sa1b_download_selected_3pct_hash.json
$SA1B_ROOT/manifests/download_done_3pct_hash/*.done
```

Verify images:

```bash
find $IMAGE_ROOT -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l
find $IMAGE_ROOT -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | head
du -sh $IMAGE_ROOT
```

## 4. Verify W&B And TensorBoard

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

## 5. Build Image Manifest

For speed, the wrapper skips per-file sha256 by default. It still records image path and dimensions. The default wrapper uses the downloaded 3% image root, keeps 100% of those images, and splits the manifest 90/10 into `train` and `val_sa1b`.

```bash
cd $SAM2D_REPO

export SKIP_FILE_SHA256=1
export IMAGE_ROOT=/danny-dataset/SA-1B/images_3pct
export SAMPLE_PERCENT=100
export VAL_FRACTION=0.1
bash scripts/company/05_run_stage1_large_mse_8xh100.sh manifest
```

Check split counts:

```bash
python - <<'PY'
import os
import pandas as pd
manifest = os.environ.get("MANIFEST", "/danny-dataset/sam2_distill/manifests/sa1b_3pct_v1.parquet")
df = pd.read_parquet(manifest)
print(df["split"].value_counts())
print(df.head())
PY
```

The train split is `train`; validation split is `val_sa1b`. The split is deterministic from `SEED` and image relative path, so rerunning manifest creation on the same `images_3pct` tree gives the same split.

## 6. Plan And Cache Teacher Embeddings

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
  --cache-root $SAM2D_ROOT/cache/stage1_teacher/sam2p1_large_sa1b_3pct_v1 \
  --check-values
```

## 7. Train TinyViT With MSE On 8xH100

This run uses:

```text
lambda_mse = 1.0
lambda_l1  = 0.0
lambda_cos = 0.0
lambda_hr  = 1.0
amp_dtype  = bf16
projection_warmup_steps = 2000
lr_warmup_steps         = 2000
max_grad_norm           = 1.0
```

So the objective is MSE on final image embedding plus MSE on high-resolution features. To train only the final image embedding, set `--lambda-hr 0.0` when calling `tools/train/train_stage1.py` directly.

MSE reduction uses PyTorch's default `mean` reduction. That means `loss_image_mse` is averaged over batch, channel, height, and width. `loss_high_res_mse` is `mean_mse(high_res_s0) + mean_mse(high_res_s1)`, so each high-resolution feature scale is normalized by its own batch/channel/spatial element count before the two scales are added.

Stability settings:

```text
projection warmup: freeze TinyViT backbone, train only 1x1 projection heads first
LR warmup:         linearly increase LR to the target LR
gradient clip:    clip trainable parameter grad norm
nonfinite loss:   error by default so failed runs stop visibly
```

Start training:

```bash
export GPUS=0,1,2,3,4,5,6,7
export BATCH_SIZE=8
export NUM_WORKERS=12
export MAX_STEPS=100000
export PROJECTION_WARMUP_STEPS=2000
export LR_WARMUP_STEPS=2000
export MAX_GRAD_NORM=1.0
export EVAL_EVERY=1000
export SAVE_EVERY=5000
export WANDB_PROJECT=sam2-distill-stage1
export WANDB_NAME=stage1-mse-sa1b-3pct-8xh100

bash scripts/company/05_run_stage1_large_mse_8xh100.sh train
```

Outputs:

```text
$RUN_DIR/checkpoints/best.pt
$RUN_DIR/checkpoints/last.pt
$RUN_DIR/tensorboard/
$RUN_DIR/wandb_run.json
```

`best.pt` is selected by the lowest `val/loss_stage1_total`. `last.pt` is refreshed periodically at `SAVE_EVERY` and again at the end, so resume should use `last.pt`.

Keep these training checkpoints in `/danny-dataset`; they are part of the active run state. After you decide which checkpoint is the final Stage 1 artifact, copy only that selected weight to `/group-volume`:

```bash
export FINAL_WEIGHT_ROOT=/group-volume/sam2_distill/final_weights
mkdir -p $FINAL_WEIGHT_ROOT

cp $RUN_DIR/checkpoints/best.pt \
  $FINAL_WEIGHT_ROOT/stage1_mse_sa1b_3pct_8xh100_best.pt

ls -lh $FINAL_WEIGHT_ROOT
```

Default run dir:

```text
/danny-dataset/sam2_distill/runs/stage1_mse_sa1b_3pct_8xh100
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
train/avg_wall_sec_per_step
train/images_seen
train/epoch
train/progress_pct
train/eta_hours
train/backbone_trainable
train/grad_norm
val/loss_stage1_total
```

During projection warmup, `train/backbone_trainable` is `0`. After `PROJECTION_WARMUP_STEPS`, it switches to `1` and the TinyViT backbone starts training with the projection heads.

The terminal also prints dataset size and progress from rank 0:

```text
Stage 1 training summary
  train_images: ...
  val_images: ...
  global_batch_size: ...
  max_steps: ...
step 1,001/100,000 | progress ... | epoch ... | images_seen ... | eta ... | loss ... | mse ... | hr_mse ...
val step 1,001 | loss ... | mse ... | hr_mse ... | best ...
```

## 8. Resume

```bash
export RUN_DIR=/danny-dataset/sam2_distill/runs/stage1_mse_sa1b_3pct_8xh100
export WANDB_RUN_ID=$(python - <<'PY'
import json, os
print(json.load(open(f"{os.environ['RUN_DIR']}/wandb_run.json"))["run_id"])
PY
)
export WANDB_RESUME=allow

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node 8 \
  tools/train/train_stage1.py \
    --manifest $SAM2D_ROOT/manifests/sa1b_3pct_v1.parquet \
    --train-split train \
    --val-split val_sa1b \
    --cache-root $SAM2D_ROOT/cache/stage1_teacher/sam2p1_large_sa1b_3pct_v1 \
    --tinyvit-checkpoint $SAM2D_ROOT/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors \
    --out-dir $RUN_DIR \
    --resume $RUN_DIR/checkpoints/last.pt \
    --batch-size 8 \
    --num-workers 12 \
    --max-steps 100000 \
    --projection-warmup-steps 2000 \
    --lr-warmup-steps 2000 \
    --max-grad-norm 1.0 \
    --lambda-mse 1.0 \
    --lambda-l1 0.0 \
    --lambda-cos 0.0 \
    --lambda-hr 1.0 \
    --amp-dtype bf16
```

## 9. GPU Utilization Knobs

If H100 utilization is low:

```bash
export BATCH_SIZE=12      # if memory allows
export NUM_WORKERS=16
export CACHE_BATCH_SIZE=12
export CACHE_NUM_WORKERS=12
```

If early loss is unstable, increase projection warmup before changing the model:

```bash
export PROJECTION_WARMUP_STEPS=5000
export LR_WARMUP_STEPS=5000
export MAX_GRAD_NORM=0.5
```

If training is stable and too slow to adapt, reduce projection warmup:

```bash
export PROJECTION_WARMUP_STEPS=500
```

If data loading is the bottleneck, keep cache and images on `/danny-dataset` and avoid reading from home, user-volume, or group-volume.

For multiple nodes/jobs, split cache generation with:

```bash
python tools/cache/plan_cache_shards.py \
  --manifest $SAM2D_ROOT/manifests/sa1b_3pct_v1.parquet \
  --shard-size 512 \
  --num-jobs 4
```

Then run separate cache jobs with `--shard-ids` using `scripts/company/03_cache_teacher_embeddings.sh`.

## 10. One Command

After env, checkpoints, and `images_3pct` are ready:

```bash
cd $SAM2D_REPO

export SAM2D_ROOT=/danny-dataset/sam2_distill
export IMAGE_ROOT=/danny-dataset/SA-1B/images_3pct
export GPUS=0,1,2,3,4,5,6,7
export WANDB_PROJECT=sam2-distill-stage1
export WANDB_NAME=stage1-mse-sa1b-3pct-8xh100

bash scripts/company/05_run_stage1_large_mse_8xh100.sh all
```

For the first large run, prefer separate `manifest`, `cache`, and `train` steps so failures are isolated.
