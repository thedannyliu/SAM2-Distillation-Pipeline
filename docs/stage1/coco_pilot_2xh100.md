# COCO Stage 1 Pilot: 1000 Train / 100 Val On 2xH100

This pilot runs the complete Stage 1 loop on the company cluster only. Do not run this full pipeline on PACE.

## 0. Paths

```bash
export SAM2D_REPO=/user-volume/repo/SAM2-Distillation-Pipeline
export SAM2_UPSTREAM=/user-volume/repo/facebookresearch-sam2
export SAM2D_ROOT=/group-volume/danny-dataset/sam2_distill
export COCO_RAW=/group-volume/danny-dataset/coco2017_raw
export SAM2D_ENV=/user-volume/env/sam2_stage1_torch24
export GPUS=0,1
export WANDB_PROJECT=sam2-distill-stage1
```

Expected final layout:

```text
/group-volume/danny-dataset/sam2_distill/
  checkpoints/
    sam2.1/sam2.1_hiera_large.pt
    tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
  pilot/coco_1000train_100val/
    images/train/*.jpg
    images/val/*.jpg
    manifests/coco_pilot_1000train_100val.parquet
    manifests/coco_pilot_boxes.jsonl
    teacher_cache/sam2p1_large/
  runs/stage1_coco_pilot/
    checkpoints/last.pt
    tensorboard/
    wandb_run.json
    benchmark_val/
      summary.json
      latencies.jsonl
      overlays/*.jpg
```

## 1. Pull Code And Prepare Env

```bash
cd /user-volume/repo
git clone https://github.com/thedannyliu/SAM2-Distillation-Pipeline.git
git clone https://github.com/facebookresearch/sam2.git facebookresearch-sam2
cd $SAM2D_REPO
git pull origin main

bash scripts/company/00_setup_env.sh \
  --venv $SAM2D_ENV \
  --sam2-upstream $SAM2_UPSTREAM
source $SAM2D_ENV/bin/activate
```

## 2. Organize Existing Checkpoints

You said these already exist under `/group-volume/danny-dataset`:

```text
SAM2.1_hiera_large.pt
model.safetensors
```

Move or copy them into the pipeline layout:

```bash
mkdir -p $SAM2D_ROOT/checkpoints/sam2.1
mkdir -p $SAM2D_ROOT/checkpoints/tinyvit

cp /group-volume/danny-dataset/SAM2.1_hiera_large.pt \
  $SAM2D_ROOT/checkpoints/sam2.1/sam2.1_hiera_large.pt

cp /group-volume/danny-dataset/model.safetensors \
  $SAM2D_ROOT/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors

sha256sum \
  $SAM2D_ROOT/checkpoints/sam2.1/sam2.1_hiera_large.pt \
  $SAM2D_ROOT/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
```

## 3. Verify W&B And TensorBoard

W&B:

```bash
wandb login
wandb status

python - <<'PY'
import wandb
run = wandb.init(project="sam2-distill-smoke", name="company-wandb-smoke")
wandb.log({"ok": 1})
run.finish()
PY
```

TensorBoard:

```bash
mkdir -p $SAM2D_ROOT/logs/tensorboard-smoke

python - <<'PY'
from torch.utils.tensorboard import SummaryWriter
logdir = "/group-volume/danny-dataset/sam2_distill/logs/tensorboard-smoke"
writer = SummaryWriter(logdir)
writer.add_scalar("smoke/ok", 1, 0)
writer.close()
print(logdir)
PY

find $SAM2D_ROOT/logs/tensorboard-smoke -type f -name 'events.out.tfevents.*' -ls
```

Launch TensorBoard if needed:

```bash
tensorboard --logdir $SAM2D_ROOT/runs --host 0.0.0.0 --port 6006
```

## 4. Download And Clean COCO

If COCO is not already extracted:

```bash
mkdir -p $COCO_RAW
cd $COCO_RAW

wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip -q train2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip
```

Keep only the pilot subset under `$SAM2D_ROOT`, then delete zip files and the full extracted `train2017/val2017` image directories from `$COCO_RAW`:

```bash
cd $SAM2D_REPO

bash scripts/company/04_run_coco_stage1_pilot.sh prepare
```

This creates exactly 1000 train images and 100 val images in:

```text
$SAM2D_ROOT/pilot/coco_1000train_100val/images/train
$SAM2D_ROOT/pilot/coco_1000train_100val/images/val
```

It also writes:

```text
$SAM2D_ROOT/pilot/coco_1000train_100val/manifests/coco_pilot_1000train_100val.parquet
$SAM2D_ROOT/pilot/coco_1000train_100val/manifests/coco_pilot_boxes.jsonl
```

After this step, `$COCO_RAW/annotations` remains, but `$COCO_RAW/train2017`, `$COCO_RAW/val2017`, and the downloaded zip files are removed.

## 5. Cache SAM2.1-Large Teacher Embeddings

Use both H100s. Each GPU writes separate shards:

```bash
cd $SAM2D_REPO
source $SAM2D_ENV/bin/activate

bash scripts/company/04_run_coco_stage1_pilot.sh cache
```

Inspect cache:

```bash
python tools/cache/inspect_teacher_cache.py \
  --cache-root $SAM2D_ROOT/pilot/coco_1000train_100val/teacher_cache/sam2p1_large \
  --check-values
```

## 6. Train TinyViT Stage 1

Run on 2xH100:

```bash
cd $SAM2D_REPO
source $SAM2D_ENV/bin/activate

export GPUS=0,1
export BATCH_SIZE=4
export MAX_STEPS=1000
export WANDB_PROJECT=sam2-distill-stage1
export WANDB_NAME=coco-pilot-stage1

bash scripts/company/04_run_coco_stage1_pilot.sh train
```

Outputs:

```text
$SAM2D_ROOT/runs/stage1_coco_pilot/checkpoints/last.pt
$SAM2D_ROOT/runs/stage1_coco_pilot/tensorboard/
$SAM2D_ROOT/runs/stage1_coco_pilot/wandb_run.json
```

Resume:

```bash
export WANDB_RUN_ID=$(python - <<'PY'
import json
print(json.load(open("/group-volume/danny-dataset/sam2_distill/runs/stage1_coco_pilot/wandb_run.json"))["run_id"])
PY
)
export WANDB_RESUME=allow

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 \
  tools/train/train_stage1.py \
    --manifest $SAM2D_ROOT/pilot/coco_1000train_100val/manifests/coco_pilot_1000train_100val.parquet \
    --cache-root $SAM2D_ROOT/pilot/coco_1000train_100val/teacher_cache/sam2p1_large \
    --tinyvit-checkpoint $SAM2D_ROOT/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors \
    --out-dir $SAM2D_ROOT/runs/stage1_coco_pilot \
    --resume $SAM2D_ROOT/runs/stage1_coco_pilot/checkpoints/last.pt \
    --batch-size 4 \
    --max-steps 1000
```

## 7. Benchmark Box-Prompt Masks And Save Overlays

This uses the trained TinyViT encoder output with the frozen SAM2 decoder and COCO val boxes:

```bash
bash scripts/company/04_run_coco_stage1_pilot.sh benchmark
```

Results:

```text
$SAM2D_ROOT/runs/stage1_coco_pilot/benchmark_val/summary.json
$SAM2D_ROOT/runs/stage1_coco_pilot/benchmark_val/latencies.jsonl
$SAM2D_ROOT/runs/stage1_coco_pilot/benchmark_val/overlays/*.jpg
```

The overlay files are qualitative sanity checks. Stage 1 only aligns image features; masks may be imperfect until Stage 2 prompt distillation.

## 8. One Command Pilot

After checkpoints and COCO are in place:

```bash
cd $SAM2D_REPO
source $SAM2D_ENV/bin/activate

export SAM2D_ROOT=/group-volume/danny-dataset/sam2_distill
export COCO_RAW=/group-volume/danny-dataset/coco2017_raw
export GPUS=0,1
export BATCH_SIZE=4
export MAX_STEPS=1000
export WANDB_PROJECT=sam2-distill-stage1

bash scripts/company/04_run_coco_stage1_pilot.sh all
```

For first run, prefer running `prepare`, `cache`, `train`, and `benchmark` separately so failures are easier to isolate.
