# SA-1B HF Online Teacher TinyViT-21M

This run distills frozen SAM2.1 Hiera-L image features into pretrained
TinyViT-21M without writing teacher embeddings to disk. Each train step computes
teacher features online under `torch.inference_mode()`.

## Why Online Teacher

SAM2 Stage 1 teacher feature storage is roughly 8 MB per image in fp16:

```text
image_embed  [256, 64, 64]     ~2 MB/image
high_res_s0  [32, 256, 256]    ~4 MB/image
high_res_s1  [64, 128, 128]    ~2 MB/image
```

That means 25k images is about 200 GB of teacher features and 100k images is
about 800 GB. Online teacher forward uses more compute but keeps disk usage
bounded by the downloaded image subset and checkpoints.

## Data Source

The default Hugging Face dataset is:

```text
hdtech/SA-1B
```

Hugging Face lists this mirror as an image dataset with a `train` split and
about 112 GB total file size. Meta describes SA-1B as 11M images and 1.1B masks;
this Stage 1 flow uses only images for encoder feature distillation.

## Setup

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline

bash scripts/company/00_setup_env.sh \
  --sam2-upstream /user-volume/repo/facebookresearch-sam2 \
  --requirements requirements-stage1.txt

python -m pip install --user fvcore iopath
wandb login
```

Download weights if they are not already present:

```bash
bash scripts/company/01_download_weights.sh \
  --out /group-volume/danny-dataset/sam2_distill/checkpoints
```

Expected weights:

```text
/group-volume/danny-dataset/sam2_distill/checkpoints/sam2.1/sam2.1_hiera_large.pt
/group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
```

## Download HF SA-1B Subset

This streams from Hugging Face and writes local JPEGs plus a manifest. It does
not download teacher embeddings.

```bash
HF_MAX_IMAGES=25000 \
HF_MAX_GB=0 \
scripts/company/11_run_sa1b_hf_online_teacher_stage1_21m.sh download
```

Use `HF_MAX_GB` to cap local image storage:

```bash
HF_MAX_IMAGES=200000 \
HF_MAX_GB=180 \
scripts/company/11_run_sa1b_hf_online_teacher_stage1_21m.sh download
```

Outputs:

```text
/group-volume/danny-dataset/SA-1B/hf_hdtech_sa1b_online_v1/images/
/group-volume/danny-dataset/SA-1B/hf_hdtech_sa1b_online_v1/download_summary.json
/group-volume/danny-dataset/sam2_distill/manifests/hf_sa1b_online_tinyvit21m_v1.parquet
```

Verify:

```bash
cat /group-volume/danny-dataset/SA-1B/hf_hdtech_sa1b_online_v1/download_summary.json
python - <<'PY'
import pandas as pd
path="/group-volume/danny-dataset/sam2_distill/manifests/hf_sa1b_online_tinyvit21m_v1.parquet"
df=pd.read_parquet(path)
print(df.shape)
print(df["split"].value_counts())
print(df.head())
PY
```

## Train Online Teacher Stage 1

Default run uses one GPU:

```bash
WANDB_PROJECT=sam2-distill-stage1-online-teacher \
WANDB_NAME=hf-sa1b-online-teacher-tinyvit21m-1gpu \
GPUS=0 \
BATCH_SIZE=1 \
MAX_STEPS=10000 \
PROJECTION_WARMUP_STEPS=1000 \
LR_WARMUP_STEPS=1000 \
scripts/company/11_run_sa1b_hf_online_teacher_stage1_21m.sh train
```

Four GPUs:

```bash
WANDB_PROJECT=sam2-distill-stage1-online-teacher \
WANDB_NAME=hf-sa1b-online-teacher-tinyvit21m-4gpu \
GPUS=0,1,2,3 \
BATCH_SIZE=1 \
MAX_STEPS=10000 \
PROJECTION_WARMUP_STEPS=1000 \
LR_WARMUP_STEPS=1000 \
scripts/company/11_run_sa1b_hf_online_teacher_stage1_21m.sh train
```

Useful overrides:

```text
MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/hf_sa1b_online_tinyvit21m_v1.parquet
RUN_DIR=/group-volume/danny-dataset/sam2_distill/runs/stage1_online_teacher_hf_sa1b_tinyvit21m
SAM2_CKPT=/group-volume/danny-dataset/sam2_distill/checkpoints/sam2.1/sam2.1_hiera_large.pt
TINYVIT_CKPT=/group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
MAX_TRAIN_ITEMS=5000
MAX_VAL_ITEMS=500
NO_WANDB=1
```

## Outputs

```text
<RUN_DIR>/tensorboard
<RUN_DIR>/wandb
<RUN_DIR>/checkpoints/last.pt
<RUN_DIR>/checkpoints/step_*.pt
<RUN_DIR>/run_config.json
<RUN_DIR>/wandb_run.json
```

W&B logs direct scalars:

```text
loss_stage1_total
loss_image_mse
loss_high_res_mse
train/sec_per_step
train/teacher_sec_per_step
train/images_seen
train/epoch
train/eta_hours
train/backbone_trainable
```

## Notes

- This is image encoder feature distillation, not mask/video finetuning.
- Teacher modules are frozen and evaluated online; no teacher zarr/parquet cache
  is written.
- If the run is too slow, increase GPU count first. RAM caching can be added
  later, but it should be optional because DDP can duplicate CPU caches per rank.
