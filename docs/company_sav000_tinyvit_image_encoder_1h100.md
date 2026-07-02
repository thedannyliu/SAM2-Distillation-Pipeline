# Company SA-V sav_000 TinyViT Image-Encoder Training

This run estimates one-H100 training time on the extracted SA-V `sav_000`
shard while freezing non-image components.

Default path:

```text
/group-volume/danny-dataset/SA-V/sav_000
```

Training phases:

```text
warmup:   freeze all modules except image_encoder.neck
finetune: freeze all modules except image_encoder
```

The TinyViT trunk starts from the open-source timm/Hugging Face checkpoint at:

```text
/group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
```

Run inside the company PyTorch container on one H100:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull

DRY_RUN=1 scripts/company/08_run_sav_tinyvit_image_encoder_1h100.sh all
scripts/company/08_run_sav_tinyvit_image_encoder_1h100.sh all
```

By default this first timing run uses at most 20 videos from `sav_000` to avoid
spending hours extracting frames before the training path is known to work. For
a full-shard timing run, set:

```bash
SAV_MAX_VIDEOS=0 scripts/company/08_run_sav_tinyvit_image_encoder_1h100.sh all
```

Useful overrides:

```bash
SAV_SHARD_ROOT=/group-volume/danny-dataset/SA-V/sav_000 \
SAV_MAX_VIDEOS=20 \
WARMUP_EPOCHS=1 \
FINETUNE_EPOCHS=1 \
BATCH_SIZE=1 \
NUM_FRAMES=8 \
MAX_OBJECTS=3 \
RESOLUTION=1024 \
IMAGE_ENCODER_BATCH=1 \
IMAGE_ENCODER_CKPT=1 \
scripts/company/08_run_sav_tinyvit_image_encoder_1h100.sh all
```

If `JPEGImages_24fps` is missing, the script extracts frames from mp4 files
under the shard with `tools/data/extract_sav_frames_local.py`. This requires
`cv2` in the container. The script auto-detects common SA-V layouts, including
annotations under either the shard itself or the parent train directory, such
as `/group-volume/danny-dataset/SA-V/train/annotations`.

Outputs:

```text
/group-volume/danny-dataset/sam2_distill/runs/sav000_tinyvit_image_encoder_1h100
  checkpoints/checkpoint.pt
  config_resolved.yaml
  runtime_warmup.json
  runtime_finetune.json
  runtime_estimate.json
  summary_warmup.json
  summary_finetune.json
```

`runtime_estimate.json` contains observed seconds per step and estimated hours
for `TARGET_STEPS`, default 1000.
