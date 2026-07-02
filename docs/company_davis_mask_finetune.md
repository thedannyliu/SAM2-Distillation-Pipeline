# Company DAVIS Mask Finetune Quick Run

Use this when SA-V download is slow or blocked and the goal is to get the
EdgeTAM/TinyViT mask-finetune path running quickly on one GPU.

The script downloads the official DAVIS 2017 trainval 480p zip, extracts a
bounded subset, loads the pretrained TinyViT-21M image encoder checkpoint, runs
the upstream SAM2 Trainer mask-loss path, and writes a runtime estimate.

## Paths

```text
code:       /user-volume/repo/SAM2-Distillation-Pipeline
data:       /group-volume/danny-dataset/DAVIS/2017
checkpoints:/group-volume/danny-dataset/sam2_distill/checkpoints
runs:       /group-volume/danny-dataset/sam2_distill/runs/davis_tinyvit_mask_finetune_1gpu
```

Default dataset:

```text
https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
```

## Run

From the company container:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline

DRY_RUN=1 scripts/company/07_run_davis_mask_finetune_1gpu.sh all
scripts/company/07_run_davis_mask_finetune_1gpu.sh all
```

Useful quick overrides:

```bash
DAVIS_MAX_FRAMES=500 \
MAX_EPOCHS=1 \
BATCH_SIZE=1 \
NUM_FRAMES=8 \
MAX_OBJECTS=3 \
RESOLUTION=1024 \
IMAGE_ENCODER_BATCH=1 \
IMAGE_ENCODER_CKPT=1 \
scripts/company/07_run_davis_mask_finetune_1gpu.sh all
```

The default loss is mask-task only for this quick run:

```text
lambda_img = 0
lambda_mem = 0
```

This keeps the run independent of SA-V teacher caches. It is intended as a
pipeline and speed check, not the final EdgeTAM training recipe.

## 403 Fallback

If the DAVIS download returns `403 Forbidden`, first retry with the default
script; it sends a browser user-agent and DAVIS referer for `aria2c`, `wget`,
`curl`, and the Python fallback.

If it still fails, download the zip in a browser or another environment and
place it here:

```text
/group-volume/danny-dataset/DAVIS/2017/raw/DAVIS-2017-trainval-480p.zip
```

Then rerun:

```bash
scripts/company/07_run_davis_mask_finetune_1gpu.sh prepare
scripts/company/07_run_davis_mask_finetune_1gpu.sh train
```

The script skips download when `DAVIS_ZIP` already exists. To use a different
local zip path:

```bash
DAVIS_ZIP=/path/to/DAVIS-2017-trainval-480p.zip \
scripts/company/07_run_davis_mask_finetune_1gpu.sh prepare
```

## Runtime Estimate

After training, read:

```text
/group-volume/danny-dataset/sam2_distill/runs/davis_tinyvit_mask_finetune_1gpu/runtime_estimate.json
```

The file reports:

```text
observed_steps
elapsed_sec
sec_per_step
steps_per_hour
target_steps
estimated_target_hours
estimated_epoch_hours_hint
```

To estimate another run length:

```bash
TARGET_STEPS=10000 scripts/company/07_run_davis_mask_finetune_1gpu.sh estimate
```

For a more stable single-GPU estimate, run at least 50-100 observed steps after
the first smoke pass. The first few steps include model construction, dataloader
startup, and checkpoint overhead.
