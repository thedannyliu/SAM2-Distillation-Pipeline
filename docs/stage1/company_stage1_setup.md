# SAM2 Stage 1 Company Setup

This document prepares encoder-only distillation from SAM2.1 teacher image features to a TinyViT-21M student. PACE should only be used for smoke tests; full teacher caching and training should run on the company cluster.

## Paths

Company:

```bash
export SAM2D_REPO=/user-volume/repo/SAM2-Distillation-Pipeline
export SAM2_UPSTREAM=/user-volume/repo/facebookresearch-sam2
export SAM2D_ROOT=/danny-dataset/sam2_distill
export SAM2D_ENV=/user-volume/env/sam2_stage1_torch24
```

PACE smoke:

```bash
export SAM2D_ROOT=/storage/scratch1/9/eliu354/sam2_distill
```

## 1. Prepare Code And Environment

Use the company container `ngc24.06/ub22/py3.10/cu12.5/cudnn9.1/pytorch2.4`.

```bash
cd /user-volume/repo
git clone https://github.com/thedannyliu/SAM2-Distillation-Pipeline.git
git clone https://github.com/facebookresearch/sam2.git facebookresearch-sam2
cd $SAM2D_REPO

bash scripts/company/00_setup_env.sh \
  --venv $SAM2D_ENV \
  --sam2-upstream $SAM2_UPSTREAM
source $SAM2D_ENV/bin/activate
```

This repository is the distillation/preparation scaffold. The official `facebookresearch/sam2` checkout supplies the actual SAM2 package and configs. The setup script keeps the container PyTorch 2.4 runtime by installing SAM2 editable with `--no-build-isolation --no-deps` after installing the non-torch Stage 1 dependencies. If SAM2 import fails because the checked-out SAM2 version requires torch >= 2.5.1, stop and either pin a compatible SAM2 commit or request a torch >= 2.5.1 image. Do not silently upgrade torch in the shared setup script.

## 2. Download Weights

This path uses `wget` and no login token.

```bash
mkdir -p $SAM2D_ROOT/checkpoints
bash scripts/company/01_download_weights.sh --out $SAM2D_ROOT/checkpoints
cat $SAM2D_ROOT/checkpoints/SHA256SUMS.txt
```

Expected files:

```text
$SAM2D_ROOT/checkpoints/sam2.1/sam2.1_hiera_base_plus.pt
$SAM2D_ROOT/checkpoints/sam2.1/sam2.1_hiera_large.pt
$SAM2D_ROOT/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
```

If company networking blocks Hugging Face, manually mirror `model.safetensors` from `timm/tiny_vit_21m_512.dist_in22k_ft_in1k` to the TinyViT path above.

## 3. Build Fixed SA-1B 1% Manifest

```bash
python tools/data/build_image_manifest.py \
  --source sa1b \
  --image-root /danny-dataset/SA-1B/images \
  --sample-percent 1 \
  --seed sam2_stage1_sa1b_1pct_v1 \
  --out $SAM2D_ROOT/manifests/sa1b_1pct_v1.parquet
```

The script ranks all image relative paths with the fixed seed and keeps the first 1%. It reads image dimensions and sha256 only for the selected 1%.

For a fast PACE smoke test, use a tiny local image folder and `--skip-file-sha256`.

## 4. Cache Teacher Image Embeddings

Small smoke:

```bash
bash scripts/company/03_cache_teacher_embeddings.sh \
  --manifest $SAM2D_ROOT/manifests/sa1b_1pct_v1.parquet \
  --teacher base_plus \
  --out $SAM2D_ROOT/cache/stage1_teacher/smoke_bplus \
  --batch-size 2 \
  --shard-size 16 \
  --limit 16

python tools/cache/inspect_teacher_cache.py \
  --cache-root $SAM2D_ROOT/cache/stage1_teacher/smoke_bplus \
  --check-values
```

Full company cache:

```bash
bash scripts/company/03_cache_teacher_embeddings.sh \
  --manifest $SAM2D_ROOT/manifests/sa1b_1pct_v1.parquet \
  --teacher large \
  --out $SAM2D_ROOT/cache/stage1_teacher/sam2p1_large_sa1b_1pct_v1 \
  --batch-size 8 \
  --num-workers 8 \
  --shard-size 512
```

For array jobs, pass `--start-shard $SLURM_ARRAY_TASK_ID --num-shards 1`.

Cache schema per shard:

```text
image_embed: fp16 [N, 256, 64, 64]
high_res_s0: fp16 [N, 32, 256, 256]
high_res_s1: fp16 [N, 64, 128, 128]
index.parquet: sample_id/source/image_path/split/shard_id/row_in_shard
```

## 5. TinyViT Projection/Adapter

Use `sam2_distill.models.tinyvit_adapter.TinyViTSAM2Adapter`. It wraps:

```python
timm.create_model(
    "tiny_vit_21m_512.dist_in22k_ft_in1k",
    features_only=True,
    pretrained=False,
    checkpoint_path="/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors",
)
```

The adapter emits:

```text
high_res_s0 -> [B, 32, 256, 256]
high_res_s1 -> [B, 64, 128, 128]
image_embed -> [B, 256, 64, 64]
```

Stage 1 trains TinyViT plus these projection heads. SAM2 teacher, prompt encoder, mask decoder, and memory modules remain frozen.
