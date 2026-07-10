# SAM3.1 to TinyViT-21M Image-Encoder Distillation

## Scope

This pipeline distills only the SAM3.1 image encoder. It does not train the
text encoder, detector, mask decoder, or multiplex video memory.

The teacher target is the raw SAM3.1 detector ViT trunk output:

| Component | Contract |
|---|---|
| Input | RGB resized to `1008 x 1008`, normalized with mean/std `0.5` |
| Teacher | SAM3.1 ViT trunk extracted from `sam3.1_multiplex.pt` |
| Target | `[B, 1024, 72, 72]` |
| Student | ImageNet-pretrained TinyViT-21M final feature |
| Interface | `1x1` projection plus BN-free residual depthwise adapter |
| Loss | `MSE + 0.25 * cosine` over the raw trunk feature |

The raw trunk contract is intentional: the trained student can replace the
ViT trunk while retaining SAM3's official neck and downstream modules. SAM3.1
multiplex changes are primarily in video tracking and memory, so they are not
distilled during this stage.

Relevant code:

- Teacher checkpoint extraction: `sam2_distill/models/sam31_teacher.py`
- Student interface: `sam2_distill/models/tinyvit_sam3_adapter.py`
- Loss: `sam2_distill/training/sam31_stage1_losses.py`
- Trainer: `tools/train/train_sam31_stage1_online_teacher.py`
- Company launcher: `scripts/company/26_run_sam31_stage1_tv21.sh`

## Company Setup

The setup command installs only SAM3 Python dependencies. It does not upgrade
the container's preinstalled PyTorch.

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline

DATA_ROOT=/group-volume/danny-dataset \
SAM3_UPSTREAM=/user-volume/repo/facebookresearch-sam3 \
scripts/company/26_run_sam31_stage1_tv21.sh setup
```

The current company checkpoint paths are:

```text
/group-volume/danny-dataset/sam3/checkpoints/sam3.1/sam3.1_multiplex.pt
/group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
```

Run the strict one-batch compatibility test before training:

```bash
DATA_ROOT=/group-volume/danny-dataset \
MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps.parquet \
GPUS=0 \
scripts/company/26_run_sam31_stage1_tv21.sh inspect
```

Expected output includes `status: pass`, the selected checkpoint prefix, and
matching teacher/student shapes `[1, 1024, 72, 72]`. A failure here means the
company SAM3 checkout, PyTorch 2.4 runtime, or checkpoint format is incompatible;
do not start the formal run until it is resolved.

## Training

First run a tracked 20-step smoke test:

```bash
DATA_ROOT=/group-volume/danny-dataset \
MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps.parquet \
GPUS=0 \
WANDB_PROJECT=sam31-distill-stage1 \
scripts/company/26_run_sam31_stage1_tv21.sh smoke
```

Formal 8-H100 run:

```bash
DATA_ROOT=/group-volume/danny-dataset \
MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps.parquet \
GPUS=0,1,2,3,4,5,6,7 \
BATCH_SIZE=2 \
NUM_WORKERS=8 \
MAX_TRAIN_ITEMS=300000 \
EPOCHS=5 \
LR=1e-4 \
MIN_LR=1e-6 \
PROJECTION_WARMUP_STEPS=2000 \
LR_WARMUP_STEPS=2000 \
LAMBDA_MSE=1.0 \
LAMBDA_COS=0.25 \
ADAPTER_MODE=residual_dwconv \
WANDB_PROJECT=sam31-distill-stage1 \
WANDB_NAME=tv21m-adapter-sam31-mse-cos025-5ep-v1 \
RUN_DIR=/group-volume/danny-dataset/sam2_distill/runs/sam31_stage1/tv21m_adapter_mse_cos025_5ep_v1 \
scripts/company/26_run_sam31_stage1_tv21.sh train
```

The 300k training frames are selected deterministically from the manifest using
seed `310107256`, rather than taking the first rows. Validation uses the official
`val_sav` split. The run writes W&B and TensorBoard metrics and retains only:

```text
checkpoints/best.pt
checkpoints/last.pt
```

Restarting the same command resumes `last.pt` and reuses the W&B run ID stored in
that checkpoint. If batch size 2 fails the one-batch test or OOMs, set
`BATCH_SIZE=1`; the launcher recomputes epoch and evaluation step counts.

## First Decision Point

Compare validation MSE, cosine loss, and downstream SAM3 image/video benchmarks
for these two runs before expanding the search:

1. `residual_dwconv`, MSE + `0.25` cosine (primary).
2. `projection`, MSE + `0.25` cosine (adapter ablation).

Do not infer segmentation quality from feature loss alone. A successful Stage 1
run still requires splicing the student trunk into the official SAM3.1 neck and
measuring mask/video metrics against the untouched teacher.

## References

- [Official SAM3 repository](https://github.com/facebookresearch/sam3)
- [SAM3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
- [Official SAM3.1 checkpoint](https://huggingface.co/facebook/sam3.1)
