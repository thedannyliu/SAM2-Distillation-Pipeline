# SAM3.1 Stage 1: 3 Nodes x 4 H100 Ablation Plan

## Research questions

SA-V has no text annotations. Increasing feature cosine weight therefore tests
preservation of the teacher's angular feature geometry; it is not direct image-text
alignment. Direct text alignment requires a future text-labelled dataset and a
target after SAM3's vision-language fusion path.

These runs use a second structure-preserving objective:

```text
L_relation = MSE(normalize(S_pool) normalize(S_pool)^T,
                 normalize(T_pool) normalize(T_pool)^T)
```

Teacher and student features are pooled from `72 x 72` to `18 x 18` before the
relation matrix is constructed. This retains spatial-token affinities while
avoiding a full `5184 x 5184` matrix.

All experiments use:

- SAM3.1 raw vision trunk teacher `[B, 1024, 72, 72]`
- TinyViT-21M ImageNet initialization
- all `807,248` video-balanced SA-V training frames
- official `val_sav` validation frames
- 5 epochs
- 4 H100, batch size 4 per GPU, global batch size 16
- bf16 teacher/student AMP
- AdamW, LR `1e-4` to `1e-6`, weight decay `0.05`
- W&B project `sam31-distill-stage1-ablation-v1`
- only `best.pt` and `last.pt`

Each run is approximately `252,265` optimizer steps. Each node executes three
runs sequentially and stops if one fails.

Validation runs over all `1,240` `val_sav` rows at the end of every epoch.
`best.pt` is selected by the lowest weighted `val/loss_stage1_total` for that
experiment, while `last.pt` is updated every epoch and at normal completion.
Component metrics (`val/loss_feature_mse`, `val/loss_feature_cos`, and
`val/loss_spatial_relation`) are also retained in W&B and TensorBoard.

## Experiment matrix

### Node 1: angular feature geometry

| Run | Adapter | Backbone | Cosine | Relation | Question |
|---|---|---:|---:|---:|---|
| `n1_cos000_adapter_ft_w2k` | residual | unfreeze at 2k | 0 | 0 | MSE baseline |
| `n1_cos025_adapter_ft_w2k` | residual | unfreeze at 2k | 0.25 | 0 | conservative angular constraint |
| `n1_cos100_adapter_ft_w2k` | residual | unfreeze at 2k | 1.0 | 0 | strong angular constraint |

### Node 2: student interface and trainability

| Run | Adapter | Backbone | Cosine | Question |
|---|---|---:|---:|---|
| `n2_projection_cos025_ft_w2k` | projection only | unfreeze at 2k | 0.25 | is the residual adapter needed? |
| `n2_adapter_cos025_frozen` | residual | always frozen | 0.25 | can the pretrained representation be retained? |
| `n2_adapter_cos025_ft_w0` | residual | train from step 0 | 0.25 | is projection warmup useful? |

The Node 1 cosine-0.25 run is the missing reference condition for Node 2:
residual adapter with backbone unfreezing at step 2,000.

### Node 3: semantic spatial structure

| Run | Cosine | Relation | Question |
|---|---:|---:|---|
| `n3_cos150_adapter_ft_w2k` | 1.5 | 0 | aggressive angular preservation |
| `n3_relation010_adapter_ft_w2k` | 0 | 0.1 | relation loss without cosine |
| `n3_cos025_relation010_adapter_ft_w2k` | 0.25 | 0.1 | complementary local and relational geometry |

## Preflight

Run setup once on each node:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline

DATA_ROOT=/group-volume/danny-dataset \
scripts/company/26_run_sam31_stage1_tv21.sh setup
```

Run checkpoint/shape inspection on at least one node:

```bash
DATA_ROOT=/group-volume/danny-dataset \
MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps.parquet \
GPUS=0 \
scripts/company/26_run_sam31_stage1_tv21.sh inspect
```

Verify batch size 4 before starting all queues:

```bash
DATA_ROOT=/group-volume/danny-dataset \
MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps.parquet \
GPUS=0 \
SMOKE_BATCH_SIZE=4 \
WANDB_NAME=sam31-tv21m-b4-smoke \
scripts/company/26_run_sam31_stage1_tv21.sh smoke
```

If this OOMs, change `BATCH_SIZE=4` to `BATCH_SIZE=2` in all three queue scripts
before launching. Do not change only one node because that changes optimizer-step
counts and global batch size across ablations.

## Launch commands

Node 1:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
DATA_ROOT=/group-volume/danny-dataset \
MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps.parquet \
GPUS=0,1,2,3 \
scripts/company/27_queue_sam31_4gpu_cosine.sh
```

Node 2:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
DATA_ROOT=/group-volume/danny-dataset \
MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps.parquet \
GPUS=0,1,2,3 \
scripts/company/28_queue_sam31_4gpu_interface.sh
```

Node 3:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
DATA_ROOT=/group-volume/danny-dataset \
MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps.parquet \
GPUS=0,1,2,3 \
scripts/company/29_queue_sam31_4gpu_relations.sh
```

Rerunning the same queue command resumes the active experiment from `last.pt` and
reuses its W&B run ID. Completed experiments load `last.pt`, perform final
validation, and proceed to the next run.

## Selection rule

Rank runs first by downstream SAM3.1 segmentation/tracking metrics after trunk
replacement, then use validation feature loss as diagnostic evidence. Do not
select a final model only by MSE or cosine loss, because lower feature loss does
not guarantee better masks or temporal memory behavior.
