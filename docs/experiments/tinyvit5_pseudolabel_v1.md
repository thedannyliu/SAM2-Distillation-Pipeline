# TinyViT-5M SAM2.1-L Pseudo-Mask Continuation v1

## Question

Can the best current TinyViT-5M task checkpoint improve further when SAM2.1
Hiera-L supplies online soft mask targets during end-to-end mask fine-tuning?

The teacher target is a soft pseudo-mask generated with the same frames and
prompt as the student. SA-V ground truth remains in the task loss. This tests
teacher boundary and confidence information without adding cache, threshold,
or unlabeled-data confounders.

## Starting point

The validation-selected 5M checkpoint is
`weekend_72h_v1/tinyvit/tv5_W1_decmem_t4_3ep`: val mIoU 0.7997, AP 0.6425,
J&F 66.0; test mIoU 0.8025, AP 0.6552, J&F 67.6.

All forks start from exactly this checkpoint. BatchNorm stays frozen because
training BatchNorm degraded both TinyViT-21M and RepViT in prior experiments.

## Experiment matrix

| Run | Start | T/epochs | Objective | Purpose |
|---|---|---:|---|---|
| `tv5_PL0_gt_t4_3ep` | current best 5M | T4/3 | GT task only | matched extra-training control |
| `tv5_PL1_sam21l_soft025_t4_3ep` | same | T4/3 | GT + 0.25 soft pseudo-mask | conservative teacher regularization |
| `tv5_PL2_sam21l_soft050_t4_3ep` | same | T4/3 | GT + 0.50 soft pseudo-mask | stronger teacher regularization |
| `tv5_PL3_selected_t8_2ep` | best pseudo fork by val J&F | T8/2 | GT + selected soft pseudo-mask weight | longer-horizon refinement |

The first three rows are independent forks with identical trainable modules,
learning rates, seed, data, and prompt simulation. The comparison therefore
separates pseudo-mask supervision from simply training longer. PL3 is selected
using validation J&F only; test metrics are descriptive.

The end-to-end trainable path includes TinyViT, mask decoder, memory attention,
and memory encoder. The prompt encoder remains frozen. The initial T4 learning
rates are 1e-7 for the encoder and 3e-7 for decoder/memory, each decaying by
10x. PL3 halves both rates.

## Decision rule

- A pseudo fork must beat PL0 on validation J&F to claim teacher benefit.
- A gain below 0.5 J&F is treated as noise unless reproduced with another seed.
- A gain of at least 1.0 J&F with no more than 0.005 mIoU or AP loss is promoted.
- PL3 must retain or improve the selected T4 branch; otherwise T8 continuation
  is rejected for 5M.

## Runtime and outputs

The lane schedules 13 T4-equivalent epochs and four full SA-V val/test
evaluations, exceeding 12 hours at the observed company runtime. Training uses
online W&B project `tinyvit5-pseudolabel-v1`, resumes the same W&B ID and
checkpoint directory, and retains only physical `last.pt` and `best.pt`.

Run root:
`/group-volume/danny-dataset/sam2_distill/runs/tinyvit5_pseudolabel_v1`.

Driver:

```bash
scripts/company/58_run_tinyvit5_pseudolabel_lane.sh describe
scripts/company/58_run_tinyvit5_pseudolabel_lane.sh run
```
