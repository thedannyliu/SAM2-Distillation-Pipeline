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

The lane only starts `PL3` when the val-selected pseudo branch reaches
`val J&F >= 66.5`. This prevents a longer-clip continuation when neither
pseudo-label weight improves enough to justify it.

## 2026-07-28 18:30 UTC Status

Source report:
`/user-volume/all_experiment_report_20260728/all_experiments_20260728.csv`.

| Run | Status | Progress | val mIoU | val AP | val image s | val J&F | val video s |
|---|---|---:|---:|---:|---:|---:|---:|
| `tv5_PL0_gt_t4_3ep` | complete | 100% | **0.8004** | **0.6446** | 0.0376 | **66.0** | 29.9622 |
| `tv5_PL1_sam21l_soft025_t4_3ep` | complete | 100% | 0.7998 | 0.6441 | **0.0326** | 65.9 | **29.6724** |
| `tv5_PL2_sam21l_soft050_t4_3ep` | not started | 0% | - | - | - | - | - |
| `tv5_PL3_selected_t8_2ep` | not started | 0% | - | - | - | - | - |

| Run | test mIoU | test AP | test image s | test J&F | test video s |
|---|---:|---:|---:|---:|---:|
| `tv5_PL0_gt_t4_3ep` | **0.8031** | 0.6570 | 0.0346 | 67.3 | 31.1594 |
| `tv5_PL1_sam21l_soft025_t4_3ep` | **0.8031** | **0.6571** | **0.0333** | **68.1** | **31.1025** |
| `tv5_PL2_sam21l_soft050_t4_3ep` | - | - | - | - | - |
| `tv5_PL3_selected_t8_2ep` | - | - | - | - | - |

Relative to the matched PL0 control, PL1 changes validation
mIoU/AP/J&F by -0.0006/-0.0005/-0.1 and test mIoU/AP/J&F by
0.0000/+0.0001/+0.8. Because selection is validation-only, the test gain is
descriptive and PL1 does not currently pass the teacher-benefit criterion.
PL0 also matches the starting checkpoint's 66.0 validation J&F, strengthening
the evidence that ordinary extra E2E iterations are saturated.

The next valid observation is PL2 versus PL0. PL3 may still characterize T8
continuation, but it can only support the pseudo-mask hypothesis if the
validation-selected pseudo source beats PL0.
