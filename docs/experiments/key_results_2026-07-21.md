# Key Results: 2026-07-21

Source: company all-experiment report generated from
`/group-volume/danny-dataset/sam2_distill/runs`. The snapshot contains 59
rows: 19 complete, 31 formally incomplete, and 9 superseded historical rows.
The 31 active rows consist of 25 not started, 4 training incomplete, and 2
validation incomplete after superseded runs are classified. Model decisions
use `sav_val`; `sav_test` is reported only as a held-out descriptive result.

## Current Pareto Leaders

| Run | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Task v1 stage 3, encoder+decoder+memory | **0.8381** | **0.7133** | 71.5 | **0.8364** | 0.7171 | **74.3** |
| Task v2 stage 3, encoder+decoder+memory | 0.8380 | 0.7131 | **71.7** | 0.8356 | 0.7163 | 73.9 |
| Stage 1 TV21 projection, MSE+HR+L1 0.25 | 0.8354 | 0.7084 | 70.9 | 0.8359 | 0.7171 | 73.8 |
| Stage 1 TV21 projection, HR weight 0.25 | 0.8352 | 0.7088 | 70.4 | 0.8363 | **0.7175** | 73.5 |
| Stage 1 TV21 projection, MSE+HR | 0.8307 | 0.7010 | 70.6 | 0.8319 | 0.7089 | 73.9 |

Task v1 stage 3 is the current balanced leader: it has the strongest val image
metrics and the best observed test mIoU/J&F. Task v2 stage 3 has the strongest
val J&F and is the correct alternative when validation tracking is the primary
selection metric. Latency was measured on separate evaluation jobs and is not
used for model ranking until hardware/load conditions are controlled.

## All Completed Results

`Image s` is mean total seconds per prompted object. `Video s` is seconds per
video. Values preserve the precision printed by the 2026-07-21 universal
report. A dash is not used for completed rows: all 19 have passing image and
VOS metrics on both splits.

### Task Fine-Tuning

| Suite | Experiment | val mIoU | val AP | val Image s | val J&F | val Video s | test mIoU | test AP | test Image s | test J&F | test Video s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task v1 | `stage1_encoder_task_2ep` | 0.8376 | 0.7129 | 0.0372 | 70.3 | 29.7049 | 0.8356 | 0.7160 | 0.0358 | 71.8 | 31.1089 |
| task v1 | `stage2_encoder_decoder_task_2ep` | 0.8379 | 0.7129 | 0.0404 | 69.9 | 29.6395 | 0.8360 | 0.7167 | 0.0360 | 71.3 | 31.1449 |
| task v1 | `stage3_encoder_decoder_memory_task_1ep` | **0.8381** | **0.7133** | 0.0422 | 71.5 | 29.9859 | **0.8364** | 0.7171 | 0.0402 | **74.3** | 31.3107 |
| task v2 | `stage1_encoder_task_2ep_v2` | 0.8373 | 0.7128 | 0.0373 | 70.2 | 35.4660 | 0.8353 | 0.7160 | 0.0387 | 71.3 | 35.3161 |
| task v2 | `stage2_decoder_only_task_1ep_v2` | 0.8374 | 0.7129 | 0.0349 | 70.0 | 32.5604 | 0.8356 | 0.7161 | 0.0371 | 71.3 | 34.0559 |
| task v2 | `stage3_encoder_decoder_memory_task_1ep_v2` | 0.8380 | 0.7131 | 0.0356 | **71.7** | 32.7550 | 0.8356 | 0.7163 | 0.0376 | 73.9 | 34.9745 |

### SAM2.1 Stage 1

| Experiment | val mIoU | val AP | val Image s | val J&F | val Video s | test mIoU | test AP | test Image s | test J&F | test Video s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tv11_adapter_sam21l_msehr` | 0.8029 | 0.6514 | 0.0166 | 65.6 | 26.9673 | 0.8054 | 0.6635 | 0.0194 | 69.0 | 31.7065 |
| `tv11_proj_sam21l_msehr` | 0.8095 | 0.6617 | 0.0997 | 66.8 | 36.4113 | 0.8128 | 0.6723 | 0.0929 | 69.4 | 37.2279 |
| `tv11_proj_sam21l_msehr_cos025` | 0.8095 | 0.6619 | 0.0446 | 67.1 | 31.4598 | 0.8126 | 0.6725 | 0.0464 | 69.5 | 32.9804 |
| `tv21_adapter_sam21l_msehr` | 0.8301 | 0.7020 | 0.1062 | 71.1 | 34.4817 | 0.8306 | 0.7063 | 0.1148 | 72.9 | 36.2079 |
| `tv21_proj_sam21l_hr025` | 0.8352 | **0.7088** | 0.0402 | 70.4 | 29.9281 | 0.8363 | **0.7175** | 0.0396 | 73.5 | 31.3156 |
| `tv21_proj_sam21l_image_only` | 0.7885 | 0.6297 | 0.0640 | 68.5 | 32.6563 | 0.7836 | 0.6247 | 0.0605 | 70.2 | 34.4005 |
| `tv21_proj_sam21l_msehr` | 0.8307 | 0.7010 | 0.0976 | 70.6 | 34.3292 | 0.8319 | 0.7089 | 0.1099 | **73.9** | 36.4368 |
| `tv21_proj_sam21l_msehr_cos025` | 0.8314 | 0.7016 | 0.1077 | 70.5 | 34.5654 | 0.8325 | 0.7096 | 0.1126 | 73.7 | 36.1136 |
| `tv21_proj_sam21l_msehr_cos1` | 0.8338 | 0.7057 | 0.0488 | 70.4 | 38.6757 | 0.8346 | 0.7121 | 0.0488 | 73.8 | 40.3627 |
| `tv21_proj_sam21l_msehr_l1_025` | **0.8354** | 0.7084 | 0.0353 | 70.9 | 29.6253 | 0.8359 | 0.7171 | 0.0332 | 73.8 | 33.7039 |
| `tv5_adapter_sam21l_msehr` | 0.7951 | 0.6371 | 0.0472 | 64.2 | 33.1468 | 0.7976 | 0.6465 | 0.0472 | 67.1 | 34.5249 |
| `tv5_proj_sam21l_msehr` | 0.7928 | 0.6321 | 0.0469 | 64.2 | 31.4687 | 0.7971 | 0.6458 | 0.0464 | 66.5 | 33.0099 |
| `tv5_proj_sam21l_msehr_cos025` | 0.7923 | 0.6318 | 0.1096 | 63.7 | 35.6212 | 0.7958 | 0.6444 | 0.1001 | 66.2 | 36.7279 |

## All Active Incomplete Experiments

These 31 rows are the formal recovery scope. `Progress` is checkpoint progress
reported in the snapshot, not evaluation progress.

| Suite | Experiment | Status | Progress | Required action |
| --- | --- | --- | ---: | --- |
| RepViT Stage 1 | `repvit_m09_proj_sam21l_msehr_cos025_l1010` | training incomplete | 80% | resume train, val, test |
| RepViT Stage 1 | `repvit_m23_proj_sam21l_msehr_cos025_l1010` | not started | 0% | train, val, test |
| mask v1 | `decoder_lr2e6` | not started | 0% | train, val, test |
| mask v1 | `decoder_lr2e7` | not started | 0% | train, val, test |
| mask v1 | `decoder_lr5e7` | not started | 0% | train, val, test |
| mask v1 | `decoder_lr5e7_boxonly` | not started | 0% | train, val, test |
| mask v1 | `encdec_low_frozenbn` | not started | 0% | train, val, test |
| mask v1 | `encdec_low_trainbn` | not started | 0% | train, val, test |
| mask v2 | `A00_e2e_t4_box1_control` | not started | 0% | train, val, test |
| mask v2 | `A01_e2e_t4_box0` | not started | 0% | train, val, test |
| mask v2 | `A02_e2e_t4_official_prompt` | not started | 0% | train, val, test |
| mask v2 | `A03_decmem_t4` | not started | 0% | train, val, test |
| mask v2 | `A04_memory_t4` | not started | 0% | train, val, test |
| mask v2 | `A05_e2e_t8` | not started | 0% | train, val, test |
| mask v2 | `A06_e2e_t8_s4_t16_hard` | not started | 0% | hardness prep, train, val, test |
| mask v2 | `A07_e2e_t4_warmup5` | not started | 0% | train, val, test |
| mask v2 | `A08_e2e_t4_gb8` | not started | 0% | train, val, test |
| mask v2 | `A09_e2e_t4_hard50x2` | not started | 0% | hardness prep, train, val, test |
| mask v2 | `A10_e2e_t4_box0_imgkd` | not started | 0% | train, val, test |
| mask v2 | `A11_e2e_t4_box0_imgmemkd` | not started | 0% | train, val, test |
| SAM3.1 Stage 1 | `n1_cos000_adapter_ft_w2k` | val incomplete | 100% | full val and test |
| SAM3.1 Stage 1 | `n1_cos025_adapter_ft_w2k` | not started | 0% | train, val, test |
| SAM3.1 Stage 1 | `n1_cos100_adapter_ft_w2k` | training incomplete | 80% | resume train, val, test |
| SAM3.1 Stage 1 | `n2_adapter_cos025_frozen` | training incomplete | 20% | resume train, val, test |
| SAM3.1 Stage 1 | `n2_adapter_cos025_ft_w0` | not started | 0% | train, val, test |
| SAM3.1 Stage 1 | `n2_projection_cos025_ft_w2k` | val incomplete | 100% | full val and test |
| SAM3.1 Stage 1 | `n3_cos025_relation010_adapter_ft_w2k` | not started | 0% | train, val, test |
| SAM3.1 Stage 1 | `n3_cos150_adapter_ft_w2k` | training incomplete | 20% | resume train, val, test |
| SAM3.1 Stage 1 | `n3_relation010_adapter_ft_w2k` | not started | 0% | train, val, test |
| SAM2.1 Stage 1 | `tv21_adapter_sam21l_msehr_cos025` | not started | 0% | train, val, test |
| SAM2.1 Stage 1 | `tv21_proj_sam21bplus_msehr` | not started | 0% | train, val, test |

## Superseded Historical Rows

These rows remain in the universal CSV for provenance but are not recovery
targets. `Last state` records what existed before applying `superseded`.

| Historical experiment | Last state | Progress | Replacement |
| --- | --- | ---: | --- |
| `sam31_stage1/tv21m_adapter_mse_cos025_5ep_v1` | not started | 0% | `sam31_stage1_ablation_v1` |
| `sam31_stage1/tv21m_adapter_mse_cos025_5ep_v1_smoke` | val incomplete | 100% | `sam31_stage1_ablation_v1` |
| `stage1_online_teacher_sav000_018_vbal32_tv11m_4gpu_b16_mse_cos_5ep_v1` | val incomplete | 100% | `sav_stage1_ablation_v2` |
| `stage1_online_teacher_sav000_018_vbal32_tv11m_8gpu_b8_mse_only_5ep_v1` | val incomplete | 100% | `sav_stage1_ablation_v2` |
| `stage1_online_teacher_sav000_018_vbal32_tv21m_4gpu_b4_highres_only_5ep_v1` | training incomplete | 40.02% | `sav_stage1_ablation_v2` |
| `stage1_online_teacher_sav000_018_vbal32_tv21m_4gpu_b4_mse_cos_5ep_v1` | training incomplete | 30% | `sav_stage1_ablation_v2` |
| `stage1_online_teacher_sav000_018_vbal32_tv21m_8gpu_b4_mse_only_5ep_v1` | training incomplete | 50% | `sav_stage1_ablation_v2` |
| `stage1_online_teacher_sav000_018_vbal32_tv5m_4gpu_b32_mse_cos_5ep_v1` | val incomplete | 100% | `sav_stage1_ablation_v2` |
| `stage1_online_teacher_sav000_018_vbal32_tv5m_4gpu_b32_mse_only_5ep_v1` | val incomplete | 100% | `sav_stage1_ablation_v2` |

## Research Signals

1. Memory tuning is the largest consistent task-fine-tuning gain. In v1,
   stage 2 to stage 3 changes val J&F from 69.9 to 71.5 and test J&F from 71.3
   to 74.3 while preserving image metrics. In v2, the corresponding changes
   are 70.0 to 71.7 and 71.3 to 73.9.
2. Decoder-only task tuning is nearly neutral. V2 stage 1 to stage 2 changes
   test mIoU from 0.8353 to 0.8356 and leaves test J&F at 71.3. This motivates
   the prompt/scope/temporal ablations rather than longer decoder-only tuning.
3. The Stage 1 image/tracking objectives are Pareto, not scalar. HR weight 0.25
   gives the highest test AP, L1 0.25 gives a stronger image/tracking balance,
   and plain MSE+HR retains the strongest Stage 1 test J&F among those three.
4. Encoder capacity matters monotonically in the completed projection runs:
   TV21 is stronger than TV11, which is stronger than TV5 on image metrics and
   J&F. Adapter variants do not yet show a consistent advantage over projection.

## Recovery Scope

The formal remaining scope is 31 resumable pipelines: 11 registered SAM2 or
SAM3.1 Stage 1 runs, 2 RepViT runs, 6 mask-ablation v1 runs, and 12
mask-ablation v2 runs. The three task-fine-tuning stages in both v1 and v2 are
already complete and are skipped.

Seven old `stage1_online_teacher_sav000_018_*` directories, the SAM3.1 smoke
run, and the old standalone SAM3.1 run remain visible in the universal report
with status `superseded` and are excluded from recovery. They are historical
legacy/smoke artifacts, not missing cells in the current formal matrices.

Three foreground recovery lanes are defined in
`scripts/company/46_run_remaining_experiment_lane.sh`. A failed job is logged
and later jobs continue, maximizing weekend utilization. Every formal job
uses its existing directory and W&B ID when resuming, then runs full SA-V val
and test image/VOS evaluation.

| Lane | Formal pipelines | Allocation |
| --- | ---: | --- |
| `node1` | 10 | SAM3.1 interface/relations lane 4, two mask v1, three mask v2 |
| `node2` | 8 | Stage 1 lane 5, both RepViT sizes, two mask v1, one mask v2 |
| `node3` | 13 | Stage 1 lanes 1-3, two mask v1, eight mask v2 plus hardness preparation |

The unequal pipeline counts account for the longer Stage 1 and RepViT jobs on
nodes 1 and 2. Each lane writes per-job logs and a final universal CSV under
`/user-volume/remaining_experiment_logs/<lane>`.
