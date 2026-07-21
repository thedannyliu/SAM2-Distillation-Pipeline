# Key Results: 2026-07-21

Source: company all-experiment report generated from
`/group-volume/danny-dataset/sam2_distill/runs`. The snapshot contains 59
rows: 19 complete, 26 not started, 7 training incomplete, and 7 validation
incomplete. Model decisions use `sav_val`; `sav_test` is reported only as a
held-out descriptive result.

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
but are excluded from recovery. They are superseded legacy/smoke artifacts,
not missing cells in the current formal matrices.

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
