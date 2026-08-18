# SAM2.1-L fixed-K reuse v1

## Registered questions

This suite separates two questions that the original SA-V R4 screen could not
answer because raw 24 FPS refreshes alias with 6 FPS annotations.

1. On the same videos and target frames, how do the already-trained O2/A--E
   checkpoints behave at each R4 feature age?
2. Does schedule-matched training of otherwise vanilla SAM2 improve over both
   frozen reuse and generic full-refresh fine-tuning at K=4/8/12/16/20?

No SA-V test data is used. All temporal units below are contiguous raw 24 FPS
frames decoded online from the original MP4.

## Eval30 protocol

The audit cohort is the first 30 IDs in the existing seed-250107256 hash order.
It is fixed without consulting model predictions. A separate, non-overlapping
10-video cohort is reserved for checkpoint selection in later work.

Current-model audit targets are O0, W0, O1 and checkpoint 1 of O2/A/B/C/D/E.
O0 and W0 run R1 once. Every R4 target runs four causal fixed phases, p=0--3.
For a given model and video, phase metrics are averaged before videos are
macro-averaged. This phase-neutral metric is primary. Anchor phase 0 is also
reported as the deployment-policy metric.

The report contains:

- official SA-V J/F/J&F for every phase;
- phase-neutral per-video macro J/F/J&F;
- paired J&F deltas versus O0 and frozen O1;
- 10,000 video-level paired bootstrap samples with a fixed seed and 95% CI;
- paired feature-age J/F/J&F;
- actual encoder refresh rate, model mean/P95, and wall latency;
- an exact decoded-mask O0/W0 identity audit.

Every fixed-phase source must satisfy `source_frame <= prediction_frame`.
Predictions from distinct phases live in distinct output directories and are
never overwritten or pooled before per-video accounting.

One dedicated one-H100 foreground process executes this audit strictly
sequentially: O0, W0, O1 phases 0--3, then O2/A/B/C/D/E phases 0--3. Each
model-phase result is committed to its own directory before the next starts.
Relaunch validates and skips complete units, resumes at the first missing unit,
and generates the detailed report only after attempting the whole sequence.
No two SAM2.1-L predictors are resident on the GPU concurrently.

## Fixed-K training matrix

Six independent runs start from the same official SAM2.1-L checkpoint:

| Run | Fixed refresh interval | Maximum age | Raw-frame clip length |
|---|---:|---:|---:|
| A1 | 1 | 0 | 8 |
| A4 | 4 | 3 | 8 |
| A8 | 8 | 7 | 9 |
| A12 | 12 | 11 | 13 |
| A16 | 16 | 15 | 17 |
| A20 | 20 | 19 | 21 |

All runs use experiment A: official per-frame spatial-memory writes, no feature
age conditioner, no recency/freshness residual, no stale-write suppression,
and no representation/pointer/score distillation. The only intended treatment
difference is the fixed refresh interval; the longer clip is the minimum
required to expose one complete interval and its next refresh. All models use
`max_feature_age=19` so their instantiated graphs and checkpoint interfaces are
identical.

The existing supervision and optimizer recipe is retained:

- group-normalized manual-GT loss plus 0.25 group-normalized pseudo-mask loss;
- frozen online SAM2.1-L teacher, supervision only, never teacher memory;
- student on-policy memory rollout;
- batch size one per GPU on four H100s;
- the existing AdamW/LR groups, LayerNorm weight-decay exclusion, bf16, and
  gradient clipping; no BatchNorm is introduced.

Each run trains for at most one full 50,337-video epoch. In addition to the
resumable epoch checkpoint, selection-only model snapshots are written after
0.10, 0.25, and 0.50 epoch. These fractional snapshots are not resume points.
W&B, TensorBoard, checkpoints, resolved config, gradient diagnostics, capacity
records, and launch metadata remain in one run directory per target.

Before formal launch, A20 must pass a T21 four-GPU capacity/gradient smoke. A
formal process is resumable from its epoch checkpoint and reuses its W&B run
ID and artifact directories.

## Comparisons and promotion

For each K, later evaluation uses the same deployment K for all three controls:

- `O1-K`: frozen official checkpoint with naive feature reuse;
- `A1@K`: generic full-refresh-trained control evaluated with reuse K;
- `AK@K`: schedule-matched training and inference.

The isolated matched-schedule gain is `AK@K - A1@K`; total gain over frozen
reuse is `AK@K - O1-K`. Exact phase-neutral evaluation averages all K phases
for a selected fixed-K checkpoint. A trained model is promoted only if its
phase-neutral gain over O1-K is at least 0.5 J&F without worse worst-age J&F.

If A-K fails at long intervals, a later D-K suite may test whether explicit
age/two-clock conditioning and stale-write suppression rescue the limitation.
D-K is not part of this first causal suite.

## Company artifacts

- Run root: `/group-volume/danny-dataset/sam2_distill/runs/sam21l_fixed_k_v1`
- W&B project: `sam2-fixed-k-v1`
- TensorBoard: within each run directory
- Terminal logs: `/user-volume/log/sam21l_fixed_k_v1`
- Eval30 root: `/group-volume/danny-dataset/sam2_distill/runs/sam21l_fixed_k_v1/eval30_current`

The company runner and final launch commands are added only after local
trajectory, config, report, shell, and smoke-contract tests pass.
