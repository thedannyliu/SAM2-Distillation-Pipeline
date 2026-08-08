# EdgeTAM ETV Supervision Diagnostic v2

## Question

Does ETV2 fail because the compressed temporal path receives mismatched
prompts and a cross-backbone memory-feature target, rather than because the
released EdgeTAM modules, dataset, BatchNorm, or gradient clipping are wrong?

This is a bounded diagnostic. It does not continue ETV3 and does not use
`sav_test`.

## Corrections under test

1. Generate the first-frame prompt plan once and copy it into the frozen
   teacher forward. Teacher and student prompt tensors must match exactly.
2. Use no iterative correction clicks in this diagnostic. Those clicks depend
   on each model's prediction error and cannot be shared without an explicit
   external correction plan.
3. Apply memory and mask-logit KD only to propagated frames, excluding the
   initial conditioning frame.
4. Divide the summed task loss by the number of frames so its scale does not
   change with T. KD terms retain mean reduction.
5. Log four-rank mean losses, pre-clip gradient norm, cumulative clip
   fraction, and non-finite gradient count.
6. Keep the released EdgeTAM two-layer attention, 2D Perceiver, official
   temporal initializer, frozen BatchNorm, and clip norm `0.1` unchanged.

## Controlled experiment

All rows start from the same ETV1 image re-anchor checkpoint used by the
original ETV2 and the same released EdgeTAM temporal initialization. Each row
uses 12,000 usable SA-V train videos, T4, batch 6 per GPU on four GPUs, global
batch 24, and one pass:

`12000 / (6 * 4) = 500` optimizer updates exactly.

| Variant | Teacher | Temporal target | Purpose |
|---|---|---|---|
| `ETD0_task_only_500` | none | GT task only | Corrected-loss baseline |
| `ETD1_hiera_l_logits_500` | SAM2.1 Hiera-L | propagated mask logits | Test Hiera behavior KD without incompatible `F_M` MSE |
| `ETD2_m0_logits_500` | functional TV21 M0 | propagated mask logits | Test same-interface behavior KD |
| `ETD3_m0_memlogits_500` | functional TV21 M0 | propagated `F_M` + mask logits | Test whether same-interface memory alignment adds value |

The prompt contract uses a first-frame box and zero correction clicks for all
rows. KD rows fail immediately if the copied prompt outputs differ.

The driver first runs one update for every row under
`runs/edgetam_etv_supervision_v2/smoke` with W&B disabled. The formal
500-update rows start only after all four teacher/student interfaces pass.
Passing a variant as the second driver argument restricts smoke, training,
and mini-validation to that row, so four independent 4-GPU allocations can
run the rows concurrently. The per-variant pipeline locks and locked central
summary update keep their artifacts isolated.

## Evaluation and decisions

Every row runs a fixed 32-video `sav_val` mini-gate after training. The gate
set and reference are shared across variants. It is a screening measurement,
not a replacement for full validation.

Primary comparisons:

- `ETD1 - ETD0`: value of safe Hiera-L behavior KD;
- `ETD2 - ETD1`: effect of removing the Hiera/TV21 interface boundary;
- `ETD3 - ETD2`: incremental value of same-interface memory-feature KD.

Diagnostics must satisfy:

- exactly 500 optimizer updates;
- `prompt_match_rate = 1.0` on every logged KD step;
- zero non-finite gradients;
- finite gradient summary and resolved `reverse_time_prob = 0.5`;
- identical data, prompt, initialization, optimizer, and mini-val cohort
  across the four rows.

Promote at most the best two variants to full `sav_val`. Do not run
`sav_test` in this diagnostic. Do not continue T8 unless a promoted T4 model
reaches at least 55 full-val J&F; 68.8 J&F is the 95% retention target relative
to the selected 72.4 reference.

## Company paths

- Run root:
  `/group-volume/danny-dataset/sam2_distill/runs/edgetam_etv_supervision_v2`
- W&B project: `sam2-edgetam-etv-supervision-v2`
- Terminal logs: `/user-volume/log/edgetam_etv_supervision_v2`
- Driver: `scripts/company/73_run_edgetam_etv_supervision_v2.sh`

## Results

The first ETD3 smoke attempt on 2026-08-08 stopped before model
instantiation and before optimizer step 0 because that company container did
not have `hydra-core`. This is an environment failure, not an experimental
result, and produced no training checkpoint. Install `hydra-core==1.3.2` in
each affected container and rerun the same variant command; resumable paths
and W&B IDs remain unchanged. The driver now checks this dependency before
launching `torchrun`.

RandomAffine warnings are expected when a sampled affine would erase a
first-frame target. Upstream returns the unmodified datapoint for that affine
attempt and continues the remaining transforms. Keep this policy identical
for all four rows and compute the warning count from each terminal log after
500 steps rather than changing augmentation mid-screen.

| Variant | Updates | Prompt match | Grad mean/max | Clip fraction | Mini-val J&F | Status |
|---|---:|---:|---:|---:|---:|---|
| ETD0 | pending | n/a | pending | pending | pending | not run |
| ETD1 | pending | pending | pending | pending | pending | not run |
| ETD2 | pending | pending | pending | pending | pending | not run |
| ETD3 | pending | pending | pending | pending | pending | not run |
