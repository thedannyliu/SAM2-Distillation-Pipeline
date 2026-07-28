# EdgeTAM Recipe Diagnostics v5

## Question

Why does the released EdgeTAM checkpoint retain useful SA-V video behavior,
while TinyViT-21M models trained through the local task pipeline collapse to
roughly 10--42 J&F? These experiments separate three hypotheses: trainer drift,
an unlearnable temporal path, and insufficiently faithful training choices.

All rows use four H100 GPUs, W&B, frozen BatchNorm, and the same full SA-V
box-prompt val and test benchmarks used by the existing experiment report.
Q0 and Q2 use seed `250107256`; Q1 uses seed `2501072` because the upstream
SAM2 trainer multiplies the seed by `max_epochs=500` and NumPy requires the
result to fit in 32 bits. Each pipeline runs train -> val -> test and retains
only `last.pt` and `best.pt`.

## Experiments

| Run | Model and initialization | Training | Falsifiable interpretation |
|---|---|---|---|
| `Q0_official_identity_t8_1ep` | Exact released RepViT-M1 EdgeTAM graph and `edgetam.pt` | T8, one SA-V epoch; only memory attention, memory encoder, 2D Perceiver, object-pointer path; temporal LR `3e-8`--`3e-7` | If video J&F collapses from the unmodified official baseline, the local training loop/objective changes a functioning checkpoint. If it remains near baseline, the loop can preserve EdgeTAM. |
| `Q1_tinyvit_overfit16_t8_500ep` | TinyViT-21M A02 image/mask path plus coherently initialized official temporal path | Fixed deterministic 16-video subset, T8, 500 epochs (about 2k optimizer steps); task + memory + propagated-logit losses | Failure to drive losses down on 16 videos means the TinyViT/temporal interface or supervision is broken. Successful overfit but weak full val/test indicates generalization/data-recipe failure. |
| `Q2_tinyvit_paper_scaled_sav_t8_5ep` | TinyViT-21M A02 plus official temporal initialization; SAM2.1 Hiera-B+ teacher | Full eligible SA-V T8 split, five epochs, three objects, seven correction clicks, `dice=20`, `focal=1`, weight decay `0.1`, and learning rates linearly scaled from batch 256 to batch 4 | Improvement over K/R rows supports recipe fidelity as the missing factor. No improvement after Q0/Q1 pass points to missing dataset mixture or capacity rather than optimizer details. |

## Scope Boundary

`Q2` is a paper-derived **available-data** experiment, not a full EdgeTAM
reproduction. The paper video stage mixes SA-V, a 10% SA-1B stream, DAVIS,
MOSE, and YouTube-VOS for 130k iterations, then uses longer T16/T32 stages.
Those datasets are not part of the current mounted training contract. The
official released model also uses RepViT-M1, while Q2 deliberately retains the
project's TinyViT-21M encoder. Therefore Q2 may test whether the feasible
recipe corrections help, but it cannot establish paper reproduction.

The project benchmark starts video tracking from a box prompt. It must not be
numerically equated with the paper's standard first-frame-mask VOS numbers.

## Selection

- Compare Q0 against the unmodified official checkpoint under the same local
  evaluator. A material drop (at least 3 J&F) is a trainer/objective alarm.
- For Q1, inspect raw and EMA W&B losses. Require a clear reduction in task,
  mask-logit, and memory losses. Object-pointer KD is disabled because the
  official teacher does not emit `obj_ptr` for this prompt path. Full val/test
  are diagnostic and are not used as an overfit success criterion.
- Promote Q2 only if full val J&F improves by at least 1 point over the strongest
  comparable compressed-memory run without more than 0.5-point mIoU/AP loss.
- Do not use test metrics for model selection.

## Driver and Outputs

- Driver: `scripts/company/49_run_edgetam_memory_ablation.sh`
- Run root:
  `/group-volume/danny-dataset/sam2_distill/runs/edgetam_recipe_diagnostics_v5`
- W&B project: `edgetam-recipe-diagnostics-v5`
- Summary:
  `/group-volume/danny-dataset/sam2_distill/runs/edgetam_recipe_diagnostics_v5/summary.csv`

Before allocating GPUs, inspect the fully resolved experiment controls:

```bash
scripts/company/49_run_edgetam_memory_ablation.sh describe Q0_official_identity_t8_1ep
scripts/company/49_run_edgetam_memory_ablation.sh describe Q1_tinyvit_overfit16_t8_500ep
scripts/company/49_run_edgetam_memory_ablation.sh describe Q2_tinyvit_paper_scaled_sav_t8_5ep
```
