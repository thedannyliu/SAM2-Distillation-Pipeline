# TinyViT size-specific maximum-J&F fine-tuning v1

## Question

Given the current 50,337-video SA-V training split, what is the strongest
full-validation J&F attainable for each existing TinyViT-5M, TinyViT-11M,
and TinyViT-21M Stage 1 student?

This is an optimization suite, not a causal architecture comparison. Each
size starts from its own strongest existing validation-J&F checkpoint and
uses a size-aware learning rate. Model selection uses full SA-V val J&F only.
SA-V test is descriptive and never participates in ranking.

## Starting points

| Size | Starting run | Adapter | val J&F | test J&F |
| --- | --- | --- | ---: | ---: |
| TV5M | `tv5_adapter_sam21l_msehr` | residual depthwise adapter | 64.2 | 67.1 |
| TV11M | `tv11_proj_sam21l_msehr_cos025` | projection | 67.1 | 69.5 |
| TV21M | `A02_e2e_t4_official_prompt` | projection | 72.0 | 74.1 |

TV21M starts from A02 rather than repeating its already completed encoder-only
and first end-to-end passes. TV5M and TV11M have not received equivalent mask
task tuning, so they retain the encoder-adaptation stage.

## Evidence shaping the recipe

- Mixed official-style prompt simulation is the only completed mask-v2 change
  that clearly improves validation tracking: A02 gains 0.4 J&F over A00.
- T8, T16 hard refinement, hard-50 resampling, warmup, image KD, and memory KD
  do not improve validation J&F in the completed TV21M ablations.
- Trainable BatchNorm damages image mIoU and AP. All stages therefore freeze
  BatchNorm statistics and affine parameters.
- Decoder/memory-only tuning is retained as a final candidate rather than
  assumed to help. A full validation comparison can reject it without using
  test.

## Per-size pipelines

Every trained row uses four H100s, batch 1/GPU, bf16, gradient clipping at
0.1, cosine decay, 50/50 point-versus-box prompt sampling, 10% GT correction
sampling, two correction frames, up to seven correction clicks, two objects,
and the full training split.

### TV5M

| Candidate | Trainable modules | Frames | Epochs | Encoder LR | Other LR |
| --- | --- | ---: | ---: | ---: | ---: |
| S1 encoder adaptation | image encoder | 2 | 2 | `1e-6 -> 1e-7` | frozen |
| S2 joint task | encoder + decoder + memory | 4 | 1 | `3e-7 -> 3e-8` | `1e-6 -> 1e-7` |
| S3 temporal polish | decoder + memory | 4 | 1 | frozen | `5e-7 -> 5e-8` |

### TV11M

The stages match TV5M except S1 uses `8e-7 -> 8e-8` and S2 uses
`2.5e-7 -> 2.5e-8` for the encoder.

### TV21M

| Candidate | Initializer | Trainable modules | Frames | Epochs | Encoder LR | Other LR |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| S1 continuation | A02 | encoder + decoder + memory | 4 | 1 | `1.5e-7 -> 1.5e-8` | `5e-7 -> 5e-8` |
| S2 lower-LR continuation | S1 | encoder + decoder + memory | 4 | 1 | `7.5e-8 -> 7.5e-9` | `5e-7 -> 5e-8` |
| S3 temporal polish | S2 | decoder + memory | 4 | 1 | frozen | `2.5e-7 -> 2.5e-8` |

The lower TV21M learning rates test whether the current validation leader is
under-trained without repeating a high-LR pass that could erase it.

## Evaluation and retention

Each trained candidate executes:

`train -> full SA-V val -> full SA-V test -> W&B summary`

The pre-existing starting checkpoint is included in the validation ranking.
The driver writes `summary.csv` and `selection.json`, ranks by full-val J&F,
and breaks an exact tie with val mIoU and then val AP. Test is not a
tiebreaker.

Each training stage writes one resumable last trainer checkpoint and one
model-only evaluation export; no step checkpoints are enabled. The size-level
`main/checkpoints` contains `last.pt` and `best.pt` (`checkpoint.pt` is a
symlink to `best.pt`). Predicted masks are deleted after scoring. The added
storage is small relative to the 450 GB limit.

W&B project: `tinyvit-max-jf-v1`

Run roots:

- `/group-volume/danny-dataset/sam2_distill/runs/tinyvit_max_jf_v1/tv5`
- `/group-volume/danny-dataset/sam2_distill/runs/tinyvit_max_jf_v1/tv11`
- `/group-volume/danny-dataset/sam2_distill/runs/tinyvit_max_jf_v1/tv21`

Entry point:

`scripts/company/52_run_tinyvit_max_jf.sh tv5|tv11|tv21 all`

## Decision rules

- Primary success: at least +0.5 full-val J&F over the size's starting point.
- A gain below 0.3 is provisional and needs a second seed.
- Always report mIoU/AP alongside J&F; maximizing tracking must not conceal a
  severe prompted-image regression.
- If the baseline wins, retain it as best and record that additional task
  tuning did not help that size. Do not choose a checkpoint by test.
