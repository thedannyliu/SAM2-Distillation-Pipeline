# TV21 EdgeTAM Memory Adaptation with an Online SAM2.1-L Teacher

## Research question

Can the selected TinyViT-21M SAM2.1 task model replace its four-block dense
SAM2 memory path with the released EdgeTAM two-block compressed-memory
architecture, retain at least 95% of full-validation tracking quality, and
provide an end-to-end latency improvement on Thor?

This is a company adaptation of EdgeTAM, not an exact reproduction of the
foundation-scale training run. The first version changes only the memory
architecture. It does not shrink the mask decoder, add learned object slots,
share K/V across objects, quantize the model, or add Thor-specific training
operators.

The method source is the
[EdgeTAM paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_EdgeTAM_On-Device_Track_Anything_Model_CVPR_2025_paper.pdf),
the released implementation is
[`facebookresearch/EdgeTAM`](https://github.com/facebookresearch/EdgeTAM), and
the training framework is the released
[`facebookresearch/sam2` trainer](https://github.com/facebookresearch/sam2/blob/main/training/README.md).

## Confirmed scope

| Item | Decision |
| --- | --- |
| First student | TinyViT-21M only |
| Later students | Transfer the validated recipe to TinyViT-11M and TinyViT-5M |
| Student image path | Selected TV21 encoder, projection/adapters, prompt encoder, and matched mask decoder |
| Memory path | Official EdgeTAM topology |
| Frozen teacher | SAM2.1 Hiera Large |
| Training data | Complete prepared SA-V train release only |
| Image distillation | Sample annotated SA-V frames online; do not write teacher-feature caches |
| Training hardware | One company node with 4 H100 GPUs |
| Training budget | Approximately 2--4 days, conditional on capacity-probe throughput |
| Quality evaluation | First-frame GT-mask VOS and company box/point prompts |
| Deployment target | Thor; H100 latency is diagnostic only |

## Fixed artifacts and storage

The existing SA-V staging and selected checkpoint remain read-only
reproducibility inputs:

```text
Full train manifest:
/group-volume/danny-dataset/sam2_distill/manifests/sav_train_6fps_full.parquet

SA-V root:
/group-volume/danny-dataset/SA-V

Selected TV21 task checkpoint:
/group-volume/danny-dataset/sam2_distill/runs/tinyvit_max_jf_v1/tv21/main/checkpoints/best.pt

TinyViT initializer:
/group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
```

New large and intermediate artifacts use Danny's project area in the shared
company volume:

```text
Run root:
/group-volume/danny-dataset/sam2_distill/runs/edgetam_tv21_sam21l_v1

Teacher checkpoint:
/group-volume/danny-dataset/sam2_distill/checkpoints/sam2.1/sam2.1_hiera_large.pt

EdgeTAM initializer:
/group-volume/danny-dataset/sam2_distill/checkpoints/edgetam/edgetam.pt

TensorBoard:
/group-volume/danny-dataset/sam2_distill/logs/edgetam_tv21_sam21l_v1

Final selected export:
/group-volume/danny-dataset/sam2_distill/checkpoints/final_weights/edgetam_tv21_sam21l_v1

Foreground terminal logs:
/user-volume/log/edgetam_tv21_sam21l_v1
```

`/group-volume/danny-dataset/...` is both the current reproducibility root and
the writable location for new large experiment artifacts. No new teacher
feature, image, or frame cache is created by this experiment.

## Baselines and known evidence

The selected TV21 standard-memory model is the primary quality baseline:

| Model | SA-V box val J&F | SA-V box test J&F |
| --- | ---: | ---: |
| Selected TV21, four-block SAM2 memory | 72.4 | 74.7 |

The selection is recorded in
[`backbone_task_expansion_v2.md`](../experiments/backbone_task_expansion_v2.md).

Existing compressed-memory evidence prevents repeating known failures:

| Experiment | Val/test J&F | Conclusion |
| --- | --- | --- |
| Strict TinyViT encoder + official EdgeTAM non-image transplant | 2.1 / 2.4 | Official prompt/mask decoder weights do not consume the selected TinyViT representation without alignment. |
| Full-SA-V EM3 curriculum | 44.4 / 45.8 | Image mIoU remained about 0.83, so the failure is temporal state propagation, not general image segmentation. |
| Paper-scaled SA-V-only Q2 | 55.6 / 58.0 | The best existing available-data compressed candidate is the minimum result the new run must exceed. |
| Functional TV21 four-block M0 | 71.5 / 74.3 | A matched TinyViT temporal path can preserve strong tracking behavior. |

See
[`edgetam_tinyvit21_behavior_v4.md`](../experiments/edgetam_tinyvit21_behavior_v4.md),
[`tinyvit21_edgetam_memory_v1.md`](../experiments/tinyvit21_edgetam_memory_v1.md),
and
[`edgetam_memory_ablation_v1.md`](../experiments/edgetam_memory_ablation_v1.md).

## Target model contract

The student retains the selected TV21 image/prompt/mask interface and replaces
the complete temporal path coherently.

| Component | Target |
| --- | --- |
| Input | 1024 x 1024 |
| Image feature | 256 x 64 x 64 |
| High-resolution features | 32 x 256 x 256 and 64 x 128 x 128 |
| Memory encoder output | 64 x 64 x 64 |
| Memory-attention layers | 2 |
| Global Perceiver latents | 256 |
| Local 2D Perceiver latents | 256, arranged as 16 x 16 |
| Local memory window | 4 x 4 over a 64 x 64 memory map |
| Perceiver depth/heads | 2 / 1 |
| Perceiver self-attention | Enabled |
| Frame memory bank | 7 |
| Maximum object pointers | 16 |
| Temporal layout | Released EdgeTAM layout and pointer flags |

The initialization must load the whole temporal contract together: memory
encoder, Perceiver, both attention blocks, pointer projections, temporal
embeddings, and no-memory/no-object embeddings. Importing only attention and
Perceiver tensors is not a valid initialization.

## Success criteria

Model selection uses full SA-V validation only. Test metrics are descriptive
and are run only for the final selected checkpoint.

| Signal | Functional gate | Production promotion gate |
| --- | ---: | ---: |
| SA-V box full-val J&F | at least 60 | at least 68.8 |
| Quality retention vs TV21 72.4 | descriptive | at least 95% |
| Image mIoU drop | no collapse | no more than 0.005 |
| Image AP drop | no collapse | no more than 0.005 |
| Identity errors | inspect | no swaps or missing objects |
| Thor single-object end-to-end speed | no regression | at least 1.2x baseline |
| Thor 4/8-object P50/P90 | report | positive gain at both counts |

The 68.8 threshold is `0.95 * 72.4`. A candidate between 60 and 68.8 is a
functional research result but does not replace the selected four-block model.

## Phase 0: environment, contract, and capacity gate

### Hypothesis

The released EdgeTAM graph, selected TV21 checkpoint, online Hiera-L teacher,
and SA-V task path can run together without a checkpoint, shape, gradient, or
memory error.

### Checks

1. Keep container PyTorch 2.4 and run a SAM2 compatibility smoke.
2. Record repository commits and SHA256 checksums for every checkpoint.
3. Strict-load the selected TV21 image/prompt/mask path.
4. Instantiate the official-layout EdgeTAM temporal path.
5. Run T1, T4, T8, and T16 forward/backward smokes.
6. Confirm the Hiera-L teacher is frozen and excluded from DDP, optimizer, and checkpoints.
7. Confirm every intended student module receives finite, nonzero gradients.
8. Confirm each unique image is encoded once and reused across sampled objects.
9. Measure real per-GPU batch capacity before fixing learning rates.

### Batch-capacity policy

The probe measures the actual planned graph rather than a backbone-only
approximation:

| Probe | Graph and loss | Candidate per-GPU batch |
| --- | --- | --- |
| `image` | T1, TV21 encoder/mask decoder, online Hiera-L F16 KD | 8, 4, 2 |
| `t4` | T4, frozen image/mask path, train EdgeTAM memory, online Hiera-L memory/logit KD | 4, 2, 1 |
| `t8` | T8 joint student, online Hiera-L image/memory/logit KD | 2, 1 |
| `t16` | T16 temporal-only, no teacher or KD | 2, 1 |

The temporal probes use the planned three-object and seven-correction-point
workload. The image probe uses up to eight objects to stress its planned
multi-object task loss. Every probe uses a bounded number of SA-V videos and
W&B is disabled. Each rank records steady step time and CUDA peak
allocated/reserved memory.

Accept a candidate only when:

- all four ranks finish;
- at least three post-warmup steps were measured;
- peak reserved memory on every GPU is at most 72 GiB;
- throughput is finite and no NaN, OOM, or disconnected-gradient error occurs.

The selected batch is the passing candidate with the highest measured global
samples/second, not automatically the largest batch. Logical dataloader batch,
global DDP batch, and image-encoder microbatch must be recorded separately.

The probe entry point is:

```text
scripts/company/71_probe_edgetam_tv21_batch.sh
```

Its generated summary is:

```text
/group-volume/danny-dataset/sam2_distill/runs/edgetam_tv21_sam21l_v1/
  batch_probe/v1/batch_probe_summary.csv
```

### 2026-08-05 capacity results

All candidates completed on four NVIDIA H100 80GB HBM3 GPUs. The acceptance
ceiling was 72 GiB peak reserved HBM per GPU.

| Stage | Batch/GPU | Global batch | T | Step s | Samples/s | Peak reserved GiB | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| image | 2 | 8 | 1 | 0.732 | 10.923 | 9.262 | yes |
| image | 4 | 16 | 1 | 1.125 | 14.223 | 12.361 | yes |
| image | 8 | 32 | 1 | 1.967 | 16.269 | 20.127 | yes |
| image | 16 | 64 | 1 | 3.730 | 17.160 | 36.584 | yes |
| image | 24 | 96 | 1 | 5.456 | 17.596 | 44.980 | yes |
| image | 32 | 128 | 1 | 7.076 | 18.090 | 59.682 | yes |
| T4 | 1 | 4 | 4 | 0.582 | 6.867 | 9.023 | yes |
| T4 | 2 | 8 | 4 | 0.842 | 9.501 | 16.758 | yes |
| T4 | 4 | 16 | 4 | 1.529 | 10.464 | 30.375 | yes |
| T4 | 6 | 24 | 4 | 1.959 | 12.253 | 39.750 | yes |
| T4 | 8 | 32 | 4 | 2.557 | 12.514 | 49.609 | yes |
| T4 | 10 | 40 | 4 | 3.233 | 12.372 | 73.217 | no |
| T8 | 1 | 4 | 8 | 1.291 | 3.098 | 15.428 | yes |
| T8 | 2 | 8 | 8 | 2.202 | 3.633 | 26.584 | yes |
| T8 | 3 | 12 | 8 | 2.943 | 4.077 | 45.312 | yes |
| T8 | 4 | 16 | 8 | 3.646 | 4.389 | 51.316 | yes |
| T8 | 5 | 20 | 8 | 4.371 | 4.576 | 63.883 | yes |
| T8 | 6 | 24 | 8 | 5.512 | 4.354 | 73.760 | no |
| T16 | 1 | 4 | 16 | 0.905 | 4.422 | 25.607 | yes |
| T16 | 2 | 8 | 16 | 1.329 | 6.019 | 52.635 | yes |

### Locked batch decision

| Stage | Batch/GPU | True global | Updates/pass | Train h/pass | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| image | 32 | 128 | 394 | 0.77 | Exact published image global batch; 20 GiB physical HBM margin |
| T4 | 6 | 24 | 2,098 | 1.14 | Batch 8 is only 2.1% faster, so retain more HBM and optimizer updates |
| T8 | 4 | 16 | 3,147 | 3.19 | Batch 5 is only 4.3% faster and has 12.6 GiB more reserved HBM |
| T16 | 2 | 8 | 6,293 | 2.32 | 36% faster than batch 1; batch 3 is projected beyond the safe ceiling |

The time column is the measured steady-step extrapolation and excludes full
validation, startup, checkpoint I/O, and dataloader tail effects.

Video stages do not accumulate gradients to effective global batch 256 in the
first run. With 50,337 usable SA-V videos, batch 256 gives only about 197
optimizer updates per pass: T4 two passes would have about 394 updates and T8
three passes about 591, versus the published EdgeTAM run's roughly 130K T8
updates on a multi-dataset mixture. Matching only the batch while omitting the
published step/data exposure would under-train the new temporal interface.
There is also no verified accumulation/resume/scheduler path in the current
trainer. The formal run therefore uses the measured true DDP batch, no
accumulation, and batch-scaled learning rates.

### Approved formal run matrix

These settings are approved. The full run remains conditional on the input
audit and one bounded forward/backward update for every stage passing on the
company container.

| Stage | Initializer | Trainable modules | Batch/GPU | Passes | Start LR | KD weights | Gate |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| image | selected TV21 + frozen official temporal | image encoder + matched mask decoder | 32 | 1 | encoder `1e-6`, decoder `5e-6` | image 1 | image mIoU/AP drop <= 0.005 |
| T4 | image winner or selected TV21 + `official_temporal` | complete EdgeTAM temporal path | 6 | 2 | temporal `2.8125e-5` | memory 1, logits 1 | full-val box J&F >= 55 |
| T8 | T4 + `current_full` | image encoder + mask decoder + complete temporal path | 4 | 3, optionally 5 | encoder `6e-7`, decoder `2e-6`, temporal `1.875e-5` | image 1, memory 1, logits 1 | full-val box J&F >= 60; promotion >= 68.8 |
| T16 | selected T8 + `current_full` | complete EdgeTAM temporal path | 2 | 1, optionally 2 | temporal `9.375e-6` | none | full-val J&F >= 60 and no more than 0.3 drop from T8 |

Every listed LR uses cosine decay to one tenth of its start value. All stages
use BF16 AdamW, weight decay 0.1, gradient clip 0.1, frozen BatchNorm, task
weights focal/mask 20, Dice 1, IoU 1, class 1, seed `250107256`, and online
SAM2.1-L teacher tensors only where a KD weight is nonzero. T4/T8 use up to
three objects, 50/50 point/box prompts, 10% GT correction sampling, two random
correction frames, and seven correction clicks. The image stage uses up to
eight objects. Object-pointer KD and gradient accumulation remain disabled.

The dedicated foreground entry point is
`scripts/company/72_run_edgetam_tv21_sam21l_v1.sh`. It provides `describe`,
`audit`, a bounded one-update-per-stage `smoke`, individual stage actions,
`all`, `test`, and `status`. The formal runner includes the separate T8
mask-decoder LR group and explicit video augmentation in each resolved config.
It reuses the same W&B ID and directories on resume. The image gate may fall
back to the original selected TV21 checkpoint; T4/T8/T16 gates stop downstream
training when they fail.

### Formal outcome: 2026-08-07 artifact audit

The dependency gates stopped the formal run after T4, as designed.

| Stage | State | Epoch / updates | Val mIoU | Val AP | Val J&F | J | F | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETV1 image re-anchor | train + full val | 1 / 393 | 0.8403 | 0.7166 | 21.2 | 20.8 | 21.6 | pass, image-only non-blocking gate |
| ETV2 T4 bootstrap | train + full val | 2 / 4,194 | 0.8403 | 0.7172 | 47.3 | 44.5 | 50.1 | **fail**, J&F below 55 |
| ETV3 T8 joint | not started | - | - | - | - | - | - | blocked by T4 gate |
| ETV4 T16 refine | not started | - | - | - | - | - | - | blocked by dependencies |

The reference checkpoint is 0.8371 mIoU, 0.7127 AP, and 72.4 J&F. ETV1
improves the two single-frame metrics but loses 51.2 J&F points, confirming
that image re-anchoring does not repair the replaced temporal interface. ETV2
then recovers 26.1 J&F points without damaging image quality, which proves
that task, memory-output, and propagated-logit supervision can partially adapt the
EdgeTAM temporal path. Its final 47.3 J&F nevertheless remains 25.1 points
below the reference and 7.7 points below the continuation gate.

Only the absolute-J&F check failed at T4; the image mIoU/AP guardrails passed.
This is a temporal compatibility failure, not an encoder or single-frame mask
failure. No formal test result exists because test access correctly depended
on ETV4 passing validation. The bounded smoke checkpoints for ETV3/ETV4 prove
only that those graphs execute; they are not trained formal candidates.

The first TV21 EdgeTAM recipe is therefore stopped. Running T8/T16 by
overriding the gate would invalidate the pre-registered decision rule. A
follow-up needs a different temporal recovery mechanism or initialization,
not merely the unfinished downstream schedule.

## Phase 1: online SA-V image re-anchoring

### Hypothesis

A short online Hiera-L feature-distillation pass can preserve or improve the
selected TV21 image interface before the temporal path is replaced. Because
the source checkpoint is already distilled and task-tuned, this is a bounded
re-anchoring stage rather than foundation pretraining.

### Data

- Randomly select one annotated frame per sampled SA-V video in the dataloader.
- Decode/read the frame online and discard teacher activations after the step.
- Use manual SA-V masks for the image task loss.
- Do not write image copies or teacher feature caches.
- Use horizontal flip and 1024-square resize.
- Cap sampled objects at the available per-frame objects, with 8 as the hard maximum for this bounded SA-V re-anchor.

### Trainable modules

- TinyViT encoder, projection/adapters, and neck.
- Matched mask decoder at the non-encoder learning rate.
- Freeze prompt encoder, memory encoder, Perceiver, attention, and pointer path.
- Freeze all BatchNorm statistics.

### Objective

```text
L_image = L_task
        + 1.0 * MSE(F16_student, F16_teacher)
```

Task weights use focal/mask 20, Dice 1, and IoU L1 1. The selected source
checkpoint already received high-resolution feature supervision during Stage
1, so this bounded re-anchor does not add a second high-resolution loss. This
keeps the probe and formal image-stage graph aligned with EdgeTAM's reported
F16 KD objective.

### Schedule and gate

- Run one complete SA-V pass: 394 optimizer updates at global batch 128.
- BF16 AdamW, weight decay 0.1, L2 clip 0.1, cosine decay, 5% warmup.
- Use encoder LR `1e-6 -> 1e-7` and mask-decoder LR `5e-6 -> 5e-7`.
- Require the online training F16 MSE to decrease without a task-loss spike.
- Require image mIoU and AP drops no greater than 0.005.

If the gate fails, discard this checkpoint and initialize Phase 2 directly
from the original selected TV21 checkpoint. Image re-anchoring is not a
dependency that can block the temporal experiment.

## Phase 2: T4 compressed-memory bootstrap

### Hypothesis

The complete official EdgeTAM temporal initialization can adapt to the fixed,
healthy TV21 image/mask interface when task, memory-output, and propagated-mask
supervision are applied before joint tuning.

### Initialization and trainability

- Image/prompt/mask path: Phase 1 winner or the original selected TV21 checkpoint.
- Temporal path: coherent released EdgeTAM temporal initializer
  (`TASK_MEMORY_INITIALIZER=official_temporal`).
- Freeze image encoder, prompt encoder, mask decoder, and BatchNorm.
- Train memory encoder, Perceiver, two attention blocks, pointer projections,
  temporal embeddings, and no-memory/no-object parameters.

### Sampling

```text
frames per clip: 4
objects per video: up to 3
point/box probability: 0.5 / 0.5
sample correction from GT probability: 0.1
random correction frames: 1--2
correction points: 7
reverse-time probability: 0.5
```

### Objective

```text
L_T4 = 1.0 * L_task
     + 1.0 * MSE(F_M_student, F_M_teacher)
     + 1.0 * BCE(student propagated logits,
                 sigmoid(teacher propagated logits))
```

Object-pointer KD remains disabled unless Phase 0 proves that the online
Hiera-L teacher emits a matching pointer for every required prompt/frame path.
If enabled later, use cosine loss with weight 0.1 and record it as an ablation.

### Schedule and gate

- Per-GPU batch 6, true global batch 24, and no gradient accumulation.
- Two complete SA-V passes: approximately 4,196 optimizer updates.
- BF16 AdamW, weight decay 0.1, L2 clip 0.1, cosine decay, 10% warmup.
- Use the same LR for memory attention, memory encoder/pointer/temporal
  parameters, and Perceiver, decayed tenfold:

```text
3e-4 * 24 / 256 = 2.8125e-5 -> 2.8125e-6
```

- Run full SA-V box validation after the stage.
- Require J&F at least 55 to continue; the formal functional gate remains 60.
- Require image mIoU/AP drops no greater than 0.005.

The 55 continuation boundary ensures the new stage at least approaches the
existing Q2 result instead of spending T8 compute on a known-broken path.

## Phase 3: T8 paper-style joint adaptation

### Hypothesis

After the temporal-only bootstrap, low-LR joint adaptation can close the
remaining image-to-memory representation gap without destroying the matched
image/mask interface.

### Trainable modules

- TinyViT encoder, projections/adapters, and neck.
- Matched mask decoder.
- Complete EdgeTAM temporal path.
- Freeze prompt encoder and BatchNorm statistics.

### Objective

```text
L_T8 = 1.0 * L_task
     + 1.0 * MSE(F16_student, F16_teacher)
     + 1.0 * MSE(F_M_student, F_M_teacher)
     + 1.0 * propagated-mask-logit KD
```

The first formal run does not add compressed-token, contrastive, or new
architecture losses. It tests the smallest objective supported by existing
working interfaces and the EdgeTAM recipe.

### Augmentation

Match the published video recipe as closely as the upstream transforms allow:

- horizontal flip;
- affine rotation 25 degrees and shear 20;
- clip-consistent color jitter 0.1;
- grayscale probability 0.05;
- 1024-square resize/crop.

The available upstream transform stack has no probability wrapper for a
second per-frame color jitter. The first formal run omits that one transform
rather than applying jitter to every frame and silently changing its meaning.
This is recorded as a remaining augmentation difference from the paper.

### Schedule and learning rates

- Initialize every module from the selected T4 checkpoint with
  `TASK_MEMORY_INITIALIZER=current_full`; do not reapply the released temporal
  initializer.
- Per-GPU batch 4, true global batch 16, and no gradient accumulation.
- Run three SA-V passes first.
- Run a fixed mini-val integrity gate after pass 1 and pass 3.
- Run full validation after pass 3.
- Extend to five passes only if full-val J&F is at least 58 and still improving.
- Use cosine decay, 10% warmup, BF16, weight decay 0.1, and clip 0.1.
- Scale the temporal LR from the published EdgeTAM batch-256 recipe:

```text
temporal LR = 3e-4 * 16 / 256
            = 1.875e-5 -> 1.875e-6
```

Do not apply that temporal LR to the already selected image path. The
successful TV21 continuation used global batch 4 with encoder
`1.5e-7 -> 1.5e-8` and other LR `5e-7 -> 5e-8`. Scaling those rates from
global batch 4 to 16 gives encoder `6e-7 -> 6e-8` and matched mask decoder
`2e-6 -> 2e-7`. This requires a separate mask-decoder optimizer group rather
than letting it inherit the memory-auxiliary default. Three passes are
approximately 9,441 optimizer updates. The optional extension to five passes
adds about 6,294 updates.

The fixed mini-val is an integrity check, not a model-selection set. Previous
32-video gates varied materially; only full validation selects a checkpoint.

### Interpretation

| Full-val J&F | Decision |
| ---: | --- |
| below 55 | interface/training failure; stop |
| 55--60 | weak compressed memory; diagnose, do not promote |
| 60--68.8 | functional research candidate; quality below production gate |
| at least 68.8 | 95% quality retention; proceed to formal Thor benchmark |

## Phase 4: conditional T16 refinement

T16 is conditional because the completed EM curriculum degraded after its
long-context stage.

- Enter only if the selected T8 model reaches at least 60 full-val J&F.
- Freeze image encoder, prompt encoder, mask decoder, and BatchNorm.
- Train only the EdgeTAM temporal path.
- Disable all distillation, following EdgeTAM progressive fine-tuning.
- Use task losses only.
- Keep the memory bank at seven frames.
- Initialize the complete model from the selected T8 checkpoint with
  `TASK_MEMORY_INITIALIZER=current_full`.
- Use per-GPU batch 2, true global batch 8, and no gradient accumulation.
- Run one SA-V pass; extend to two only when full validation improves.
- Use temporal LR `9.375e-6 -> 9.375e-7`, half the T8 temporal LR.

The automated gate accepts T16 only if full-val J&F is at least 60 and does
not drop by more than 0.3 from T8. The fixed
fast-motion/occlusion/disappearance cohort remains a promotion diagnostic: it
must improve before deployment promotion, but it does not block the first
full-test measurement because that cohort gate is not yet implemented in the
driver. T32 is outside the first few-day run and is considered only after T16
passes.

## Evaluation matrix

Every metric must identify checkpoint, model config, prompt protocol, seed,
split, and evaluator commit.

### Paper-comparable protocol

- First-frame GT mask.
- Full SA-V validation for candidate selection.
- Full SA-V test only for the selected checkpoint.
- Report J, F, and J&F.
- Add DAVIS, MOSE, and YTVOS only when those datasets become available.

### Stage evaluation order

The first formal pipeline runs full SA-V box-prompt validation after image,
T4, T8, and T16. Validation metrics and gates are written before the next
stage begins. Full SA-V test is run only after T16 passes its full-validation
gate, so test data never selects a checkpoint. The paper-comparable GT-mask
and company point-prompt protocols remain required in the final evaluation
matrix but do not gate this first training launch.

The driver enforces these dependencies for both `all` and individual actions:
T8 requires a passing T4 gate, T16 requires a passing T8 gate, and `test`
requires a passing T16 gate. The image stage is the only non-blocking stage;
on failure, T4 uses the original selected TV21 checkpoint.

### Company prompt protocol

- First-frame box.
- First-frame one-click.
- Optionally report three-click and five-click without changing selection.
- Report image mIoU/AP, video J/F/J&F, empty masks, missing objects, identity
  swaps, and seconds/video.

### Baselines

Run both protocols on:

1. selected TV21 with four-block dense memory;
2. released EdgeTAM;
3. T4 checkpoint;
4. best T8 checkpoint;
5. T16 checkpoint, if promoted.

## Thor benchmark and deployment gate

Thor is the only final efficiency judge. Compare the four-block baseline and
compressed candidate with the same encoder, prompt/mask decoder, input,
precision, videos, prompts, object IDs, and frame range.

Measure separately:

1. PyTorch model-only latency;
2. exported TensorRT model-only latency;
3. end-to-end decode, preprocess, model, and mask-output latency.

For 1, 2, 4, and 8 objects, use at least 50 warmup frames, 500 measured
frames, and three repeats. Record P50/P90/P99, FPS, component latency, peak
memory, CPU utilization, power, temperature, throttling, first-prompt latency,
and propagation latency.

TensorRT correctness compares the same checkpoint against PyTorch and requires
per-mask IoU at least 0.99 with no shape, order, or object-ID change. Model
quality remains the full SA-V evaluation; two different trained models are not
expected to be bit-exact.

The current Thor handoff and runtime boundary are documented in
[`sam2_tinyvit_multiobject_thor.md`](../deployment/sam2_tinyvit_multiobject_thor.md).

## Tracking and resumability

Use W&B project `sam2-edgetam-tv21-sam21l-v1` plus TensorBoard. Each formal
stage uses one resumable W&B ID, the same TensorBoard directory, and the same
checkpoint directory across retries.

```text
/group-volume/danny-dataset/sam2_distill/runs/edgetam_tv21_sam21l_v1/formal/
  ETV{1,2,3,4}_*/main/
    resolved_config.yaml
    checkpoints/
      last.pt
      best.pt
    tensorboard/
    wandb/
    sav_val_box_benchmark/
      metrics.csv
    gate_status.json
  ETV4_t16_refine_1ep/main/sav_test_box_benchmark/
    metrics.csv

/user-volume/log/edgetam_tv21_sam21l_v1/formal/
  describe.log
  smoke.log
  train.log
```

Retain only the resumable `last.pt` and validation-selected `best.pt`; do not
write a dense step-checkpoint series. Log global batch, clips/frames seen,
epoch/pass, ETA, step time, data time, LR per group, gradient norm per module,
peak HBM, task loss, F16 loss, memory loss, logit loss, and validation metrics.

Operational order is `describe`, `smoke`, then `all`. The `smoke` action uses
a separate `formal/smoke` run root, disables W&B, and performs exactly one
four-GPU optimizer update at each formal batch shape. The `all` action repeats
the read-only input audit, resumes completed work by default, performs the
four train/val/gate stages in dependency order, and ends with the single T16
test evaluation. Use `status` for a read-only checkpoint and metric summary.

## Estimated four-H100 budget after the capacity probe

| Phase | Measured/extrapolated time |
| --- | ---: |
| Environment, contract, and capacity probe | complete |
| Online image re-anchoring, one pass | 0.77 h training |
| T4 bootstrap, two passes | 2.28 h training |
| T8 three passes | 9.56 h training |
| Optional T8 extension to five passes | additional 6.37 h training |
| Conditional T16, one pass | 2.32 h training |
| Full validation, dual-protocol final evaluation, startup, and export | not measured by the probe |
| Initial image + T4 + T8 three-pass training | approximately 12.6 h before evaluation |

The extrapolation uses the slowest-rank mean step time and
`ceil(50337 / true_global_batch)` updates per pass. Allow at least 25--35%
additional wall time for startup, dataloader tails, checkpointing, and
validation. The full end-to-end sequence is therefore expected to fit within
roughly one to two days on one four-H100 node, but only measured stage status
files may be reported as actual duration.

## Differences from the original EdgeTAM training

| Item | Original EdgeTAM | Company TV21 adaptation |
| --- | --- | --- |
| Image encoder | ImageNet-pretrained RepViT-M1 | Distilled and task-tuned TinyViT-21M |
| Prompt/mask weights | Learned in the EdgeTAM/SAM2 pipeline | Matched weights from the selected TV21 checkpoint |
| Memory architecture | Two blocks plus 256 global and 256 local Perceiver latents | Same architecture |
| Memory bank/pointers | 7 / 16 | Same |
| Teacher | Public SAM2 Hiera-B+ | SAM2.1 Hiera-L |
| Image data | SA-1B | Online annotated frames from SA-V |
| Image schedule | About 175K steps, global batch 128 | One SA-V pass (394 updates) from a strong warm start |
| Video data | SA-V, 10% SA-1B, DAVIS, MOSE, and YTVOS according to Table 5 | Complete SA-V only |
| Video schedule | T8 about 130K steps, global batch 256 | T4 two passes then T8 three-to-five passes |
| Progressive stage | T16 about 43K and T32 about 43K steps | Conditional T16 one-to-two passes; no first-round T32 |
| Image KD | F16 MSE | F16 MSE; the source checkpoint already received high-resolution Stage 1 KD |
| Video KD | F16 MSE and memory-attention output MSE | F16, memory output, and propagated-mask-logit KD |
| Object-pointer KD | Not reported | Disabled unless its target contract is verified |
| Batch | Image global 128; video global 256 | Image global 128; video true globals 24/16/8 with no accumulation |
| LR | Published fixed values | Published values linearly scaled by measured global batch |
| Augmentation | Full published image/video recipe | Image stage is conservative; T8 matches the video recipe |
| Validation | VOS/PVS/SA across multiple datasets | GT-mask and box/point on SA-V initially |
| Deployment target | iPhone 15 Pro Max and A100 | Thor |
| Teacher features | Caching policy not reported | Online only; no persistent teacher cache |

The paper is internally ambiguous about the video dataset mix: its
implementation table lists five datasets while another paragraph describes
SA-V/SA-1B-only training. The released EdgeTAM repository does not include the
paper-scale training launcher or mixture weights. This experiment therefore
must be described as an EdgeTAM-architecture, paper-inspired company
adaptation, not an exact training reproduction.

The paper body also reverses focal and Dice weights in one sentence. Table 5
and the upstream SAM2 convention use focal/mask 20 and Dice 1; the company run
uses that convention.

## Decision after TV21

Only after TV21 meets the quality and Thor gates should the exact validated
schedule, losses, batch-selection procedure, and evaluation matrix be copied
to TV11M and TV5M. The smaller models receive independent Thor benchmarks;
parameter count or H100 throughput is not used as a proxy for Thor latency.
