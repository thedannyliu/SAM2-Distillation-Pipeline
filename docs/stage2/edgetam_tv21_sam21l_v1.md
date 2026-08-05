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

Formal learning rates and commands are intentionally deferred until the probe
results are reviewed.

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
- Matched mask decoder at one tenth of the encoder LR.
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

- Run 0.5 SA-V pass, then at most one full pass.
- BF16 AdamW, weight decay 0.1, L2 clip 0.1, cosine decay, 5% warmup.
- Start from approximately `1e-5 -> 1e-6` for the encoder and
  `1e-6 -> 1e-7` for the decoder; finalize after the batch probe.
- Require validation feature-MSE reduction.
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
- Temporal path: coherent released EdgeTAM temporal initializer.
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

- Two complete SA-V passes.
- BF16 AdamW, weight decay 0.1, L2 clip 0.1, cosine decay, 10% warmup.
- Paper-scaled starting temporal LR:

```text
3e-4 * actual_global_batch / 256
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
- color jitter 0.1;
- grayscale probability 0.05;
- per-frame color jitter probability 0.1;
- 1024-square resize/crop.

The resolved config must be inspected before launch because the existing task
config has weaker augmentation than the published EdgeTAM recipe.

### Schedule and learning rates

- Run three SA-V passes first.
- Run a fixed mini-val integrity gate after pass 1 and pass 3.
- Run full validation after pass 3.
- Extend to five passes only if full-val J&F is at least 58 and still improving.
- Use cosine decay, 10% warmup, BF16, weight decay 0.1, and clip 0.1.
- Scale published video LRs by the measured global batch:

```text
encoder LR = 6e-5 * actual_global_batch / 256
other LR   = 3e-4 * actual_global_batch / 256
decoder LR = 0.5 * other LR
```

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
- Run one SA-V pass; extend to two only when full validation improves.
- Start at half the T8 temporal LR.

Accept T16 only if overall J&F does not drop by more than 0.3 and the fixed
fast-motion/occlusion/disappearance cohort improves. T32 is outside the first
few-day run and is considered only after T16 passes.

## Evaluation matrix

Every metric must identify checkpoint, model config, prompt protocol, seed,
split, and evaluator commit.

### Paper-comparable protocol

- First-frame GT mask.
- Full SA-V validation for candidate selection.
- Full SA-V test only for the selected checkpoint.
- Report J, F, and J&F.
- Add DAVIS, MOSE, and YTVOS only when those datasets become available.

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
stage_name/
  resolved_config.yaml
  checkpoints/
    last.pt
    best.pt
  tensorboard/
  logs/
  val/
  training_status.json
```

Retain only the resumable `last.pt` and validation-selected `best.pt`; do not
write a dense step-checkpoint series. Log global batch, clips/frames seen,
epoch/pass, ETA, step time, data time, LR per group, gradient norm per module,
peak HBM, task loss, F16 loss, memory loss, logit loss, and validation metrics.

## Estimated four-H100 budget

| Phase | Provisional time |
| --- | ---: |
| Environment, contract, and capacity probe | 2--4 h |
| Online image re-anchoring | 2--6 h |
| T4 bootstrap and validation | 10--16 h |
| T8 three-to-five-pass training and validation | 24--40 h |
| Conditional T16 | 8--16 h |
| Final dual-protocol evaluation and export | 4--8 h |
| Total | approximately 2--4 days |

These values are provisional. The completed EM curriculum took about 34 hours
for nine SA-V passes on 4 H100s with a lighter teacher. The online Hiera-L
capacity probe is required before commands or wall-time claims are finalized.

## Differences from the original EdgeTAM training

| Item | Original EdgeTAM | Company TV21 adaptation |
| --- | --- | --- |
| Image encoder | ImageNet-pretrained RepViT-M1 | Distilled and task-tuned TinyViT-21M |
| Prompt/mask weights | Learned in the EdgeTAM/SAM2 pipeline | Matched weights from the selected TV21 checkpoint |
| Memory architecture | Two blocks plus 256 global and 256 local Perceiver latents | Same architecture |
| Memory bank/pointers | 7 / 16 | Same |
| Teacher | Public SAM2 Hiera-B+ | SAM2.1 Hiera-L |
| Image data | SA-1B | Online annotated frames from SA-V |
| Image schedule | About 175K steps, global batch 128 | At most 0.5--1 SA-V pass from a strong warm start |
| Video data | SA-V, 10% SA-1B, DAVIS, MOSE, and YTVOS according to Table 5 | Complete SA-V only |
| Video schedule | T8 about 130K steps, global batch 256 | T4 two passes then T8 three-to-five passes |
| Progressive stage | T16 about 43K and T32 about 43K steps | Conditional T16 one-to-two passes; no first-round T32 |
| Image KD | F16 MSE | F16 MSE; the source checkpoint already received high-resolution Stage 1 KD |
| Video KD | F16 MSE and memory-attention output MSE | F16, memory output, and propagated-mask-logit KD |
| Object-pointer KD | Not reported | Disabled unless its target contract is verified |
| Batch | Fixed published global batches | Largest throughput-efficient true batch on 4 H100s |
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
