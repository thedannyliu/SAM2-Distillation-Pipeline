# Key Results through 2026-07-28

Source: company all-experiment reports generated from
`/group-volume/danny-dataset/sam2_distill/runs`. The 2026-07-28 03:42 UTC
snapshot contains 140 rows: 94 complete, 3 final-checkpoint incomplete,
1 finalization incomplete, 27 not started, 1 training incomplete,
2 training-state unknown, 3 validation incomplete, and 9 superseded
historical rows. The three final-checkpoint-incomplete rows are EdgeTAM
recovery C0-C2; they completed their configured training but did not
produce full val/test results. Two validation-incomplete rows are A01
smoke checks rather than formal experiments. Model decisions use
`sav_val`; `sav_test` is held-out and descriptive only.

All 12 mask-v2 rows and all eight EdgeTAM-memory rows completed train -> full
SA-V val -> full SA-V test. EdgeTAM recovery C0-C2 completed their configured
training but did not proceed to full val/test. The remaining
`finalization_incomplete` row is RepViT-M09, whose observed accuracy is not
competitive.

## 2026-07-28 EdgeTAM Recipe Diagnostics

| Run | Model | val mIoU | val AP | val J&F | test J&F |
| --- | --- | ---: | ---: | ---: | ---: |
| Q0 official identity | released RepViT-M1 EdgeTAM, low-LR temporal epoch | 0.8224 | 0.6864 | **68.3** | **69.5** |
| Q1 overfit16 | TinyViT-21M + official temporal stack, 2k subset updates | 0.8404 | 0.7166 | 15.5 | 15.5 |

Q0 rules out a universal trainer/checkpoint/evaluator failure: the exact
official model remains at the released checkpoint's 68.0-val-J&F level after
local training. Q1 retains excellent prompted-image metrics but does not
generalize temporal behavior from its fixed 16-video training subset. The
full-val result does not establish whether Q1 memorized its training videos;
that requires its W&B loss curves or a train-subset VOS measurement.

Q2 has no completed checkpoint in this snapshot. Its initial launch failed
before an optimizer step on an optional teacher-output assumption that is now
fixed, so it is not negative evidence about the paper-scaled recipe.

Together with E1, M2, and R0-R3, these controls identify the encoder-to-memory
representation contract as the primary unresolved EdgeTAM variable. With two
available nodes, the next formal block is:

- same-TinyViT-interface EdgeTAM compression: K1b/c and K2b/c;
- TinyViT-5M continuation: matched GT-only versus SAM2.1-L soft pseudo-mask
  weights 0.25/0.50, followed by a val-selected T8 refinement.

The first lane tests whether behavior transfer becomes learnable after
removing the RepViT/TinyViT interface boundary. The second asks whether the
5M model's current 66.0 val J&F is a supervision ceiling rather than a capacity
ceiling. The official-interface W curriculum remains documented and deferred.

## 2026-07-27 Main conclusions

1. TinyViT fine-tuning is near a data/model ceiling, not ineffective.
   The best full-val J&F is now 66.0/68.6/72.4 for 5M/11M/21M. Weekend
   continuation adds only 0.2/0.1/0.0 over the previous winners, confirming
   saturation under the current data and objective.
2. RepViT is highly recoverable but remains behind TinyViT. Task
   fine-tuning raises val J&F from 37.1 to a val-selected 61.4 (+24.3).
   Decoder/memory adaptation is the best controlled fork, but the result
   remains 4.6 points below TinyViT-5M.
3. BatchNorm should stay frozen. RepViT train-BN loses 2.5 test J&F,
   0.0472 mIoU, and 0.0743 AP against its frozen-BN control, confirming
   the earlier TinyViT signal.
4. Coherent task-only EdgeTAM compression is learnable but not yet viable.
   K1a reaches 38.7 val J&F and its T8 continuation K2a reaches 42.1 while
   preserving image metrics. This remains 29.4 points behind M0 and below
   the pre-registered 60-J&F threshold; behavior-loss forks are still
   unstarted.

The highest selected TinyViT-21M result is 72.4 val J&F. Its selected
test result is 74.7; the unselected S1 row reaches 74.8 test J&F but is
not selected because its val J&F is lower at 72.2.

The four-node weekend block and its partial results are specified in
`docs/experiments/weekend_72h_v1.md`. TinyViT and RepViT completed; only the
task-only K branch completed on EdgeTAM. The official-transfer lane and the
same-interface behavior-loss branches remain outstanding.

## 2026-07-27 Weekend Results

### Current val-selected backbone frontier

| Backbone | Selected run | val mIoU | val AP | val J&F | test J&F |
| --- | --- | ---: | ---: | ---: | ---: |
| TinyViT-5M | weekend W1 decoder/memory | 0.7997 | 0.6425 | **66.0** | 67.6 |
| TinyViT-11M | weekend W1 decoder/memory | 0.8141 | 0.6717 | **68.6** | 70.5 |
| TinyViT-21M | pre-weekend max-J&F winner | 0.8371 | 0.7127 | **72.4** | 74.7 |
| RepViT-M0.9 | weekend W3 decoder/memory | 0.7596 | 0.5733 | **61.4** | 60.0 |

The weekend TinyViT gains are +0.2/+0.1/0.0 val J&F. T8 continuations have
higher test values in several cases but do not win on validation. The formal
RepViT result is W3 at 61.4 val J&F; W6's 61.1 test J&F does not supersede
W3 because its val J&F is only 60.9.

### EdgeTAM task-only compression control

| Run | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K1a T4 task-only, 5 epochs | 0.8404 | 0.7165 | 38.7 | 0.8389 | 0.7190 | 39.0 |
| K2a T8 task-only, 2 epochs | 0.8404 | 0.7160 | **42.1** | 0.8389 | 0.7190 | **40.0** |

K2a improves over K1a by 3.4 val and 1.0 test J&F without damaging image
quality. It still trails M0 by 29.4/34.3 val/test J&F and M1 by 11.2/16.1.
K1b-K1d and K2b-K2d are required before concluding whether same-interface
behavior distillation can close the remaining gap. W1-W3 official-transfer
experiments are also not started.

## 2026-07-24 EdgeTAM Official Fidelity and Behavior Transfer

The unmodified released EdgeTAM checkpoint passed two independent local
32-video SA-V val gates:

| Run | mIoU | AP | J&F |
| --- | ---: | ---: | ---: |
| official E0, primary gate | 0.8344 | 0.7229 | 65.2 |
| official E0, seed-2 gate | 0.8456 | 0.7451 | 72.1 |
| official E0, full val | 0.8224 | 0.6862 | 68.0 |

The local evaluator is functional. C0/C1 near 31 J&F must therefore be
attributed to the hybrid interface/behavior transfer, not a universal
evaluation failure. The next formal suite uses a strict E1 transplant:
A02 contributes only `image_encoder.*`; the official EdgeTAM checkpoint
contributes every non-image tensor.

The v4 suite compares staged image-then-temporal adaptation, joint
adaptation, and a temporal-from-scratch control. In addition to task, image,
and final memory-feature losses, v4 directly distills propagated mask logits
and object pointers. The scratch control randomizes the Perceiver, memory,
and object-pointer path while retaining the mature image/prompt/mask model;
it does not mislabel a whole-model random initialization as feasible under
SA-V-only data. See
`docs/experiments/edgetam_tinyvit21_behavior_v4.md`.

The parallel backbone expansion protects each distilled TinyViT encoder
before a low-LR joint pass, and gives the weak RepViT-M0.9 checkpoint an
encoder-recovery curriculum plus a controlled frozen-BN/train-BN fork. See
`docs/experiments/backbone_task_expansion_v2.md`.

### Completed v4 controls

| Run | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official E0 full val | 0.8224 | 0.6862 | 68.0 | - | - | - |
| `E1_a02_official_nonimage` | 0.0200 | 0.0001 | 2.1 | 0.0151 | 0.0000 | 2.4 |
| `S0_scratch_temporal_task_2ep` | 0.8402 | 0.7165 | 15.6 | 0.8389 | 0.7190 | 13.8 |

E1 proves that replacing only the official image encoder with A02
TinyViT breaks the official prompt/mask interface even though every tensor
loads strictly. S0 retains the healthy A02 image/prompt/mask path, so its
strong image metrics and collapsed J&F isolate failure to learning the
random temporal stack. D1-D3, J1-J3, and S1-S2 were not started in this
snapshot; behavior distillation is therefore still untested.

## 2026-07-23 EdgeTAM Memory Results

All runs retain the TinyViT-21M image encoder and start from the selected A02
end-to-end checkpoint. M0-M2 isolate memory topology; R0-R3 test whether
end-to-end task/KD training can recover the compressed EdgeTAM path.

| Run | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `M0_sam2_mem4` | 0.8405 | 0.7167 | **71.5** | 0.8391 | 0.7191 | **74.3** |
| `M1_sam2_mem2` | **0.8406** | 0.7167 | 53.3 | 0.8391 | **0.7197** | 56.1 |
| `M2a_edgetam_hybrid2_official` | 0.8405 | 0.7166 | 15.6 | 0.8391 | 0.7190 | 12.8 |
| `M2b_edgetam_hybrid2_current` | **0.8406** | 0.7167 | 13.2 | 0.8391 | 0.7191 | 10.6 |
| `R0_edgetam_e2e_t4_task` | 0.8369 | 0.7096 | 23.0 | 0.8364 | 0.7137 | 21.5 |
| `R1_edgetam_e2e_t4_imgkd` | **0.8379** | **0.7121** | 23.6 | 0.8373 | 0.7165 | 21.7 |
| `R2_edgetam_e2e_t4_imgmemkd` | 0.8377 | 0.7117 | **25.3** | **0.8374** | **0.7167** | **23.2** |
| `R3_edgetam_e2e_t8_imgmemkd` | 0.8374 | 0.7114 | 21.9 | 0.8367 | 0.7157 | 19.1 |

The image metrics remain healthy while VOS collapses, which is strong evidence
of temporal-interface incompatibility rather than a general TinyViT or mask
decoder failure. M1 loses 18.2 J&F on both splits versus M0, showing that
two-layer truncation alone is already destructive. Official M2 attention
initialization is slightly better than retaining the A02 layers, but both are
unusable. Memory KD is the best recovery signal (+1.7 val, +1.5 test J&F over
R1), while the eight-frame run is worse.

Keep M0 as an uncompressed continued-memory baseline. Stop the present M1/M2/R
line as production candidates and do not spend the next compute block on a
broad hyperparameter sweep. First require a 20-50-video temporal compatibility
gate, coherent full-temporal-stack initialization, and frozen-module memory
distillation to recover at least 60 mini-val J&F. The detailed causal analysis
and next protocol are in `docs/experiments/edgetam_memory_ablation_v1.md`.

The implemented recovery protocol is
`docs/experiments/edgetam_memory_recovery_v2.md`. It uses M0 as a
same-TinyViT/decoder teacher, compares coherent versus partial temporal
initialization, and compares staged versus joint training at equal two-epoch
data exposure. All candidates must pass a fixed 32-video validation gate before
full val/test.

### EdgeTAM recovery v2 gate results

| Run | Initialization | Gate mIoU | Gate AP | Gate J&F | J&F vs M0 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| M0 gate reference | functional four-layer memory | 0.852800 | 0.756663 | 71.6 | 0.0 | reference |
| `C0_coherent_m0mem_align` | coherent official temporal | 0.852409 | 0.755670 | **31.5** | -40.1 | gate failed |
| `C1_partial_m0mem_align` | v1 partial/legacy | **0.852822** | **0.756625** | 31.1 | -40.5 | gate failed |

Both candidates completed one full SA-V epoch and passed the image guardrails,
but neither approached the 60 J&F compatibility threshold. Coherent transfer
improves tracking by only 0.4 over partial transfer, so incomplete official
initialization was not the dominant cause of collapse. Same-interface M0
distillation on final memory-conditioned `F_M` is also insufficient: it
preserves prompted-image quality while leaving temporal propagation broken.

C2 subsequently completed its configured training but, like C0/C1, did
not produce full val/test metrics and is classified
`final_checkpoint_incomplete`. C3 remains not started. Without C2 gate or
full-evaluation metrics, it cannot be interpreted as recovery. The next
experiment should first localize per-frame decay and then supervise
intermediate memory tokens, object pointers, or attention behavior rather
than relying only on final-feature MSE.

The next protocol is now split into a gated fidelity ladder rather than
another hybrid sweep. `docs/experiments/edgetam_fidelity_v3.md` first evaluates
the unmodified released EdgeTAM checkpoint under the same SA-V evaluator
(`E0`), then permits a zero-training TinyViT-21M encoder swap only if upstream
fidelity passes. C3 remains paused.

## Completed size-specific maximum-J&F fine-tuning

`docs/experiments/tinyvit_max_jf_v1.md` defines one independent four-H100 lane
for TV5M, TV11M, and TV21M. Each lane includes its strongest existing
checkpoint in the full-val ranking, freezes BatchNorm, uses the A02
official-style prompt simulation, and evaluates every trained candidate on
full SA-V val/test with W&B. TV5M/TV11M run encoder adaptation followed by
joint and decoder-memory stages; TV21M continues from A02 at lower learning
rates rather than duplicating completed work. Selection uses full-val J&F
only.

| Capacity | Selected run | val J&F | test J&F |
| --- | --- | ---: | ---: |
| TinyViT-5M | weekend W1 decoder/memory | **66.0** | 67.6 |
| TinyViT-11M | weekend W1 decoder/memory | **68.6** | 70.5 |
| TinyViT-21M | max-J&F selected | **72.4** | 74.7 |

The weekend continuation raises the validation-selected 5M and 11M results
by only 0.2 and 0.1 J&F, respectively. None of the three 21M weekend
candidates beats the existing 72.4-val selection. TV21 F2 is now complete
at 71.7 val and 74.4 test J&F, also below the selected 72.4/74.7.

### RepViT task recovery

| Stage | val J&F | test J&F |
| --- | ---: | ---: |
| distill-only | 37.1 | 37.5 |
| P1 encoder recovery | 58.6 | 57.4 |
| P2 frozen-BN joint | 59.7 | 59.5 |
| P2b train-BN joint | 58.1 | 57.0 |
| P3 T8 decoder/memory | 60.3 | 60.1 |
| Weekend W3 decoder/memory | **61.4** | 60.0 |
| Weekend W6 joint-low | 60.9 | **61.1** |

The result supports encoder recovery followed by frozen-BN joint and
temporal refinement. W3 is the formal validation-selected winner, improving
P3 by 1.1 val J&F. W6 has the largest observed test J&F but is not selected
because it reaches only 60.9 on validation. The lane continues to reject
trainable BN for this data/batch regime.

## 2026-07-22 Mask Fine-Tuning Results

### Mask v2: completed

All variants use the same Stage 1 task-fine-tuned starting point. Status is
kept explicit so these rows are not misrepresented as fully closed pipelines.

| Variant | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `A00_e2e_t4_box1_control` | 0.8376 | 0.7131 | 71.6 | 0.8356 | 0.7164 | 73.7 | complete |
| `A01_e2e_t4_box0` | 0.8377 | 0.7132 | 71.3 | 0.8355 | 0.7161 | 73.7 | complete |
| `A02_e2e_t4_official_prompt` | 0.8374 | 0.7129 | **72.0** | 0.8347 | 0.7151 | 74.1 | complete |
| `A03_decmem_t4` | 0.8377 | 0.7134 | 71.8 | 0.8357 | 0.7167 | 73.4 | complete |
| `A04_memory_t4` | 0.8373 | 0.7128 | 71.7 | 0.8353 | 0.7160 | 73.8 | complete |
| `A05_e2e_t8` | 0.8375 | 0.7132 | 71.9 | 0.8355 | 0.7164 | **74.3** | complete |
| `A06_e2e_t8_s4_t16_hard` | 0.8377 | 0.7138 | 71.4 | 0.8357 | 0.7169 | 73.9 | complete |
| `A07_e2e_t4_warmup5` | 0.8374 | 0.7131 | 71.3 | 0.8356 | 0.7163 | 73.9 | complete |
| `A08_e2e_t4_gb8` | 0.8374 | 0.7133 | 71.9 | 0.8355 | 0.7165 | 73.4 | complete |
| `A09_e2e_t4_hard50x2` | **0.8379** | 0.7137 | 70.8 | 0.8357 | 0.7166 | 72.9 | complete |
| `A10_e2e_t4_box0_imgkd` | 0.8377 | 0.7134 | 71.3 | **0.8361** | 0.7167 | 72.8 | complete |
| `A11_e2e_t4_box0_imgmemkd` | 0.8378 | **0.7143** | 71.3 | 0.8358 | **0.7168** | 73.3 | complete |

Relative to A00, the largest validation signals are orthogonal rather than one
variant winning every metric:

- A02 improves val J&F by 0.4 while reducing val mIoU/AP by 0.0002 each.
- A05 improves val J&F by 0.3 and matches the best observed test J&F, but its
  val mIoU is 0.0001 lower.
- A09 improves val mIoU by 0.0003 and AP by 0.0006 but loses 0.8 J&F.
- A11 improves val mIoU by 0.0002 and AP by 0.0012 but loses 0.3 J&F.
- A06 gains 0.0007 val AP but loses 0.2 val J&F, so the hard T16 refinement
  does not justify its extra complexity in this run.

### Mask v1: completed

| Variant | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `decoder_lr2e6` | 0.8374 | 0.7127 | 70.3 | 0.8352 | 0.7161 | 70.6 |
| `decoder_lr2e7` | 0.8375 | 0.7128 | 70.0 | 0.8356 | 0.7161 | 71.5 |
| `decoder_lr5e7` | 0.8376 | 0.7128 | 70.2 | 0.8354 | 0.7162 | 71.4 |
| `decoder_lr5e7_boxonly` | 0.8376 | 0.7128 | **70.3** | **0.8357** | 0.7161 | **71.7** |
| `encdec_low_frozenbn` | 0.8374 | 0.7127 | 69.9 | 0.8355 | 0.7163 | 71.2 |
| `encdec_low_trainbn` | 0.8343 | 0.7039 | 69.2 | 0.8347 | 0.7143 | 71.2 |

Decoder-only mask tuning is tightly clustered and does not improve over the
task-fine-tuned checkpoints. Training the encoder with trainable BN is clearly
harmful to image metrics, supporting frozen normalization for later E2E work.

### Newly completed Stage 1 result

| Variant | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tv21_adapter_sam21l_msehr_cos025` | 0.8347 | 0.7078 | 70.5 | 0.8359 | 0.7171 | 74.2 |

Adding cosine 0.25 to the TV21 adapter improves its earlier plain-adapter
result substantially on image metrics and test tracking, but it remains below
task fine-tuning on the validation objectives used for selection.

### Current cross-family validation Pareto set

| Run | val mIoU | val AP | val J&F | Interpretation |
| --- | ---: | ---: | ---: | --- |
| EdgeTAM suite M0 | **0.8405** | **0.7167** | 71.5 | strongest image metrics; uncompressed memory |
| TinyViT-21M max-J&F | 0.8371 | 0.7127 | **72.4** | strongest selected validation tracking |
| Mask v2 A02 | 0.8374 | 0.7129 | 72.0 | strongest one-pass mask-v2 tracking |
| Mask v2 A11 | 0.8378 | 0.7143 | 71.3 | strongest mask-v2 val AP |

M0 remains the image-metric leader but is not a compressed EdgeTAM model.
Longer low-LR TinyViT-21M fine-tuning raises selected val J&F from A02's
72.0 to 72.4 without improving the image objectives. Prompt simulation,
hard sampling, and KD move different objectives; the EdgeTAM result shifts
the next experiment from another mask-finetuning grid to temporal
compatibility and behavior distillation.

## 2026-07-22 Historical Pipeline Status

The original 31-run recovery launch has produced 17 newly evaluated or
completed formal results: all 12 mask-v2 variants, four mask-v1 variants, and
`tv21_adapter_sam21l_msehr_cos025`. Fourteen formal pipelines still require
compute or evaluation; the two A01 smoke directories are excluded.

| Category | Runs | Action |
| --- | ---: | --- |
| Mask v2 finalization only | 12 | sync existing val/test metrics to W&B, update summary, remove marker |
| Formal not started | 8 | train -> val -> test |
| Formal training incomplete | 3 | resume same checkpoint and W&B run -> val -> test |
| Formal validation incomplete | 3 | finish val -> test |
| A01 smoke validation incomplete | 2 | no action; retain as smoke provenance |
| Superseded history | 9 | no action |

The 14 formal compute/evaluation rows are both RepViT runs, two remaining mask
v1 runs, and nine SAM3.1 runs. `tv21_proj_sam21bplus_msehr` is the remaining
SAM2.1 Stage 1 not-started row. The current 3-lane processes should be allowed
to resume these directories; none of the 12 mask-v2 training jobs should be
launched again.

### Selected continuation on three 4-H100 nodes

After reviewing the completed mask results, the broad recovery queue is
reduced to experiments that are already near completion, close a necessary
causal comparison, or provide a missing architecture/teacher control.
`scripts/company/48_run_selected_continuation_lane.sh` distributes them by
remaining training load:

| Lane | Training/resume work | Evaluation/finalization work |
| --- | --- | --- |
| `selected1` | frozen-BN mask-v1 control; SAM2.1-B+ Stage 1 control | W&B-only A00-A03 finalization |
| `selected2` | SAM3.1 cosine-0.25 anchor | full eval for cosine 0 and 1; W&B-only A04-A07 finalization |
| `selected3` | RepViT-M09 final 20%; SAM3.1 frozen-adapter final 80% | full eval for SAM3.1 projection; W&B-only A08-A11 finalization |

Mask-v2 `finalize` validates and reuses the existing passing full val/test
metrics, syncs them to the original W&B run, updates the summary, and removes
`.full_eval_required`; it does not rerun training or GPU evaluation. RepViT
M23, decoder LR 2e-6, SAM3.1 warmup-0/cosine-1.5/relation variants, and new
T16/warmup/batch sweeps remain paused pending the selected results and W&B
curve audit. Do not run the selected lanes concurrently with the old recovery
lanes because the Stage 1 runners share checkpoint directories.

## 2026-07-21 Baseline Pareto Leaders

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

## 2026-07-21 Completed Baseline Results

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

## 2026-07-21 Launch-Time Incomplete Experiments

These 31 rows were the formal recovery scope before the 3-lane launch.
`Progress` is checkpoint progress reported in that snapshot, not evaluation
progress. Refer to the 2026-07-22 pipeline status above for current state.

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

## 2026-07-21 Recovery Launch

The formal remaining scope is 31 resumable pipelines: 11 registered SAM2 or
SAM3.1 Stage 1 runs, 2 RepViT runs, 6 mask-ablation v1 runs, and 12
mask-ablation v2 runs. The three task-fine-tuning stages in both v1 and v2 are
already complete and are skipped.

Seven old `stage1_online_teacher_sav000_018_*` directories, the SAM3.1 smoke
run, and the old standalone SAM3.1 run remain visible in the universal report
with status `superseded` and are excluded from recovery. They are historical
legacy/smoke artifacts, not missing cells in the current formal matrices.

The actual company launch used the earlier three-lane allocation from commit
`c68e7a8`; all 31 formal recovery pipelines are already queued on those three
nodes. Do not launch the later six-lane rebalance concurrently. A failed job is
logged and later jobs continue. Every formal job uses its existing directory
and W&B ID when resuming, then runs full SA-V val and test image/VOS
evaluation.

| Lane | Formal pipelines | Allocation |
| --- | ---: | --- |
| `node1` | 10 | SAM3.1 interface/relations lane 4, two mask v1, three mask v2 |
| `node2` | 8 | Stage 1 lane 5, both RepViT sizes, two mask v1, one mask v2 |
| `node3` | 13 | Stage 1 lanes 1-3, two mask v1, eight mask v2 plus hardness preparation |

Each recovery lane writes per-job logs and a final universal CSV under
`/user-volume/remaining_experiment_logs/<lane>`.

Stage 1 and RepViT runs disable periodic step checkpoints and retain only
`last.pt` and `best.pt`. The upstream SAM2 task trainer requires its native
names: `checkpoint.pt` is the resumable last state and `stage.pt` is the final
export used for validation/test; no periodic task checkpoints are retained.

## Priority Mask Fine-Tuning Pull-Forward

The recovery queues place the mask experiments after long Stage 1/RepViT
work. Three additional nodes should therefore pull forward the nine highest
information mask-v2 experiments without launching a second copy of the full
31-run scope. The first concurrent wave is the exact-box causal KD trio:

| Variant | Only material difference | Question |
| --- | --- | --- |
| `A01_e2e_t4_box0` | no KD | exact-box E2E control |
| `A10_e2e_t4_box0_imgkd` | image KD 0.5 | does interface anchoring stabilize E2E? |
| `A11_e2e_t4_box0_imgmemkd` | image KD 0.5 + memory KD 0.25 | is the temporal interface an additional bottleneck? |

With three additional nodes, the follow-up allocations are:

| Priority lane | Sequence | Research comparison |
| --- | --- | --- |
| `priority1` | A01 → A00 → A02 | correction/prompt simulation |
| `priority2` | A10 → A03 → A04 | image KD, then decoder/memory trainable scope |
| `priority3` | A11 → hardness → A05 → A06 | memory KD, T8, and T16 hard refinement |

When five additional nodes are available, use the final two lanes to pull the
remaining optimization/data-selection variants forward as well:

| Priority lane | Sequence | Research comparison |
| --- | --- | --- |
| `priority4` | A07 → A08 | warmup and global-batch stability |
| `priority5` | hardness → A09 | hard-example sampling at two epochs |

Priority 5 starts the shared resumable hardness mining early; priority 3 later
reuses the same fingerprinted outputs for A05/A06. The priority runner is
`scripts/company/47_run_priority_mask_finetune_lane.sh`. Per-variant `flock`
locks prevent the priority and recovery nodes from writing the same checkpoint
directory concurrently; a later recovery invocation waits, then observes and
skips the completed run. All priority runs require online W&B, execute
train → full val → full test, and update the shared mask `summary.csv`.
