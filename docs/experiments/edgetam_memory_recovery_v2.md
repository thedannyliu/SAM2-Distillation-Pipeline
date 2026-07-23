# EdgeTAM memory recovery v2

## Research question

Can a two-layer EdgeTAM 2D Spatial Perceiver memory path be made compatible
with the already functional TinyViT-21M SAM2 hybrid when the available video
training data is SA-V only?

This suite follows the failed v1 transfer; it does not repeat its broad
task/KD variants. The primary hypothesis is that compression was introduced
without a functional warm start, and one SA-V epoch was then asked to repair
several temporal interfaces simultaneously.

## Evidence from EdgeTAM

The [EdgeTAM paper](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_EdgeTAM_On-Device_Track_Anything_Model_CVPR_2025_paper.html)
identifies memory attention as a latency bottleneck in SAM2. Its default model
uses two memory-attention blocks and a 2D Spatial Perceiver with 256 global
and 256 spatial latents. Spatial latents matter because video segmentation is
a dense prediction problem. The
[official implementation](https://github.com/facebookresearch/EdgeTAM)
also changes the memory cross-attention and object-pointer temporal layout as
one coherent model.

The official video stage uses 1024-pixel inputs, eight-frame samples with
about three objects, task loss plus unit-weight image and memory feature MSE,
and SA-V + 10% SA-1B + DAVIS + MOSE + YTVOS for 130K iterations at batch 256.
It then progressively fine-tunes with 16 and 32 frames while freezing the
image encoder and removing distillation. Its released recipe uses
SAM2-Hiera-B+ as teacher; the v1 local reproduction instead used Hiera-L.

The paper's 43K-step ablations show that the architecture itself is not
expected to collapse: the uncompressed two-block RepViT baseline reaches
63.5/62.1 SA-V val/test J&F, global+2D Perceiver reaches 64.4/62.5, and
distillation raises it to 65.7/65.8. Two memory-attention blocks also beat one
and four in that controlled setting. Global-only and 2D-only latents are both
worse than their combination on validation, and Perceiver self-attention adds
1.8 val J&F. These results support reproducing the complete 256+256,
self-attending design instead of searching smaller latent counts now.

This scale is not reproducible with the current data budget:

| Exposure | Official video stage | This suite, one epoch | This suite, two epochs |
| --- | ---: | ---: | ---: |
| Video clips | 33,280,000 | 50,337 | 100,674 |
| Frames at T8/T4 | 266,240,000 | 201,348 | 402,696 |
| Relative frame exposure | 1x | about 1/1,322 | about 1/661 |

The comparison ignores repeated augmentation and dataset diversity, so it is
already favorable to this suite. A randomly or partially initialized temporal
path should not be expected to relearn the official model from SA-V alone.

## Diagnosis of v1

The v1 results isolate the failure:

| Run | Transfer | val J&F | test J&F | val mIoU |
| --- | --- | ---: | ---: | ---: |
| M0 | functional four-layer SAM2 memory | 71.5 | 74.3 | 0.8405 |
| M1 | first two A02 attention layers | 53.3 | 56.1 | 0.8406 |
| M2a | official Perceiver + attention only | 15.6 | 12.8 | 0.8405 |
| R2 | M2a + task/image/memory KD | 25.3 | 23.2 | 0.8377 |

Image quality remains healthy while tracking collapses. The problem is
therefore in temporal conditioning, not the TinyViT image encoder or basic
mask decoder.

Two implementation mismatches are material:

1. M2a imported `spatial_perceiver.*` and `memory_attention.*`, but retained
   A02/M0 `memory_encoder`, `maskmem_tpos_enc`, no-memory/no-object embeddings,
   and object-pointer projection.
2. The local A02 topology adds projected signed temporal encodings to object
   pointers and a spatial no-object embedding. Official EdgeTAM disables both.

The R variants also used SAM2.1 Hiera-L as teacher. Its different image
encoder and decoder make `F_M` alignment solve representation transfer and
memory compression at once.

## Recovery hypotheses

- **H1 — coherent transfer:** importing the complete official temporal stack
  and matching its pointer flags gives a better starting contract than M2a.
- **H2 — same-interface teacher:** M0 is a better functional teacher because
  student and teacher share TinyViT-21M, prompt encoder, and mask decoder.
- **H3 — alignment before task loss:** pure `F_M` alignment protects the
  compressed path from an unstable task gradient during its first epoch.
- **H4 — longer clips are downstream:** T8 is useful only after T4 temporal
  compatibility exists. R3 already showed that increasing T4 to T8 at a
  broken starting point loses 3.4 val J&F.

## Controlled experiments

All rows use 50,337 usable SA-V train videos, T4, exact first-frame box
prompting, at most two objects, frozen TinyViT-21M, frozen mask decoder,
frozen BatchNorm, and an online M0 teacher. The trained modules are the memory
encoder, two-layer memory attention, Perceiver, memory temporal embeddings,
and object-pointer projection. `Lmem` is fp32 MSE on memory-conditioned
features `F_M`. The paper prose and supplementary hyperparameter table disagree
on focal/Dice ordering; detailed Table 5 and the official SAM2 training config
both use focal 20 and Dice 1. This suite retains that implementation-backed
setting, with IoU and occlusion weights of 1.

| Run | Initializer/layout | Epochs | Objective | Causal question | Status |
| --- | --- | ---: | --- | --- | --- |
| `C0_coherent_m0mem_align` | coherent official temporal | 1 | `Lmem` | Can pure functional alignment cross the compatibility gate? | gate failed |
| `C1_partial_m0mem_align` | v1 partial/legacy | 1 | `Lmem` | At fixed M0 teacher, how much does coherent initialization matter? | gate failed |
| `C2_coherent_m0mem_joint2ep` | coherent official temporal | 2 | `Ltask + Lmem` | Does joint training work when given two equal data passes? | not started |
| `C3_coherent_m0mem_staged` | C0 checkpoint | 1 after C0 | `Ltask + Lmem` | Does one alignment epoch before one joint epoch beat C2? | blocked by C0 |

C2 and C0 -> C3 each see exactly two epochs, so the curriculum comparison is
not confounded by data exposure. C3 is blocked unless C0 passes its gate.

The coherent initializer imports these official tensors together:

- `memory_encoder.*`, `memory_attention.*`, and `spatial_perceiver.*`;
- `maskmem_tpos_enc`, `no_mem_embed`, `no_mem_pos_enc`, and `no_obj_ptr`;
- `obj_ptr_proj.*`.

It retains A02 TinyViT, prompt encoder, and mask decoder. The topology follows
official EdgeTAM by disabling object-pointer temporal projection, signed
pointer position, and `no_obj_embed_spatial`.

## Temporal compatibility gate

Before full val/test, the driver creates a deterministic 32-video sample from
the official SA-V val list using seed `edgetam-memory-gate-v2`. It evaluates
M0 once on the same list and requires every candidate to satisfy all four:

| Gate | Requirement |
| --- | ---: |
| Absolute mini-val J&F | at least 60.0 |
| J&F relative to M0 | no more than 10.0 points lower |
| mIoU relative to M0 | no more than 0.005 lower |
| AP relative to M0 | no more than 0.005 lower |

A failed candidate stops after mini-val and does not access test. A passing
candidate continues with full SA-V val, checkpoint selection, full SA-V test,
W&B finalization, and the central summary CSV. This is both statistically
cleaner and much cheaper than evaluating a known-broken model on test.

The 32-video gate is only a triage device. Final model selection still uses
full SA-V val J&F. Changes below 0.3 full-val J&F require another seed.

## Recovery results through 2026-07-23

C0 and C1 completed their full training epochs and were evaluated on the same
deterministic 32-video gate. Both passed the image guardrails but failed both
tracking requirements:

| Run | Gate mIoU | Gate AP | Gate J&F | J&F vs M0 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| M0 fixed reference | 0.852800 | 0.756663 | 71.6 | 0.0 | reference |
| C0 coherent + `Lmem` | 0.852409 | 0.755670 | **31.5** | -40.1 | fail |
| C1 partial + `Lmem` | **0.852822** | **0.756625** | 31.1 | -40.5 | fail |

The universal report labels C0 and C1 `final_checkpoint_incomplete` because a
failed gate intentionally prevents creation of `best.pt` and full val/test
artifacts. This is not incomplete training: each run reached 100% of its
planned epoch and retains `last.pt`. Their research status is `gate_failed`.

The coherent stack gains only 0.4 gate J&F over the partial initializer while
remaining 40.1 points below M0. This is too small to support H1 and shows that
initialization provenance and pointer layout are not the dominant remaining
failure. H2 and H3 are also insufficient in their tested form: even an M0
teacher with a pure memory-output objective does not recover temporal
propagation in one full SA-V pass.

The nearly exact M0 image metrics strengthen the temporal diagnosis. The
compressed model still answers a box correctly on the prompted image, but the
memory-conditioned state does not preserve the object across frames. A single
final `F_M` MSE permits internal memory tokens, temporal embeddings, object
pointers, and attention dynamics to remain functionally misaligned.

C3 is therefore blocked as designed. C2 had not started at report time. It
should not be launched as an expected recovery run after this gate result;
retain it only if a deliberately controlled negative joint-training baseline
is worth two additional epochs.

## Optimization and tracking

- Four H100s, batch 1/GPU because the M0 teacher is online.
- Perceiver LR `1e-5 -> 1e-6`; memory attention LR `3e-6 -> 3e-7`; other
  memory parameters `1e-6 -> 1e-7`.
- Five-percent linear warmup, cosine decay, bf16, gradient clipping at 0.1.
- W&B project `edgetam-memory-recovery-v2`; the same run ID is reused on
  resume. TensorBoard writes to the same run directory.
- Log raw/EMA component losses, task/image/memory weights, LR groups,
  outlier batches, gate metrics, full val, and full test.
- Each run retains one physical `last.pt`; `best.pt` and `checkpoint.pt` are
  symlinks. Predictions are deleted after scoring. Expected incremental
  storage is below 10 GB and safely below the 450 GB project limit.

Run root:

`/group-volume/danny-dataset/sam2_distill/runs/edgetam_memory_recovery_v2`

Central summary:

`/group-volume/danny-dataset/sam2_distill/runs/edgetam_memory_recovery_v2/summary.csv`

## Two-node allocation

| Lane | Sequence | Reason |
| --- | --- | --- |
| `recovery1` | C0 -> C3 | staged path; C3 requires C0 gate pass |
| `recovery2` | C1 -> C2 | independent initialization control and joint baseline |

Entry points:

- Single experiment/config audit:
  `scripts/company/49_run_edgetam_memory_ablation.sh describe <variant>`
- Two-node foreground lane:
  `scripts/company/51_run_edgetam_memory_recovery_lane.sh recovery1|recovery2`
- Universal status report:
  `scripts/company/45_report_all_experiments.sh`

## Decision after this suite

1. Stop the staged C3 path because C0 failed its gate.
2. Do not interpret coherent initialization as a recovery: C0 exceeds C1 by
   only 0.4 gate J&F and both remain near 31.
3. Before further optimization, measure first-frame J/F, per-frame decay,
   memory-token norms, object-pointer norms, and attention outputs for M0,
   C0, and C1 on the fixed gate.
4. The next training method must constrain intermediate temporal state or
   behavior, not only final `F_M`: candidates include layer-wise attention
   distillation, memory-token projection/alignment, and teacher-mask
   propagation targets.
5. Only after a compact T4 model reaches at least 60 gate J&F should T8,
   stronger augmentation, or progressive T16 be reconsidered.
