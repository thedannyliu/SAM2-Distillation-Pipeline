# EdgeTAM TinyViT-21M behavior transfer v4

## Question

Can a TinyViT-21M image encoder preserve the released EdgeTAM temporal
behavior when every non-image tensor is transferred coherently, and is
staged image-then-temporal adaptation more stable than joint adaptation
under the available SA-V-only budget? Separately, how much temporal quality
can the same architecture learn when its Perceiver, memory, and object
pointer are randomly initialized?

The implementation is pinned to the
[official EdgeTAM repository](https://github.com/facebookresearch/EdgeTAM)
and follows the image-then-video distillation motivation in the
[EdgeTAM paper](https://arxiv.org/abs/2501.07256).

## Evidence and hypothesis

The local official EdgeTAM fidelity baseline passed two independent
32-video gates at 65.2 and 72.1 J&F and reached 68.0 full-val J&F. The local
evaluator is therefore functional. In contrast, C0/C1 retained the A02
prompt/mask decoder while importing temporal tensors and reached only
31.5/31.1 gate J&F. Their prompted-image metrics remained healthy.

The primary hypothesis is that the previous transfer broke a complete
non-image representation contract. A final memory-conditioned feature MSE
could not repair propagated masks, identity-bearing object pointers, and
the internal temporal trajectory.

## E1 strict zero-training baseline

`E1_a02_official_nonimage` merges exactly:

- every `image_encoder.*` tensor from the completed A02 TinyViT-21M task
  checkpoint;
- every non-`image_encoder.*` tensor from the released `edgetam.pt`.

No A02 prompt encoder, mask decoder, memory tensor, temporal embedding, or
object-pointer tensor is retained. The generated config uses the official
two-block attention layout, 256 global plus 256 2D Perceiver latents, and
official object-pointer flags. Full SA-V val/test evaluation validates a
strict model load before any training result is interpreted.

## Behavior loss

All training candidates retain the official task loss and use an online,
frozen official EdgeTAM teacher. The extended objective is:

`L = Ltask + λimg L(F16) + λmem L(FM) + λlogit LBCE(mask logits) + λptr Lcos(object pointer)`

Mask-logit BCE transfers propagation behavior directly. Object-pointer
cosine loss constrains identity without forcing pointer magnitude. These
targets complement, rather than replace, the final `FM` feature MSE that
failed alone in C0.

## Controlled experiment table

All rows use the same E1 initializer, 50,337 usable SA-V train videos,
exact first-frame boxes, at most two objects, frozen BatchNorm, bf16,
five-percent warmup, and four H100s. Every row runs train, full val, full
test, W&B finalization, and summary CSV recording. `best.pt` is a symlink
to the sole retained physical `last.pt`.

| Run | Trainable modules | T | Epochs | KD `(img, FM, logits, ptr)` | Question |
| --- | --- | ---: | ---: | --- | --- |
| `D1_staged_image_align_1ep` | TinyViT image encoder | 2 | 1 | `(1, 0, 1, 0)` | Can image-interface alignment preserve the official temporal stack? |
| `D2_staged_temporal_2ep` | memory encoder, Perceiver, attention, pointer | 4 | 2 | `(0, .5, 1, .1)` | Does temporal adaptation work after isolating image alignment? |
| `D3_staged_t8_refine_1ep` | same temporal modules | 8 | 1 | `(0, .5, 1, .1)` | Does longer context help after staged T4 training? |
| `J1_joint_behavior_2ep` | TinyViT image encoder plus temporal modules | 4 | 2 | `(1, .5, 1, .1)` | Is joint behavior transfer more data-efficient than staging? |
| `J2_joint_temporal_refine_1ep` | temporal modules | 4 | 1 | `(0, .5, 1, .1)` | Does freezing the adapted image interface stabilize refinement? |
| `J3_joint_t8_refine_1ep` | temporal modules | 8 | 1 | `(0, .5, 1, .1)` | Does joint initialization benefit from longer context? |

The staged and joint lanes each consume four SA-V passes. This controls
total data exposure while testing curriculum. T8 uses the existing
audited `eligible_t8.txt` list rather than silently dropping short videos.

## Temporal-from-scratch control

“From scratch” means random initialization of the complete temporal path:
the 2D Spatial Perceiver, two memory-attention blocks, memory encoder,
object-pointer projections, and temporal/no-memory embeddings. The
TinyViT-21M image encoder and A02 prompt/mask decoder are retained. Training
the entire segmentation model from random initialization is not a useful
SA-V-only control because the available data was selected for video masks,
not for learning a general image segmentation foundation model.

| Run | Initializer and objective | T | Epochs | Question |
| --- | --- | ---: | ---: | --- |
| `S0_scratch_temporal_task_2ep` | random temporal stack, task loss | 4 | 2 | How much temporal function is identifiable from current SA-V supervision alone? |
| `S1_scratch_behavior_2ep` | continue S0 with `(FM=.5, logits=1, ptr=.1)` | 4 | 2 | How much of the gap is teacher behavior rather than architecture? |
| `S2_scratch_t8_refine_1ep` | continue S1 on eligible T8 clips | 8 | 1 | Does longer context finish recovery or expose remaining drift? |

This five-pass curriculum is compared against E1/D and the official E0
baseline. It is not compared to a fully random image model.

## Learning rates

The official Perceiver is already trained, so v4 uses learning rates an
order of magnitude below the failed v1 hybrid:

- image encoder: at most `3e-7`, decayed tenfold;
- Perceiver: `1e-6`, then `5e-7`, each decayed tenfold;
- memory attention: `3e-7`, then `1.5e-7`;
- memory encoder, pointer, and temporal embeddings: `1e-7`, then `5e-8`.

The prompt encoder and mask decoder remain frozen. This prevents a healthy
official decoder from moving while the image/memory interfaces adapt.

## Selection and interpretation

Full SA-V val J&F is the selection metric. Test is recorded but never used
to select a checkpoint.

- below 55 full-val J&F: strict interface or training implementation failure;
- 55–60: temporal conditioning remains weak; do not claim recovery;
- 60–68: functional compressed memory, below the local official baseline;
- 68–71.5: matches/exceeds local official EdgeTAM and approaches M0;
- above 71.5: exceeds the functional four-block M0 baseline.

Because the two 32-video E0 gates differed by 6.9 J&F, gate subsets are
integrity checks only. Fine ranking uses full val.

## Four-node allocation

| Lane | Sequence |
| --- | --- |
| Node 1 `staged` | E1 full val/test -> D1 -> D2 -> D3 |
| Node 2 `scratch` | S0 -> S1 -> S2 |
| Backlog `joint` | E1 checkpoint preparation -> J1 -> J2 -> J3 |
| Node 3 | TinyViT capacity/freeze continuation; see `backbone_task_expansion_v2.md` |
| Node 4 | RepViT task recovery and BN control; see `backbone_task_expansion_v2.md` |

Entry point:

`scripts/company/55_run_edgetam_behavior_lane.sh staged|joint|scratch`

W&B project:

`edgetam-tinyvit21-behavior-v4`

Run root:

`/group-volume/danny-dataset/sam2_distill/runs/edgetam_tinyvit21_behavior_v4`

Every trained row runs W&B training, full SA-V validation, full SA-V test,
and atomic CSV recording. Only `last.pt` is physical; `best.pt` points to
the val-evaluated endpoint. The eval-only E1 transplant is the explicit
pipeline-integrity baseline.
