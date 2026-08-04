# SAM2-TV multiplex v1

## Research question

Can SAM 3.1's object multiplexing data flow be adapted to the TinyViT-21M
SAM2 tracker so that the expensive temporal read and mask decode execute once
per object bucket, without the object-identity collapse observed in the earlier
`shared_kv` experiments?

This experiment is an architectural adaptation, not a claim that SAM2-TV has
become checkpoint-compatible or numerically identical to SAM 3.1.

## Why the previous path failed

The earlier `SharedSlotMemoryAttention` combined independent SAM2 object
memories immediately before K/V projection. It delivered large N=8 speedups,
but the shared representation no longer had a clean way to recover which mask
belonged to which object. The measured outcome was consistent:

| Path | Val J&F | Test J&F | N=8 FPS | Interpretation |
|---|---:|---:|---:|---|
| Quality reference | 72.4 | 74.7 | 22.07 | Persistent runtime bucket |
| MX5 decoder slots | 72.3 | 74.6 | 24.40 | Slot decoder is quality-safe; memory remains per object |
| MX4/MX6-style shared K/V | about 63.6 | about 60.3 | about 49 | Fast, but object memory was mixed too early |

The new invariant is therefore: **reduce the bucket batch dimension only after
object identity has been encoded as a slot dimension**.

## Architecture

For a bucket capacity `S=8`, the propagation path is:

```text
SAM2 per-object masks [N,1,H,W]
          |
          | stable contiguous slot assignment, zero-pad final bucket
          v
bucket masks [B,S,H,W]
          |
          | one multi-channel mask-memory encoder call
          v
bucket dense memory [B,C,h,w]
          |
          +-- spatial memory: one K/V map per bucket
          +-- object pointers: S private tokens with learned slot positions
          v
one temporal-memory attention call per bucket
          |
          | S learned mask/IoU/object-score tokens
          v
one mask-decoder transformer call per bucket
          |
          v
demultiplexed masks [N,1,H,W]
```

### 1. Slot-preserving mask memory

`SlotPreservingMemoryEncoder` clones SAM2's mask downsampler and changes only
its first convolution from one mask channel to eight. Its inputs are the eight
object masks in a bucket, rather than a sum or average of their already encoded
memories. The remainder of the SAM2 memory encoder—pixel projection, ConvNeXt
fuser, output projection, and positional encoding—is retained.

The new first-layer channel weights start from the selected SAM2 checkpoint's
single-channel weights. They are independent parameters after initialization,
so training can learn a different interpretation for every slot while keeping
a stable starting scale for a single active channel.

Code: `sam2_distill/models/sam2_object_slots.py`, class
`SlotPreservingMemoryEncoder`.

### 2. One temporal read with private pointer tokens

The dense memory is read once per bucket. Object pointers are not averaged:
each previous pointer remains a private memory token and receives a learned
slot positional embedding. Consequently the temporal reader sees one shared
spatial map plus the ordered pointer sequence for all eight slots.

This is the main correction to the failed `shared_kv` design. The only spatial
averaging left in the compatibility layer combines copies of an already-joint
bucket memory (and SAM2's optional no-object offset); it does not average the
eight input masks or eight independent memory encodings.

Code: `sam2_distill/models/sam2_object_slots.py`, class
`SlotPreservingMemoryAttention`.

### 3. Multiplex decoder tokens

The existing learned slot decoder is retained because MX5 showed that it can
preserve full quality. One transformer call receives one mask token, one IoU
token, and one object-score token per slot. Each mask token produces its own
hypernetwork mask head.

Code: `sam2_distill/models/sam2_object_slots.py`, class
`LearnedObjectSlotDecoder`.

### 4. Routing and compatibility

During training, all object counts use the multiplex path, including partial
buckets, so the model learns padding and low-occupancy cases. During inference,
N=1 and N=2 use the original per-object SAM2 path; N>=4 uses multiplexing.
The selected threshold is recorded in the resolved config as
`object_slot_min_objects: 4`.

The first prompted frame still uses the original SAM2 prompt encoder and mask
decoder. Multiplexing begins when masks are written to temporal memory and on
subsequent propagation frames.

## Training recipe

The run uses the full prepared SA-V train manifest. It starts from MX5 because
that checkpoint already established a quality-preserving slot decoder, while
the new memory encoder and private-pointer layout are initialized from its
SAM2-compatible weights.

| Stage | Frames | Epochs | Trainable path | Purpose |
|---|---:|---:|---|---|
| SMX1 | 4 | 2 | memory encoder, memory attention, SAM decoder, slot decoder, pointer projections | Stabilize the new memory representation |
| SMX2 | 8 | 8 | same | Main full-SA-V temporal training |
| SMX3 | 16 | 2 | same, lower LR | Long-context refinement |

All stages use four GPUs, batch size one per GPU, at most eight objects, mixed
point/box prompting, random correction frames, and seven correction points.
The loss is task loss plus `2x` per-object mask-logit distillation from the best
TV21 checkpoint. Dense-memory distillation is intentionally disabled: the
teacher has eight independent memory tensors while the student has one joint
slot-preserving tensor, so an elementwise memory loss would optimize an
incompatible representation and recreate the old averaging pressure.

Checkpoints, W&B state, resolved configs, and evaluation outputs are written to
`$SAM2D_ROOT/runs/sam2_tv_multiplex_v1`. Each stage resumes from its existing
`last.pt` and W&B run ID.

## Evaluation and decision metrics

The `all` action runs, in order:

1. input audit;
2. SMX1 training;
3. SMX2 training;
4. SMX3 training;
5. full SA-V validation image and VOS evaluation;
6. full SA-V test image and VOS evaluation;
7. isolated N=1,2,4,8 propagation latency.

The initial promotion criteria are:

| Metric | Gate |
|---|---:|
| Val J&F retention vs 72.4 | at least 95% |
| Test J&F retention vs 74.7 | at least 95% |
| N=1 FPS retention | at least 95% |
| N=8 FPS | greater than 22.07 FPS |
| Primary selection | Pareto trade-off of J&F and N=8 latency |

No result is recorded until full val/test completes. Training loss or the
32-video screen alone is not sufficient evidence for promotion.

## Company command

Run in the foreground on one 4xH100 node:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only origin main
mkdir -p /user-volume/sam2_tv_multiplex_v1_logs

GPUS=0,1,2,3 \
WANDB_MODE=online \
SKIP_DONE=1 \
scripts/company/70_run_sam2_tv_multiplex_v1.sh all 2>&1 | \
tee "/user-volume/sam2_tv_multiplex_v1_logs/all_$(date +%Y%m%d_%H%M%S).log"
echo "SAM2-TV multiplex status: ${PIPESTATUS[0]}"
```

The command auto-detects the prepared full-SA-V manifest, SA-V val/test roots,
MX5 checkpoint, and the current company data root. Override `SAM2D_ROOT`,
`SAV_ROOT`, `MANIFEST`, or `MX5_SLOT8_CHECKPOINT` only when the automatic path
selection is wrong.

This is a long curriculum rather than a 20-hour screen. Use the first completed
SMX1 epoch to estimate remaining wall time on the actual container and storage
mount; T8 and T16 stages are substantially more expensive than T4.

## What matches SAM 3.1

The following mechanisms follow the released SAM 3.1 implementation:

- fixed-capacity object slots grouped into buckets;
- masks multiplexed as channels before the mask-memory encoder;
- one current visual feature and one dense memory stream per bucket;
- one ordered object-pointer entry per slot;
- per-slot mask, IoU, and object-score decoder tokens;
- one temporal read and one decoder transformer call per bucket.

Primary references:

- [SAM 3.1 multiplex release](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
- [SAM 3.1 multiplex model builder](https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py)
- [Multiplex state and controller](https://github.com/facebookresearch/sam3/blob/main/sam3/model/multiplex_utils.py)
- [Multiplex video tracker](https://github.com/facebookresearch/sam3/blob/main/sam3/model/video_tracking_multiplex.py)
- [Multiplex mask decoder](https://github.com/facebookresearch/sam3/blob/main/sam3/model/multiplex_mask_decoder.py)
- [Multi-channel mask memory](https://github.com/facebookresearch/sam3/blob/main/sam3/model/memory.py)

## What still differs from SAM 3.1

| Area | This SAM2-TV experiment | SAM 3.1 |
|---|---|---|
| Backbone/model | TinyViT-21M inside SAM2.1 tracking graph | SAM 3.1 native backbone and tracker |
| Capacity | 8 slots, selected for the measured SAM2 workload | 16 slots by default |
| State ownership | SAM2 keeps per-object dictionaries; bucket memory is repeated for API compatibility and collapsed before attention | Native bucket-first `MultiplexState` stores bucket tensors directly |
| Object lifecycle | Stable contiguous object order; objects should be added before propagation | Explicit controller supports persistent slot allocation and richer add/remove flows |
| Mask-memory input | 8 object-mask channels; first conv begins with 1 output-channel unit | 16 mask channels plus 16 conditioning channels (`input_channel_multiplier=2`), first width 4 |
| Memory width | SAM2's 64-channel memory | 256-channel multiplex memory |
| Temporal reader | SAM2 four-layer, one-head memory attention with packed private pointers | Four-layer decoupled 8-head transformer, saved image features, and separate image/memory handling |
| Decoder | Existing SAM2 decoder transformer with learned per-slot tokens | Native `MultiplexMaskDecoder` and its exact token/head options |
| Temporal/no-object features | SAM2 temporal positions and fixed no-object pointer inherited from the selected checkpoint | v2 temporal memory positions, linear no-object pointer, output-suppression embeddings, and conditioning-mask inputs |
| Resolution/system work | 1024 input, stride 16, ordinary PyTorch path | 1008 input, stride 14, FlashAttention 3 options, `torch.compile`, operation fusion, batched postprocessing, and reduced CPU/GPU synchronization |
| Training recipe | Full SA-V, global batch 4, three-stage T4/T8/T16 curriculum | Meta's private training mixture, batch scale, schedule, and weights are not reproduced |
| Dynamic padding mask | Zero-padded final bucket; invalid outputs are removed after decoding | Native multiplex state/controller carries richer validity metadata throughout |

Therefore this experiment tests the architectural hypothesis fairly inside
SAM2, but it cannot isolate every source of SAM 3.1's published end-to-end
speed. If quality succeeds and storage/runtime overhead remains material, the
next engineering step is to replace SAM2's per-object output dictionaries with
native bucket-first state rather than changing the learned representation
again.
