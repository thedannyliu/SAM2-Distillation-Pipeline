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
SAM2 per-object masks [N,1,H,W] + condition flags [N,1,H,W]
          |
          | persistent object-id assignment; padding and tombstones differ
          v
bucket inputs [B,2S,H,W]
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

The same `PersistentMultiplexLayout` instance controls all three learned
modules. Training creates a fresh random assignment per tracking step; video
inference keeps the assignment by client object ID for the lifetime of the
session.

### 1. Slot-preserving mask memory

`SlotPreservingMemoryEncoder` clones SAM2's mask downsampler and changes its
first convolution from one channel to 16: eight mask channels followed by
eight binary conditioning channels. This matches the released SAM 3.1
`input_channel_multiplier=2` layout. Its inputs are the eight object masks in
a bucket, rather than a sum or average of their already encoded memories. The
remainder of the SAM2 memory encoder—pixel projection, ConvNeXt fuser, output
projection, and positional encoding—is retained.

The new first-layer channel weights start from the selected SAM2 checkpoint's
single-channel weights. Mask-channel weights repeat that initializer;
conditioning-channel weights start at zero, so enabling the second half does
not perturb the selected checkpoint before training. They become independent
parameters during multiplex training.

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
decoder. Its output masks are then re-encoded jointly into multiplex memory
before propagation, so the first temporal read already consumes slot-preserving
bucket memory.

### 5. Runtime state and dynamic objects

The initial implementation incorrectly left the official SAM2
`propagate_in_video` loop in control. That loop calls
`_run_single_frame_inference(..., batch_size=1)` once per object, so using
`execution-mode=legacy` prevented an N>=4 model from entering its multiplex
path. The current predictor fixes this at the model boundary:

- `SAM2ObjectSlotVideoPredictor` owns a persistent layout in each inference
  state;
- additions fill true padding slots; removals become tombstones and are not
  reused, preventing stale temporal memory from changing identity;
- all modules receive the same selected layout before each bucket call;
- mux/demux uses cached device index tensors rather than Python per-slot tensor
  writes; this is the sparse gather/scatter equivalent of SAM 3.1's cached
  transition matrices;
- asynchronous prompt histories retain the frame's shared bucket memory while
  filling absent per-object masks, scores, and pointers with no-object values,
  instead of forcing the whole session back to per-object inference;
- after SAM2's standard prompt consolidation, every affected conditioning
  frame is re-encoded jointly by bucket with a per-object conditioning vector;
  this prevents the first temporal read from averaging separately encoded
  prompt memories;
- on a mixed conditioning frame, prompted or already-computed objects keep
  their fixed outputs while the remaining objects run as one bucketed batch;
- outputs are split back into SAM2's per-object dictionaries, preserving the
  public predictor API.

Code: `sam2_distill/models/sam2_object_slot_predictor.py`, class
`SAM2ObjectSlotVideoPredictor`; `sam2_distill/models/sam2_object_slots.py`,
class `PersistentMultiplexLayout`; and
`sam2_distill/models/sam2_object_buckets.py`, class
`SAM2ObjectBucketAdapter`.

SAM2 still stores compact outputs per object, whereas SAM 3.1 stores native
bucket tensors. The compatibility-history merge is therefore an extra cost:
missing object histories reuse the bucket's shared memory, while only their
per-object masks, scores, and pointers receive no-object values. This makes the
learned path real for full VOS evaluation rather than only for the synchronized
latency cohort.

### 6. Kernel compilation

`SAM2_TV_COMPILE=1` compiles the image encoder, multiplex mask downsampler,
memory projection/fuser, individual temporal-attention layers, and mask-decoder
transformer after strict checkpoint loading. The Python session controller and
dynamic layout remain eager. This boundary is intentional: object
additions/removals change Python state, while the selected tensor kernels have
stable H100 shapes and benefit from fusion.

This is not FlashAttention 3. The company image is pinned to Python 3.10,
PyTorch 2.4, and CUDA 12.5; the released SAM 3.1 repository specifies Python
3.12+, PyTorch 2.7+, CUDA 12.6+, and lists FA3 as an optional installation.
Exact FA3 comparison therefore needs a separate company image and cannot be
safely installed by mutating this experiment's PyTorch runtime.

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
3. full SA-V validation image and VOS evaluation for SMX1;
4. SMX2 training;
5. full SA-V validation image and VOS evaluation for SMX2;
6. SMX3 training;
7. full SA-V validation image and VOS evaluation for SMX3;
8. full SA-V test image and VOS evaluation for SMX3 only;
9. isolated N=1,2,4,8 propagation latency.

Validation is stage-level rather than epoch-level: it runs once after each of
the three training stages. All three validation passes use every video in
`sav_val.txt` (`MAX_VIDEOS=0`) and four evaluation shards. Test remains held
out until the final SMX3 stage has trained and completed validation.

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

After eager latency completes, compare compiled kernels using the same final
checkpoint and cohort:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
SAM2_TV_COMPILE=1 \
WANDB_MODE=online \
scripts/company/70_run_sam2_tv_multiplex_v1.sh latency-compiled 2>&1 | \
tee "/user-volume/sam2_tv_multiplex_v1_logs/latency_compiled_$(date +%Y%m%d_%H%M%S).log"
echo "Compiled latency status: ${PIPESTATUS[0]}"
```

The output directory is `point_n1-2-4-8_compiled`, so it does not overwrite
the eager measurement.

## What matches SAM 3.1

The following mechanisms follow the released SAM 3.1 implementation:

- fixed-capacity object slots grouped into buckets;
- persistent slot allocation, distinct padding/removal state, and dynamic
  object add/remove;
- masks multiplexed as channels before the mask-memory encoder;
- one conditioning channel per mask slot;
- one current visual feature and one dense memory stream per bucket;
- one ordered object-pointer entry per slot;
- per-slot mask, IoU, and object-score decoder tokens;
- one temporal read and one decoder transformer call per bucket;
- random slot assignments during training and deterministic assignments at
  inference;
- native batched propagation for asynchronous object histories.

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
| State ownership | Persistent layout plus SAM2 per-object dictionaries; asynchronous holes reuse shared bucket memory while absent per-object outputs are neutral | Native bucket-first `MultiplexState` stores bucket tensors directly |
| Object lifecycle | Persistent IDs, padding/tombstones, add/remove supported through the SAM2 predictor API | Native controller provides the same semantics without the compatibility merge |
| Mask-memory input | 8 mask plus 8 conditioning channels; cloned SAM2 downsampler starts at width inherited from SAM2 | 16 mask plus 16 conditioning channels; first width 4 |
| Memory width | SAM2's 64-channel memory | 256-channel multiplex memory |
| Temporal reader | SAM2 four-layer, one-head memory attention over joint dense memory and private pointers | Four-layer decoupled 8-head transformer jointly receives current image, saved historical image features, multiplex memory, and pointers |
| Decoder | Existing SAM2 two-way transformer, but separate mask/IoU/object-score token per slot and a shared hypernetwork family | Native `MultiplexMaskDecoder` has independent slot embeddings and additional multimask/token-sharing modes |
| Temporal/no-object features | SAM2 temporal positions and fixed no-object pointer inherited from the selected checkpoint | v2 temporal memory positions, linear no-object pointer, output-suppression embeddings, and conditioning-mask inputs |
| Resolution/system work | 1024 input, stride 16; optional compile of stable tensor kernels | 1008 input, stride 14, FA3 options, broader compile/fusion, batched postprocessing, and reduced CPU/GPU synchronization |
| Training recipe | Full SA-V, global batch 4, three-stage T4/T8/T16 curriculum | Meta's private training mixture, batch scale, schedule, and weights are not reproduced |
| Conditioning identity | A second channel per slot is trained; runtime preflight reconstructs a per-object conditioning vector before every affected bucket is jointly re-encoded, while the SAM2 training loader still exposes only a frame-level `is_mask_from_pts` flag | A native per-object `conditioning_objects` set is available directly in both training and inference |
| Mux/demux dispatch | Cached device index tensors and `index_select`; no per-slot CPU/GPU writes | Cached partial-permutation transition matrices and matrix multiplication |
| Dynamic padding mask | Shared persistent validity layout across encoder, reader, and decoder | Native multiplex state additionally keeps bucket tensors as the primary state representation |

These remaining differences are not import-path problems:

1. **Temporal reader.** The SAM 3.1 reader signature takes current image
   features, saved historical image features, multiplex memory, and pointer
   tokens separately. SAM2 checkpoints contain a 64-d cross-attention K/V
   projection and do not store historical image features in compact outputs.
   Swapping the class would leave incompatible tensor shapes and no
   `memory_image` input. A fair port is a new model stage with 256-d memory,
   modified compact state, and a newly trained reader—not a strict-load change
   to the current checkpoint.
2. **Exact decoder.** Token topology is now aligned, but Meta's decoder owns
   different embedding tables and multimask policies. Replacing the SAM2
   decoder invalidates the high-resolution upscaler, IoU head, object-score
   head, and transformer checkpoint. It should be a controlled decoder
   reinitialization experiment only after the runtime/reader result is known.
3. **FA3 and full compile.** The released dependencies exceed the pinned
   company container. The safe integration here is PyTorch 2.4 compile on
   stable subgraphs; exact FA3 requires a separately versioned image and a
   correctness/latency comparison.
4. **Native bucket state.** The persistent controller is aligned, but SAM2's
   public API and checkpoints still assume per-object compact histories. The
   compatibility merge is now correct for dynamic objects, yet removing it
   requires a new state schema and migration logic for interactive sessions.

Consequently the current experiment isolates the most important transferable
hypothesis—slot-preserving shared memory with a real multiplex runtime—while
keeping the TV21/SAM2 checkpoint usable. The next architecture experiment, if
this version clears the 95% J&F gate, is a separately named 256-d decoupled
reader stage rather than silently changing this experiment mid-run.
