# SAM2 efficient memory and multiplex progress

## Scope

This document consolidates the efficient-memory and multi-object multiplex
work through the company-side snapshot generated on 2026-08-04. It explains
what each experiment changed, how it was trained and evaluated, what the
results establish, and what remains unresolved.

The normalized source of record is
[`all_experiment_results_2026-08-04.md`](all_experiment_results_2026-08-04.md).
Statuses in this document describe that snapshot, not necessarily the current
state of a still-running company process.

The selected TinyViT-21M reference is:

| Item | Value |
|---|---:|
| Full SA-V val J&F | 72.4 |
| Full SA-V test J&F | 74.7 |
| Legacy N=1 propagation | 72.47 FPS / 13.80 ms |
| Legacy N=8 propagation | 14.96 FPS / 66.83 ms |

The research objective is not merely higher aggregate mask throughput. A
candidate must preserve object identity and temporal tracking while reducing
the object-axis cost. The working quality threshold is at least 95% val and
test J&F retention; the runtime objective is lower N=4/N=8 propagation
latency without sacrificing the N=1 path.

## 1. Where the multi-object cost comes from

TinyViT reduces the image-encoder cost, but SAM2 still maintains
object-conditioned temporal state. Memory attention, object pointers, mask
decoding, state updates, and output tensors continue to scale with the object
batch. The selected model's measured scaling was:

| Objects | Propagation FPS | Median ms/frame | Relative latency vs N=1 |
|---:|---:|---:|---:|
| 1 | 72.57 | 13.78 | 1.00× |
| 2 | 48.17 | 20.76 | 1.51× |
| 4 | 27.64 | 36.19 | 2.63× |
| 8 | 15.02 | 66.68 | 4.84× |

This motivated two complementary directions:

1. **Runtime multiplexing:** keep the model fixed but organize object state
   into persistent bounded buckets.
2. **Learned multiplexing and efficient memory:** share decoder and memory
   computation across fixed object slots, then train the new representation.

The original scaling design is recorded in
[`sam2_multiobject_scaling_v1.md`](sam2_multiobject_scaling_v1.md).

## 2. Runtime-only object buckets

### 2.1 Design

The runtime adapter is implemented in
[`sam2_object_buckets.py`](../../sam2_distill/models/sam2_object_buckets.py).
It wraps the existing predictor and does not modify model weights or the
public prompt/propagation API.

For a capacity `B`, stable object IDs are partitioned into bounded groups:

```text
ordered object IDs
  -> [0 ... B-1], [B ... 2B-1], ...
  -> one persistent history for each bucket
  -> bounded SAM2 inference calls
  -> split outputs back to the original object order
```

This is not yet a learned SAM3.1-style multiplexer. The total object batch
across all bucket calls is still `N`; the optimization comes from bounded
shapes, persistent packed histories, reduced allocator traffic, and improved
kernel behavior.

The adapter preserves tracker semantics through explicit fallbacks:

- sessions below `min_bucket_objects` use the original predictor;
- objects with unsynchronized histories use the original predictor;
- object order and per-object output dictionaries remain unchanged;
- unsupported state options are rejected instead of silently changing output.

### 2.2 Non-persistent prototype

The first capacity-four prototype repacked and split every object's complete
history on every frame. It showed a small high-object benefit but introduced a
large fixed overhead and allocation peak:

| Objects | Legacy FPS | Prototype FPS | FPS change | Extra peak memory |
|---:|---:|---:|---:|---:|
| 1 | 72.86 | 35.82 | -50.8% | +1,269 MB |
| 2 | 45.89 | 35.49 | -22.7% | +2,715 MB |
| 4 | 27.24 | 28.54 | +4.8% | +5,607 MB |
| 8 | 14.15 | 15.71 | +11.0% | +5,655 MB |

The result rejected the implementation, not the bucket hypothesis. It
identified repeated packing and transient storage as the dominant regression.

### 2.3 Persistent bucket result

The second implementation packs synchronized histories once, appends frame
outputs in place, and retains bucket membership throughout propagation. It
also keeps N<4 on the exact legacy path.

| Objects | Legacy FPS | Persistent FPS | FPS gain | Latency reduction |
|---:|---:|---:|---:|---:|
| 1 | 72.47 | 71.36 | -1.5% | -1.6% |
| 2 | 47.68 | 47.15 | -1.1% | -1.1% |
| 4 | 26.83 | 38.09 | +42.0% | +29.6% |
| 8 | 14.96 | 22.07 | +47.5% | +32.2% |

At N=8, extra peak memory fell to 584 MB. Binary-mask agreement was
0.9999849 and minimum per-mask IoU was 0.9690. The automatic decision was
`REJECT` because the run was not bit-exact and did not pass the original
strict promotion-count gate. Under the project's later 95%-quality tolerance,
this remains the strongest near-term deployment candidate.

The deployment contract and integration details are in
[`sam2_tinyvit_multiobject_thor.md`](../deployment/sam2_tinyvit_multiobject_thor.md).

## 3. Reducing memory-attention depth: MO0--MO3

### 3.1 Controlled protocol

These experiments asked whether the four-layer SAM2 memory stack was
redundant. All four lanes used:

- the selected TinyViT-21M checkpoint;
- a deterministic SA-V cohort with at least eight visible objects over at
  least four annotated frames;
- T4 clips and up to eight sampled objects;
- 4×H100 DDP, per-GPU batch one, global batch four;
- five epochs and a matched 50,337-sample update budget;
- frozen image encoder and BatchNorm;
- a 50/50 point/box prompt mixture without iterative correction clicks;
- full SA-V val, full SA-V test, then isolated N=1/2/4/8 latency.

Only memory depth and distillation terms changed:

| Run | Memory layers | Objective | Hypothesis |
|---|---:|---|---|
| `MO0_mem4_task_dense8_5ep` | 4 | task | Dense multi-object continuation control |
| `MO1_mem2_task_dense8_5ep` | 2 | task | Measure the pure depth speed/quality tradeoff |
| `MO2_mem2_logits_dense8_5ep` | 2 | task + propagated-logit KD | Test whether output behavior restores quality |
| `MO3_mem2_memlogits_dense8_5ep` | 2 | task + logit KD + memory KD 0.5 | Test whether temporal-state matching helps |

### 3.2 Results and interpretation

| Run | Val J&F | Test J&F | N=8 FPS | N=8 ms |
|---|---:|---:|---:|---:|
| MO0 | 69.4 | 72.4 | 24.16 | 41.39 |
| MO1 | 56.9 | 58.5 | 34.32 | 29.14 |
| MO2 | 57.1 | 58.5 | 34.98 | 28.59 |
| MO3 | 56.9 | 58.4 | 35.19 | 28.42 |

Two layers provide a real speed gain but lose roughly 14 test J&F points.
Neither propagated-logit nor memory-feature KD repairs the loss. Four-layer
memory attention therefore carries important temporal association, occlusion,
and identity refinement rather than only redundant compute.

## 4. Learned object-slot decoder

### 4.1 Architecture

The learned implementation is in
[`sam2_object_slots.py`](../../sam2_distill/models/sam2_object_slots.py), with
the inference wrapper in
[`sam2_object_slot_predictor.py`](../../sam2_distill/models/sam2_object_slot_predictor.py).

`LearnedObjectSlotDecoder` changes the expensive decoder batch from objects
to buckets. For capacity `B`:

```text
N object-conditioned features
  -> fuse each group of B into one bucket feature
  -> one existing SAM2 two-way transformer call per bucket
  -> B learned mask/IoU/object-score token groups
  -> one mask per valid slot
```

Each slot receives a fixed cosine channel code. For a bucket, object features
are decomposed into a mean and object-specific centered deviations:

```text
mean = average(valid object features)
fused = mean + sum(slot_code[i] * (feature[i] - mean)) / sqrt(count)
```

Learned slot-token embeddings are added to the existing SAM2 mask, IoU, and
object-score tokens. The pretrained two-way transformer and output
hypernetwork are reused. Invalid padded slots are masked before outputs are
returned in the original object order.

The original image encoder, prompt encoder, mask-decoder weights, memory
encoder, memory attention, and pointer head remain frozen in the v1 slot
study. Only the new slot tensors are trained. This isolates the causal effect
of multiplexing from broad model fine-tuning.

At inference, learned buckets are used only on synchronized, unprompted
propagation frames. Point/mask-input frames, small sessions, or asynchronous
object histories fall back explicitly to the selected legacy tracker.

### 4.2 V1 training protocol

All lanes used the same dense-eight cohort, seed, global batch four, T4 clips,
three epochs, BF16, task loss, and propagated-mask logit KD. Shared-K/V lanes
also used memory-feature KD. Object-pointer KD was disabled because the
teacher contract did not expose `teacher_obj_ptr` on every path.

| Run | Capacity | Decoder shared | Memory K/V shared | Trainable additions |
|---|---:|---:|---:|---|
| MX1 | 4 | yes | no | decoder slot tensors |
| MX2 | 8 | yes | no | decoder slot tensors |
| MX3 | 4 | yes | yes | decoder slots + memory slot codes |
| MX4 | 8 | yes | yes | decoder slots + memory slot codes |

### 4.3 Decoder-only results

| Run | Val/Test J&F | Learned-path mask IoU | N=1 FPS | N=8 FPS | N=8 gain vs persistent |
|---|---|---:|---:|---:|---:|
| MX1 slot4 | 72.3/74.6 | 0.000 | 59.68 | 21.68 | -1.8% |
| MX2 slot8 | 72.3/74.6 | 1.000 | 61.65 | 24.46 | +10.8% |

MX1's full-set quality is protected in part by fallback behavior; its zero
synchronized-path IoU means it is not a valid learned-path result. MX2 is the
important positive result: one slot-eight decoder call preserves 99.9% of
full tracking quality and has perfect measured synchronized-path agreement.

The remaining speedup is modest because four-layer memory attention is still
object-specific. MX2 also retains only 86.4% of reference N=1 FPS. A
deployment engine can potentially remove this overhead by routing N<4 around
the learned wrapper externally, but this has not yet been established by a
matched engine benchmark.

The complete v1 protocol and result interpretation are in
[`sam2_learned_object_slots_v1.md`](sam2_learned_object_slots_v1.md).

## 5. Fully shared memory K/V

### 5.1 Architecture

`SharedSlotMemoryAttention` compresses the object axis before the existing
memory-attention K/V projections. Per-object memory is weighted by learned
slot codes and superposed into one memory stream per bucket:

```text
bucket_memory =
  sum(object_memory[i] * memory_slot_code[i]) / sqrt(valid_count)
```

Memory positional encodings are averaged. One representative current feature
per bucket attends to this shared memory through the original four memory
layers. The bucket result is then repeated back to its objects before the
slot decoder separates masks.

For N=8 and capacity eight, this changes eight object-conditioned memory
streams into one bucket memory-attention stream. It is the source of the
large speedup and also the main information bottleneck.

### 5.2 V1 shared-K/V results

| Run | Val/Test J&F | Learned-path mask IoU | N=8 FPS | N=8 ms |
|---|---|---:|---:|---:|
| MX3 slot4 shared K/V | 63.8/60.9 | 0.413 | 34.52 | 28.97 |
| MX4 slot8 shared K/V | 63.6/60.2 | 1.000 | 49.03 | 20.39 |

MX4 establishes the speed upper bound: 2.22× the 22.07-FPS persistent-bucket
reference. However, test J&F falls by 14.5 points. Image-only metrics stay
unchanged at approximately 0.84 mIoU and 0.72 AP, localizing the failure to
temporal tracking rather than image segmentation.

A learned-path mask IoU of 1.0 does not mean MX4 retains teacher quality. It
only means bucket and legacy execution agree for the same already-degraded
trained checkpoint. Full val/test J&F remains the authoritative quality
measure.

## 6. Longer rollout and stronger KD: MX5--MX8

V2 tested whether the shared-K/V failure was merely insufficient training or
distillation. All runs initialized from MX2, used capacity eight, the same
dense cohort and seed, four frozen standard memory layers, T8 clips, five
epochs, BF16, and the same teacher.

| Run | Shared K/V | Memory KD | Logit KD | Val/Test J&F | N=8 FPS |
|---|---:|---:|---:|---|---:|
| MX5 | no | 0 | 2 | 72.3/74.6 | 24.40 |
| MX6 | yes | 1 | 2 | 63.6/60.3 | 49.06 |
| MX7 | yes | 4 | 2 | 63.7/60.5 | 48.72 |
| MX8 | yes | 1 | 4 | 63.6/60.3 | 48.79 |

MX5 confirms that longer T8 decoder training preserves quality but does not
remove the memory bottleneck. MX6--MX8 reproduce the approximately 49-FPS
shared-K/V endpoint and the approximately 60-J&F test endpoint. Increasing
memory KD from 1 to 4 adds only 0.1/0.2 val/test J&F; doubling logit KD adds
nothing. The failure is therefore representational rather than a simple
optimization-duration or KD-weight problem.

The full v2 record is in
[`sam2_learned_object_slots_v2.md`](sam2_learned_object_slots_v2.md).

## 7. Shared K/V plus low-rank per-object residual

### 7.1 Architecture

`LowRankObjectMemoryResidual` attempts to restore cheap private state after
bucket-shared attention. Each enabled spatial or pointer path applies:

```text
LayerNorm
  -> Linear(memory_dim, rank)
  -> GELU
  -> Linear(rank, hidden_dim)
  -> add to shared-attention output
```

The final projection is zero-initialized, so every candidate begins exactly
at its shared-K/V parent. Spatial memory can be pooled across time with:

- `mean`: equal contribution from all memory frames;
- `latest`: only the newest aligned frame;
- `recency`: exponentially weighted frames with a configured decay.

Object-pointer tokens are pooled separately and can use an independent rank.
This avoids another full per-object attention stack, but it adds private state
only after shared attention has already routed information.

### 7.2 MX13--MX28 screen

The screen used eight independent 4×H100 nodes, two sequential experiments
per node, the completed MX5 initializer, T8 dense-object clips, three epochs,
and seed `250107256`. Quality used the same deterministic 32-video SA-V val
cohort for every candidate; latency used one dense-eight video and one
repetition on an isolated H100. Full five-epoch val/test was reserved for
screen-passing Pareto candidates.

The matrix varied one factor at a time:

| Variants | Axis |
|---|---|
| MX13--MX16 | spatial residual rank 2/4/8/16 |
| MX17--MX18 | object-pointer rank 4/8 |
| MX19--MX22 | latest or recency pooling; decay 0.25/0.50/0.75 |
| MX23--MX24 | slot capacity 4/6 instead of 8 |
| MX25--MX26 | object-pointer KD 0.25/1.0 |
| MX27--MX28 | shared-path threshold N=2/N=3 instead of N=4 |

Unless changed by the row, candidates used capacity eight, spatial rank
eight, pointer rank eight, temporal mean pooling, minimum shared object count
four, memory KD one, and mask-logit KD two.

All 16 variants produced latency aggregates. Most capacity-eight runs reached
approximately 46--48 FPS at N=8 and 50--54 FPS at N=4. Examples:

| Run | Main change | N=4 FPS | N=8 FPS | N=8 ms |
|---|---|---:|---:|---:|
| MX15 | spatial rank 8 | 51.03 | 46.76 | 21.39 |
| MX19 | latest pooling | 54.17 | 48.43 | 20.65 |
| MX21 | recency decay 0.25 | 53.56 | 47.25 | 21.16 |
| MX26 | pointer KD 1.0 | 53.21 | 47.40 | 21.10 |
| MX27 | share from N=2 | 53.64 | 46.81 | 21.36 |

Only six quality screens were complete in the snapshot, and all failed:

| Run | Mini J&F | Retention vs MX5 | Mask IoU | Promote |
|---|---:|---:|---:|---:|
| MX15 | 53.60 | 0.757 | 1.000 | no |
| MX17 | 53.70 | 0.758 | 1.000 | no |
| MX19 | 53.00 | 0.749 | 1.000 | no |
| MX21 | 52.90 | 0.747 | 1.000 | no |
| MX23 | 53.70 | 0.758 | 0.000 | no |
| MX27 | 29.50 | 0.417 | 1.000 | no |

These results do not support rank, temporal pooling, threshold, or KD weight
as the missing solution. The most plausible interpretation is that adding a
pooled residual after attention is too late: object identity has already been
lost during shared attention routing.

The matrix and screen policy are documented in
[`sam2_multiplex_overnight_v4.md`](sam2_multiplex_overnight_v4.md).

## 8. EdgeTAM-style efficient temporal memory

The shared-K/V work is an aggressive project-specific compression. EdgeTAM
provides a separate, known efficient temporal design, so the EdgeTAM line asks
whether a coherent official temporal stack can be adapted to TinyViT-21M.

### 8.1 Early ablations

| Run | Temporal configuration | Val/Test J&F |
|---|---|---|
| M0 | standard SAM2, four memory layers | 71.5/74.3 |
| M1 | standard SAM2, two memory layers | 53.3/56.1 |
| M2a | EdgeTAM hybrid2, official initializer | 15.6/12.8 |
| M2b | EdgeTAM hybrid2, current initializer | 13.2/10.6 |
| R0--R3 | TinyViT + EdgeTAM end-to-end variants | about 19--25/about 19--23 |

The early results alone could not distinguish a broken trainer/interface from
an inadequate recipe or representation mismatch.

### 8.2 Q0--Q2 diagnostic experiments

The diagnostics in
[`edgetam_recipe_diagnostics_v5.md`](edgetam_recipe_diagnostics_v5.md)
separated those hypotheses.

**Q0: official identity control**

- exact released RepViT-M1 EdgeTAM graph and `edgetam.pt`;
- T8, one SA-V epoch;
- only memory attention, memory encoder, 2D Perceiver, object-pointer, and
  temporal paths trainable;
- temporal learning rates between `3e-8` and `3e-7`.

Q0 reached 68.3 val and 69.5 test J&F. The local trainer, checkpoint loader,
and evaluator can therefore preserve a functioning official EdgeTAM model.

**Q1: TinyViT 16-video overfit diagnostic**

- TinyViT-21M A02 image/mask path;
- coherent official temporal initialization;
- deterministic 16-video subset, T8, 500 epochs, approximately 2,000 updates;
- task, memory, and propagated-logit losses.

Full val/test J&F remained 15.5. This shows that repeated small-subset updates
do not produce generalizable temporal behavior. Because the formal evaluator
did not score that train subset, it does not by itself prove whether those 16
videos were memorized.

**Q2: paper-scaled available-data recipe**

- TinyViT-21M A02 plus official temporal initialization;
- SAM2.1 Hiera-B+ teacher;
- full eligible SA-V, T8, five epochs, up to three objects;
- seven correction clicks;
- Dice weight 20, focal weight 1, weight decay 0.1;
- learning rates linearly scaled from the paper's batch 256 to batch four.

Q2 reached 55.6 val and 58.0 test J&F. This is a material recovery from the
10--25-J&F early runs and shows that recipe fidelity matters, but it remains
below both Q0 and the selected TinyViT reference.

Q2 is not a full EdgeTAM reproduction. The official video stage mixes SA-V,
10% SA-1B, DAVIS, MOSE, and YouTube-VOS for about 130k iterations before
longer T16/T32 stages, and it uses RepViT-M1 rather than TinyViT-21M.

### 8.3 Full-SA-V EM1--EM3 curriculum

The active feasibility flow is described in
[`tinyvit21_edgetam_memory_v1.md`](tinyvit21_edgetam_memory_v1.md). It uses
the prepared full-SA-V manifest, the best TV21 task checkpoint for the image
and mask path, and official EdgeTAM temporal components.

| Stage | T | Epochs | Trainable modules | Initialization | Main losses |
|---|---:|---:|---|---|---|
| EM1 | 4 | 2 | EdgeTAM attention, memory encoder, Perceiver, pointer/temporal params | TV21 + official temporal stack | task 1, memory 0.5, logits 2 |
| EM2 | 8 | 5 | TinyViT encoder + full EdgeTAM temporal stack | EM1 | task 1, image 1, memory 0.5, logits 2 |
| EM3 | 16 | 2 | EdgeTAM temporal stack; TinyViT frozen again | EM2 | task 1, memory 0.25, logits 1 |

All stages use three sampled objects, mixed point/box prompts, random
correction frames, BF16, global batch four, resumable checkpoints, W&B, and
the same TensorBoard/run directories on resume. Object-pointer KD remains
disabled because it is not exposed reliably on every teacher path.

The 50,337-video manifest gives approximately 12,585 optimizer updates per
epoch at global batch four. The full 2+5+2 curriculum schedules about 113,265
updates. More raw frames increase clip diversity, but do not change updates
per epoch unless the manifest length or dataset multiplier changes.

At the snapshot:

| Stage | State | Persisted progress | Trainable parameters | Evaluation |
|---|---|---:|---:|---|
| EM1 | training complete | epoch 2 / 25,170 updates | 4,775,392 | intentionally deferred |
| EM2 | active | at least epoch 2 / 25,170 updates | 25,561,040 | pending |
| EM3 | not started | 0 | pending construction | pending |

This is an engineering feasibility result only: coherent official temporal
initialization loads, gradients reach the intended modules, and multi-epoch
training persists. It is not yet an accuracy or latency result. The first
research decision comes after EM3 full val/test.

## 9. Consolidated frontier

| Method | Val/Test J&F | N=8 FPS | What it establishes |
|---|---|---:|---|
| Selected TV21 | 72.4/74.7 | 14.96 legacy | quality reference |
| Persistent runtime bucket | reference model behavior; min mask IoU 0.969 | 22.07 | deployable systems gain without learned sharing |
| MO0 four-layer dense training | 69.4/72.4 | 24.16 | dense training alone gives a modest gain |
| MO1--MO3 two-layer memory | about 57/58.5 | 34--35 | memory-depth speed is real but quality is unacceptable |
| MX2/MX5 decoder slots | 72.3/74.6 | about 24.4 | decoder multiplexing can preserve quality |
| MX4/MX6--MX8 shared K/V | about 63.6/about 60.3 | about 49 | shared temporal compute gives 2.2× speed but loses identity |
| MX13--MX28 low-rank residual screens | completed screens about 53 J&F | about 46--48 | post-attention residual is insufficient |
| EdgeTAM Q2 | 55.6/58.0 | not yet isolated | recipe fidelity recovers part of temporal quality |
| EM1--EM3 | pending | pending | full-data EdgeTAM/TinyViT feasibility in progress |

The strongest established conclusions are:

1. Persistent bounded histories solve most of the original bucket
   orchestration problem and reduce N=8 latency by 32.2%.
2. A learned slot-eight decoder can emit multiple masks per transformer call
   while retaining 99.9% full tracking quality.
3. Fully shared memory K/V approximately doubles N=8 throughput but removes
   object-specific temporal information.
4. More epochs, longer rollout, stronger memory/logit KD, and a cheap
   post-attention residual do not repair that representational loss.
5. The official EdgeTAM trainer path is viable; adapting it to TinyViT remains
   a quality question pending the EM curriculum.

## 10. Recommended next architecture and experiment

The evidence points to a hybrid temporal representation rather than another
pure shared-K/V KD sweep:

```text
shared scene memory K/V
  + a small set of private object K/V tokens
  + object-conditioned query or per-layer modulation
  + the quality-preserving slot-eight decoder
  + an external exact legacy bypass for N < 4
```

This differs from MX13--MX28 in where private information is introduced. The
next intervention should preserve object identity inside every attention
layer instead of adding a pooled correction only after shared attention.

A controlled first matrix should vary only private temporal bandwidth, for
example 2/4/8 private tokens or low-rank K/V residuals per object, using MX5
as the quality parent. It should keep the same seed, T8 cohort, update budget,
teacher, and full evaluation gates. Selection order should be:

1. fixed-cohort val J&F retention at least 95%;
2. synchronized-path mask IoU at least 0.95;
3. N=8 faster than the 22.07-FPS persistent reference;
4. full SA-V val, then test, for only screen-passing candidates;
5. repeated latency and identity testing on a broader held-out 8/16/32-object
   cohort before deployment.

Until such a candidate passes, the recommended engineering policy is the
persistent runtime bucket for N>=4 and the exact legacy path for smaller or
asynchronous sessions. Learned shared-K/V checkpoints remain research
artifacts, not deployment candidates.
