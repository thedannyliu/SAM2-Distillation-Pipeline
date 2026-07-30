# SAM2 Learned Object Slots and Shared Memory K/V v1

## Question and acceptance criteria

Can the selected TinyViT-21M SAM2 tracker replace the remaining per-object
decoder and memory-attention work with fixed-capacity learned slots, while
keeping at least 95% of tracking quality?

The selected quality reference is `tinyvit_max_jf_v1/tv21`: full SA-V val
J&F 72.4 and test J&F 74.7. The selected runtime reference is the persistent
four-object bucket result from `sam2_multiobject_bucket_mx1p_v1`, which
measured 71.36 FPS at one object and 22.07 FPS at eight objects. A learned run
is promotable only when all four conditions hold:

1. full val and test J&F each retain at least 95% of the selected model;
2. the synchronized learned path has minimum per-mask IoU at least 0.95
   against the selected legacy path over 128 dense-cohort frames;
3. one-object FPS retains at least 95% of the runtime reference;
4. eight-object FPS is higher than the runtime reference.

The primary optimization metric is eight-object propagation milliseconds per
frame. Test J&F is reported but not used to choose between runs until a
candidate passes the validation gate.

## Evidence that motivates the design

The unmodified selected model scaled from 72.47 FPS at one object to 14.96 FPS
at eight objects. Persistent four-object buckets kept one-object performance
within 1.5% and improved four/eight-object FPS by 42.0%/47.5%, reaching 22.07
FPS at eight objects. Its minimum comparison-mask IoU was 0.969, which is
within the current 95% tolerance.

Reducing SAM2 memory attention from four layers to two reached about 35 FPS at
eight objects, but test J&F fell from 72.4 to about 58.5. Mask-logit and
memory-feature distillation did not recover the loss. Therefore this study
keeps the selected four-layer temporal path and changes how objects share it.

## Relation to SAM3.1 multiplex

The official SAM3.1 implementation assigns tracks to persistent fixed slots,
stores state in bucket space, shares image features across the slots, and
uses a multiplex mask decoder that emits one mask per slot. The number of
expensive tracker batches is therefore the number of buckets instead of the
number of objects:

```text
legacy SAM2 batch       = number of objects
multiplex tracker batch = ceil(number of objects / slot capacity)
```

The relevant upstream references are the
[release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md),
[multiplex state](https://github.com/facebookresearch/sam3/blob/main/sam3/model/multiplex_utils.py),
[bucket-space tracker](https://github.com/facebookresearch/sam3/blob/main/sam3/model/video_tracking_multiplex.py),
and
[multiplex mask decoder](https://github.com/facebookresearch/sam3/blob/main/sam3/model/multiplex_mask_decoder.py).

This implementation is a controlled SAM2 port of those principles, not a
weight-compatible copy of SAM3.1:

- `LearnedObjectSlotDecoder` superposes synchronized per-object spatial
  features into fixed slots, adds learned slot tokens, and calls the existing
  pretrained SAM2 two-way mask transformer once per bucket.
- `SharedSlotMemoryAttention` additionally superposes raw object memories
  before the existing memory-attention K/V projections. One four-layer
  memory-attention path then serves each bucket.
- At inference, point-prompted frames and sessions below four objects use the
  selected legacy path. This protects interactive and one-object latency.
- During training, unprompted frames use slots at every available object
  count so small sampled batches still teach the conditional slot tensors.
  Prompted legacy fallbacks carry a zero-valued autograd anchor; this changes
  no prediction but keeps frozen-base slot-only optimization and DDP valid.
- Full SA-V evaluation uses learned buckets only when object histories are
  synchronized. Objects introduced on different frames explicitly fall back
  to the selected legacy tracker and are counted in
  `bucket_execution_stats`; they are never silently forced into an invalid
  bucket.
- Existing SAM2 image encoder, memory encoder, memory-attention weights, mask
  decoder weights, prompt encoder, and object-pointer head stay frozen. Only
  new slot tensors are trained.

The shared-K/V intervention is deliberately aggressive. It has the greatest
latency upside, but the learned slot codes must preserve which memory belongs
to which object after compression. Comparing decoder-only and shared-K/V runs
separates mask-decoder savings from memory-attention savings.

## Four independent 4×H100 experiments

All lanes use the same dense-eight-object SA-V cohort, global batch four,
four-frame clips, three epochs, bf16, the same seed, task loss, and
propagated-mask logit KD from the selected TinyViT-21M teacher. Object-pointer
KD is disabled because the current teacher-output contract does not expose
`teacher_obj_ptr` on every sampled path. Shared-K/V lanes additionally use
memory-feature KD. Runs resume in the same checkpoint/TensorBoard/W&B
directories after interruption.

| Node | Variant | Slot capacity | Shared memory K/V | Trainable tensors | Question |
|---:|---|---:|---|---|---|
| 1 | `MX1_slot4_decoder_kd_3ep` | 4 | no | slot decoder | Does one mask-transformer call per four objects beat the runtime bucket? |
| 2 | `MX2_slot8_decoder_kd_3ep` | 8 | no | slot decoder | Does a larger decoder bucket improve N=8 enough without excess interference? |
| 3 | `MX3_slot4_sharedkv_kd_3ep` | 4 | yes | slot decoder + memory slot codes | What is the quality/latency effect of true shared K/V at capacity four? |
| 4 | `MX4_slot8_sharedkv_kd_3ep` | 8 | yes | slot decoder + memory slot codes | What is the maximum single-bucket N=8 speedup? |

Each node runs:

```text
3-epoch dense8 train on 4 H100
  -> full SA-V val
  -> full SA-V test
  -> N=1/2/4/8 latency on one isolated H100
  -> W&B + TensorBoard + per-run CSV + central comparison
```

Expected wall time is approximately 7–10 hours per node and stays inside a
12-hour allocation under the storage throughput observed in prior runs.

### Runtime preflight record

The first 2026-07-29 launches stopped before any optimizer update. MX2 exposed
an unavailable optional `teacher_obj_ptr` target, while MX3/MX4 exposed the
EdgeTAM `num_spatial_mem` interface. After those were corrected, MX2 reached
backward but its first prompted batch used only the frozen legacy head. The
current configuration disables object-pointer KD, forwards
`num_spatial_mem`, trains slots on unprompted small-object batches, and keeps
prompt fallbacks differentiable with a numerical-zero anchor. Retries reuse
the original W&B IDs and run directories.

The completed runs later exposed one evaluation-only compatibility issue:
EdgeTAM inference state does not contain the optional official-SAM2
`frames_tracked_per_obj` bookkeeping map. Commit `c398c34` made the persistent
bucket adapter update that map only when the predictor provides it. No
training weights changed, and all four existing checkpoints resumed directly
into the unfinished VOS and latency stages.

## Company data handling

No additional labels or teacher cache are required. The runner deterministically
selects SA-V videos with at least eight visible objects across the sampled
training frames and writes only object IDs to
`runs/sam2_object_slots_v1/cohorts/dense8_train_ids.txt`. Raw frames and masks
remain in the data lake. Training checkpoints and intermediate evaluations
remain under `/danny-dataset/sam2_distill/runs`; `/group-volume` is used only
when that legacy mount contains the existing reference runs. Do not copy raw
company videos, masks, or intermediate checkpoints into the Git repository.

For a stronger company-only conclusion, rerun the final promoted checkpoint
on a held-out internal cohort containing multiple videos at 8/16/32 objects.
The current SA-V validation audit found only one video with eight objects on
one shared annotated frame, so its latency estimate is causal and repeatable
but not statistically broad.

## Company commands

Pull the implementation once before opening the four node terminals:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only origin main
chmod +x scripts/company/61_run_sam2_object_slots.sh
echo "Repository sync status: $?"
```

Run one block in each independent 4×H100 node. Every command stays in the
foreground and records live output.

Node 1:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
mkdir -p /user-volume/sam2_object_slot_logs
GPUS=0,1,2,3 \
SKIP_DONE=1 \
scripts/company/61_run_sam2_object_slots.sh \
  run MX1_slot4_decoder_kd_3ep 2>&1 | \
tee /user-volume/sam2_object_slot_logs/MX1_slot4_decoder_kd_3ep.log
echo "MX1 status: ${PIPESTATUS[0]}"
```

Node 2:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
mkdir -p /user-volume/sam2_object_slot_logs
GPUS=0,1,2,3 \
SKIP_DONE=1 \
scripts/company/61_run_sam2_object_slots.sh \
  run MX2_slot8_decoder_kd_3ep 2>&1 | \
tee /user-volume/sam2_object_slot_logs/MX2_slot8_decoder_kd_3ep.log
echo "MX2 status: ${PIPESTATUS[0]}"
```

Node 3:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
mkdir -p /user-volume/sam2_object_slot_logs
GPUS=0,1,2,3 \
SKIP_DONE=1 \
scripts/company/61_run_sam2_object_slots.sh \
  run MX3_slot4_sharedkv_kd_3ep 2>&1 | \
tee /user-volume/sam2_object_slot_logs/MX3_slot4_sharedkv_kd_3ep.log
echo "MX3 status: ${PIPESTATUS[0]}"
```

Node 4:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
mkdir -p /user-volume/sam2_object_slot_logs
GPUS=0,1,2,3 \
SKIP_DONE=1 \
scripts/company/61_run_sam2_object_slots.sh \
  run MX4_slot8_sharedkv_kd_3ep 2>&1 | \
tee /user-volume/sam2_object_slot_logs/MX4_slot8_sharedkv_kd_3ep.log
echo "MX4 status: ${PIPESTATUS[0]}"
```

The default runner discovers either `/danny-dataset` or the current
`/group-volume/danny-dataset` legacy mount. Override `SAM2D_ROOT`, `SAV_ROOT`,
or `REFERENCE_LATENCY_DIR` only when automatic discovery prints the wrong
path.

## Result collection

After any node finishes, this command is safe to rerun and shows partial or
complete results:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
scripts/company/61_run_sam2_object_slots.sh summarize 2>&1 | \
tee /user-volume/sam2_object_slot_logs/summary.log
echo "Summary status: ${PIPESTATUS[0]}"
```

Primary artifacts:

```text
<SAM2D_ROOT>/runs/sam2_object_slots_v1/summary.csv
<SAM2D_ROOT>/runs/sam2_object_slots_v1/comparison/object_slot_results.md
<SAM2D_ROOT>/runs/sam2_object_slots_v1/comparison/object_slot_results.csv
<SAM2D_ROOT>/runs/sam2_object_slots_v1/comparison/object_slot_results.json
```

The comparison report ranks only runs that pass all four gates. Full val/test
J&F measures the deployable hybrid policy, while the 128-frame dense
verification prevents asynchronous fallback from hiding a poor learned path.
`bucket_execution_stats` in each VOS summary records how many sessions used
learned buckets, the small-object legacy path, or the asynchronous legacy
fallback.

The company image does not necessarily provide `rg`. If the optional log scan
prints `xargs: rg: No such file or directory`, that is a missing audit utility,
not an experiment failure; the pipeline status, metrics, and comparison
artifacts above remain authoritative.

## Result table

This table answers whether decoder sharing alone is enough and whether true
shared memory K/V advances the quality–latency frontier.

All four 2026-07-30 pipelines completed train, full SA-V val, full SA-V test,
and isolated N=1/2/4/8 latency. None passes every promotion gate.

| Variant | Val J&F | Test J&F | Min quality retention | Learned mask IoU | N1 FPS | N1 retention | N8 FPS | N8 gain vs 22.07 | N8 ms | Peak MB | Promote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MX1 slot4 decoder | 72.3 | 74.6 | 0.999 | 0.000 | 59.68 | 0.836 | 21.68 | -1.8% | 46.13 | 11,977 | no |
| MX2 slot8 decoder | 72.3 | 74.6 | 0.999 | 1.000 | 61.65 | 0.864 | 24.46 | +10.8% | 40.88 | 12,513 | no |
| MX3 slot4 shared K/V | 63.8 | 60.9 | 0.815 | 0.413 | 60.66 | 0.850 | 34.52 | +56.4% | 28.97 | 11,581 | no |
| MX4 slot8 shared K/V | 63.6 | 60.2 | 0.806 | 1.000 | 59.87 | 0.839 | 49.03 | +122.1% | 20.39 | 11,662 | no |

The image-only metrics are effectively unchanged across all four runs
(validation mIoU 0.8403/AP 0.7166 and test mIoU 0.8391/AP 0.7191), as expected
from the frozen image path. The differences are temporal:

- **MX1 is rejected.** Hybrid full-set J&F is retained, but minimum learned
  mask IoU is zero and N=8 is slightly slower than the persistent runtime
  bucket. The full-set score is therefore not evidence that its synchronized
  slot4 path is correct; legacy/asynchronous fallback protects much of that
  evaluation.
- **MX2 is the best learned-decoder result.** It retains 99.9% of full
  tracking quality, has perfect measured learned-path mask agreement, and
  improves N=8 FPS by 10.8% over the persistent bucket. It is not promoted
  because N=1 retains only 86.4% of reference FPS rather than the required
  95%.
- **MX3 confirms that shared K/V creates real speed but loses object
  identity.** N=8 improves 56.4%, while minimum quality retention falls to
  81.5% and learned-path mask IoU to 0.413.
- **MX4 establishes the latency upper bound.** One slot8 shared-K/V bucket
  reaches 49.03 FPS at N=8, 2.22× the persistent-bucket reference, but full
  tracking retention is only 80.6%. Its learned mask IoU of 1.0 does not mean
  reference-model quality is preserved: that check compares bucket and
  legacy execution inside the same already-degraded trained model.

The causal conclusion is that decoder multiplexing is compatible with the
selected SAM2 behavior, while fully collapsing temporal memory K/V removes
information needed for persistent object identity. The useful frontier is
between MX2 and MX4, not at either extreme.

## Decision and next experiment

Do not deploy any v1 learned-slot checkpoint. Keep the existing persistent
runtime bucket as the deployment candidate and use MX2 as the parent for the
next learned experiment.

The next controlled intervention should preserve an explicit per-object
temporal residual while sharing most K/V computation:

```text
shared bucket K/V
  + low-rank per-object residual K/V
  + temporal rollout distillation from the selected TV21M teacher
```

Run capacity eight first because MX2 shows that its decoder slot assignment is
stable. Compare three residual ranks against MX2 and MX4 under the same seed,
full val/test, and latency cohort. Select on validation J&F retention before
examining test. In parallel, profile MX2 with a true external legacy bypass
for N<4 and a paired same-checkpoint legacy/bucket benchmark; this determines
whether its N=1 loss is avoidable orchestration overhead rather than learned
model cost.

Before any promotion, repeat the latency and identity tests on a company-only
held-out cohort with multiple 8/16/32-object videos. The current N=8 latency
table has three repetitions but only one eligible SA-V video.
