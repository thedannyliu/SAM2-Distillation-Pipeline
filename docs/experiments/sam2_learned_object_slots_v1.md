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
- Point-prompted frames and sessions below four objects use the selected
  legacy path. This protects interactive and one-object latency.
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
four-frame clips, three epochs, bf16, the same seed, task loss, propagated-mask
logit KD, and object-pointer KD from the selected TinyViT-21M teacher. They
resume in the same checkpoint/TensorBoard/W&B directories after interruption.

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

## Result table

This table answers whether decoder sharing alone is enough and whether true
shared memory K/V advances the quality–latency frontier.

| Variant | Val J&F | Test J&F | Quality retention | N1 FPS retention | N8 FPS | N8 gain vs 22.07 | N8 ms | Peak MB | Promote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MX1 slot4 decoder | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| MX2 slot8 decoder | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| MX3 slot4 shared K/V | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| MX4 slot8 shared K/V | pending | pending | pending | pending | pending | pending | pending | pending | pending |
