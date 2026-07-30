# SAM2 Object-Specific Temporal Residuals v3

## Question

Can a small per-object temporal residual recover at least 95% of TV21M
tracking quality while retaining most of the slot-eight shared-K/V speed?

The v2 result separates the problem cleanly:

- decoder-only MX5 retains 99.9% val/test J&F but reaches only 24.40 FPS at
  eight objects;
- fully shared-K/V MX6–MX8 reach about 49 FPS but retain only 80.7–81.0% of
  val/test J&F;
- longer T8 rollout and stronger memory or mask-logit KD do not recover the
  shared-K/V quality.

This points to missing object-specific temporal information rather than
insufficient optimization.

## Intervention

`SharedSlotMemoryAttention` still runs the frozen four-layer SAM2 attention
once per slot-eight bucket. Instead of repeating the resulting temporal
feature unchanged across all objects, v3 adds a cheap object-specific
residual:

```text
bucket-shared four-layer attention output
    + low-rank aligned per-object spatial-memory projection
    + optional low-rank object-pointer projection
    -> learned slot decoder
```

The spatial path excludes object-pointer tokens, adds the existing spatial
and temporal positional encoding, projects each memory token into a low-rank
space, groups tokens by the current feature-grid length, and averages
corresponding low-rank tokens over memory frames:

```text
64-dimensional positioned object memory
    -> rank r
    -> aligned temporal aggregation
    -> 256-dimensional SAM feature
```

The optional pointer path pools each object's pointer tokens and applies a
separate low-rank projection. The final projection in each new path is
zero-initialized, so model initialization exactly starts from the shared-K/V
output. The frozen SAM2 attention stack and selected TV21M weights do not
change.

This is not another per-object attention layer. It avoids an additional
query-by-memory attention matrix and adds work proportional to
`objects × spatial tokens × rank`.

Implementation:

- `sam2_distill/models/sam2_object_slots.py`
- `tools/train/run_sam2_task_training.py`
- `scripts/company/49_run_edgetam_memory_ablation.sh`
- `scripts/company/63_run_sam2_object_slots_v3.sh`

## Controlled four-node matrix

All four lanes initialize from
`sam2_object_slots_v2/MX5_slot8_decoder_t8_logits2_5ep`, use slot capacity eight,
shared K/V, standard frozen four-layer memory attention, T8 clips, five
epochs, global batch four, bf16, memory KD 1, mask-logit KD 2, and the same
dense-eight SA-V cohort.

| Node | Variant | Spatial rank | Pointer rank | Added parameters | Question |
|---:|---|---:|---:|---:|---|
| 1 | `MX9_slot8_sharedkv_r4_t8_5ep` | 4 | 0 | 1,408 | Is minimal spatial identity bandwidth sufficient? |
| 2 | `MX10_slot8_sharedkv_r8_t8_5ep` | 8 | 0 | 2,688 | Does doubling spatial rank materially recover J&F? |
| 3 | `MX11_slot8_sharedkv_r16_t8_5ep` | 16 | 0 | 5,248 | Does quality continue to scale with residual capacity? |
| 4 | `MX12_slot8_sharedkv_r8_ptr8_t8_5ep` | 8 | 8 | 5,376 | Does an explicit pointer identity bypass help beyond rank-eight spatial state? |

The parameter counts include LayerNorm and the two bias-free projections in
each enabled residual path. MX10 is the direct control for MX12.

## Decision logic

A run is promotable only when:

1. val and test J&F each retain at least 95% of 72.4/74.7;
2. synchronized learned-path minimum mask IoU is at least 0.95;
3. N=1 FPS retains at least 95% of 71.36;
4. N=8 FPS exceeds 22.07.

Interpret the matrix as follows:

- rank 4 fails but rank 8/16 recover: choose the smallest passing rank;
- all spatial ranks plateau below 95% but MX12 improves: object-pointer
  identity is the missing signal;
- rank 16 still remains near 81% and MX12 does not improve: the aligned
  additive residual is too weak; stop rank scaling and restore a sparse
  per-object attention path;
- quality passes but N8 approaches decoder-only speed: residual computation
  restored too much object-dependent work; reduce rank or temporal frequency.

Test J&F remains descriptive until a candidate passes the validation gate.

## Four company node commands

Pull once because `/user-volume/repo` is shared by the four nodes:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only
chmod +x scripts/company/63_run_sam2_object_slots_v3.sh
mkdir -p /user-volume/sam2_object_slots_v3_logs
scripts/company/63_run_sam2_object_slots_v3.sh describe
echo "Repository/preflight status: $?"
```

Node 1:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 \
SKIP_DONE=1 \
scripts/company/63_run_sam2_object_slots_v3.sh \
  run MX9_slot8_sharedkv_r4_t8_5ep 2>&1 | \
tee /user-volume/sam2_object_slots_v3_logs/MX9_slot8_sharedkv_r4_t8_5ep.log
echo "MX9 status: ${PIPESTATUS[0]}"
```

Node 2:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 \
SKIP_DONE=1 \
scripts/company/63_run_sam2_object_slots_v3.sh \
  run MX10_slot8_sharedkv_r8_t8_5ep 2>&1 | \
tee /user-volume/sam2_object_slots_v3_logs/MX10_slot8_sharedkv_r8_t8_5ep.log
echo "MX10 status: ${PIPESTATUS[0]}"
```

Node 3:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 \
SKIP_DONE=1 \
scripts/company/63_run_sam2_object_slots_v3.sh \
  run MX11_slot8_sharedkv_r16_t8_5ep 2>&1 | \
tee /user-volume/sam2_object_slots_v3_logs/MX11_slot8_sharedkv_r16_t8_5ep.log
echo "MX11 status: ${PIPESTATUS[0]}"
```

Node 4:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 \
SKIP_DONE=1 \
scripts/company/63_run_sam2_object_slots_v3.sh \
  run MX12_slot8_sharedkv_r8_ptr8_t8_5ep 2>&1 | \
tee /user-volume/sam2_object_slots_v3_logs/MX12_slot8_sharedkv_r8_ptr8_t8_5ep.log
echo "MX12 status: ${PIPESTATUS[0]}"
```

After all nodes finish:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
scripts/company/63_run_sam2_object_slots_v3.sh summarize 2>&1 | \
tee /user-volume/sam2_object_slots_v3_logs/summary_$(date +%Y%m%d_%H%M%S).log
echo "Summary status: ${PIPESTATUS[0]}"
cat /group-volume/danny-dataset/sam2_distill/runs/sam2_object_slots_v3/comparison/object_slot_results.md
```

## Results

| Variant | Val J&F | Test J&F | Learned mask IoU | N1 FPS | N8 FPS | N8 ms | Promote |
|---|---:|---:|---:|---:|---:|---:|---|
| MX9 | pending | pending | pending | pending | pending | pending | pending |
| MX10 | pending | pending | pending | pending | pending | pending | pending |
| MX11 | pending | pending | pending | pending | pending | pending | pending |
| MX12 | pending | pending | pending | pending | pending | pending | pending |
