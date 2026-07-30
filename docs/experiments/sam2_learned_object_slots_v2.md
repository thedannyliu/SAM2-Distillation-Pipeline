# SAM2 Learned Object Slots v2: 20-Hour Recovery

## Question

Can longer temporal rollout and stronger behavior distillation move the
capacity-eight learned multiplexer from the v1 quality endpoint (MX2) toward
the v1 speed endpoint (MX4) while retaining at least 95% of selected TV21M
tracking quality?

All lanes initialize from
`sam2_object_slots_v1/MX2_slot8_decoder_kd_3ep`, the only v1 learned decoder
with 99.9% full quality retention and perfect synchronized-path mask
agreement. The reference remains val/test J&F 72.4/74.7 and persistent-bucket
N=1/N=8 FPS 71.36/22.07.

## Controlled 4-node matrix

Every lane uses the same dense-eight-object cohort, seed, global batch four,
T8 clips, five epochs, bf16, four frozen standard SAM2 memory-attention
layers, and the same TV21M teacher. The estimated wall time is 16–20 hours per
4×H100 node, including full val, full test, and isolated latency.

| Node | Variant | Shared K/V | Memory KD | Mask-logit KD | Question |
|---:|---|---|---:|---:|---|
| 1 | `MX5_slot8_decoder_t8_logits2_5ep` | no | 0 | 2 | Does longer T8 decoder training improve robustness without temporal compression? |
| 2 | `MX6_slot8_sharedkv_t8_mem1_5ep` | yes | 1 | 2 | Is balanced temporal KD enough to recover shared-K/V identity? |
| 3 | `MX7_slot8_sharedkv_t8_mem4_5ep` | yes | 4 | 2 | Does strong memory matching recover J&F? |
| 4 | `MX8_slot8_sharedkv_t8_mem1_logits4_5ep` | yes | 1 | 4 | Is propagated-mask behavior more useful than stronger feature matching? |

Only the KD weighting differs between nodes 2–4. Node 1 is the matched
high-quality continuation control. This matrix tests whether the v1 shared-K/V
failure is an optimization/rollout problem before adding a new per-object
residual architecture.

Each node executes:

```text
5-epoch T8 dense-object train on 4 H100
  -> full SA-V validation
  -> full SA-V test
  -> N=1/2/4/8 latency on one isolated H100
  -> W&B + TensorBoard + comparison CSV/Markdown
```

## Promotion gates

A run is promotable only when:

1. val and test J&F each retain at least 95%;
2. synchronized-path minimum mask IoU is at least 0.95;
3. N=1 FPS retains at least 95%;
4. N=8 FPS exceeds 22.07.

Select using validation quality and N=8 latency. Test remains descriptive.
If every shared-K/V lane remains below 95% quality, stop increasing KD and
implement shared K/V plus an explicit low-rank per-object temporal residual.

## Company commands

Pull once because `/user-volume/repo` is shared:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only
chmod +x scripts/company/62_run_sam2_object_slots_v2.sh
mkdir -p /user-volume/sam2_object_slots_v2_logs
echo "Repository sync status: $?"
```

Node 1:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 \
SKIP_DONE=1 \
scripts/company/62_run_sam2_object_slots_v2.sh \
  run MX5_slot8_decoder_t8_logits2_5ep 2>&1 | \
tee /user-volume/sam2_object_slots_v2_logs/MX5_slot8_decoder_t8_logits2_5ep.log
echo "MX5 status: ${PIPESTATUS[0]}"
```

Node 2:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 \
SKIP_DONE=1 \
scripts/company/62_run_sam2_object_slots_v2.sh \
  run MX6_slot8_sharedkv_t8_mem1_5ep 2>&1 | \
tee /user-volume/sam2_object_slots_v2_logs/MX6_slot8_sharedkv_t8_mem1_5ep.log
echo "MX6 status: ${PIPESTATUS[0]}"
```

Node 3:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 \
SKIP_DONE=1 \
scripts/company/62_run_sam2_object_slots_v2.sh \
  run MX7_slot8_sharedkv_t8_mem4_5ep 2>&1 | \
tee /user-volume/sam2_object_slots_v2_logs/MX7_slot8_sharedkv_t8_mem4_5ep.log
echo "MX7 status: ${PIPESTATUS[0]}"
```

Node 4:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 \
SKIP_DONE=1 \
scripts/company/62_run_sam2_object_slots_v2.sh \
  run MX8_slot8_sharedkv_t8_mem1_logits4_5ep 2>&1 | \
tee /user-volume/sam2_object_slots_v2_logs/MX8_slot8_sharedkv_t8_mem1_logits4_5ep.log
echo "MX8 status: ${PIPESTATUS[0]}"
```

After all nodes finish:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
scripts/company/62_run_sam2_object_slots_v2.sh summarize 2>&1 | \
tee /user-volume/sam2_object_slots_v2_logs/summary.log
echo "Summary status: ${PIPESTATUS[0]}"
```

## Results

| Variant | Val J&F | Test J&F | Min quality | Learned mask IoU | N1 FPS | N8 FPS | N8 gain | Promote |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MX5 | pending | pending | pending | pending | pending | pending | pending | pending |
| MX6 | pending | pending | pending | pending | pending | pending | pending | pending |
| MX7 | pending | pending | pending | pending | pending | pending | pending | pending |
| MX8 | pending | pending | pending | pending | pending | pending | pending | pending |
