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

Recorded from
`/group-volume/danny-dataset/sam2_distill/runs/sam2_object_slots_v2`.
MX5–MX8 have their pipeline-complete marker and all expected checkpoint,
validation, test, and latency artifacts.

| Variant | Status | Val J&F | Test J&F | Min quality | Learned mask IoU | N1 FPS | N1 retention | N8 FPS | N8 gain | N8 ms | Promote |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MX5 | complete | 72.30 | 74.60 | 0.999 | not reported | 60.11 | 0.842 | 24.40 | +10.5% | 40.99 | no |
| MX6 | complete | 63.60 | 60.30 | 0.807 | 1.000 | 60.28 | 0.845 | 49.06 | +122.3% | 20.38 | no |
| MX7 | complete | 63.70 | 60.50 | 0.810 | 1.000 | 60.88 | 0.853 | 48.72 | +120.7% | 20.52 | no |
| MX8 | complete | 63.60 | 60.30 | 0.807 | 1.000 | 59.26 | 0.830 | 48.79 | +121.0% | 20.50 | no |

The N8 gain is relative to the 22.07 FPS persistent-bucket reference. The
minimum quality column is the lower of val and test retention relative to
72.4/74.7.

### Multiplex latency

| Variant | N1 ms / FPS | N2 ms / FPS | N4 ms / FPS | N8 ms / FPS | N8 peak memory |
|---|---:|---:|---:|---:|---:|
| MX5 | 16.64 / 60.11 | 20.13 / 49.68 | 26.59 / 37.61 | 40.99 / 24.40 | 12497 MB |
| MX6 | 16.59 / 60.28 | 20.00 / 49.99 | 18.01 / 55.52 | 20.38 / 49.06 | 11698 MB |
| MX7 | 16.42 / 60.88 | 20.23 / 49.43 | 18.02 / 55.50 | 20.52 / 48.72 | 11680 MB |
| MX8 | 16.88 / 59.26 | 20.03 / 49.93 | 18.22 / 54.89 | 20.50 / 48.79 | 11698 MB |

MX5 fails the configured N2, N4, and N8 relative-latency targets. MX6–MX8 pass
N4 and N8 but fail N2. All four N1 retention values are only 83.0–85.3%,
below the 95% gate.

### Quality localization

All four completed variants preserve the image-only result:

| Split | mIoU | AP |
|---|---:|---:|
| SA-V val | 0.8403 | 0.7166 |
| SA-V test | 0.8391 | 0.7191 |

MX5 also preserves temporal quality at val/test J&F 72.3/74.6, or 99.9% of
the reference. The loss is therefore specific to shared K/V: MX6 reaches
63.6/60.3, MX7 reaches 63.7/60.5, and MX8 reaches 63.6/60.3. A learned-path
mask IoU of 1.0 means the synchronized bucket and legacy execution paths agree
for the same trained model; it does not mean that the model retains the
72.4/74.7 reference tracking quality. MX5 learned-path IoU was not included
in the recorded artifact output.

## Interpretation and decision

The speed hypothesis succeeds but the quality-recovery hypothesis fails.
Shared K/V raises N8 throughput from 22.07 FPS to about 49 FPS, a 2.21–2.22×
gain, while keeping image segmentation unchanged. However, every shared-K/V
lane retains only 80.7–81.0% of reference video quality and loses 14.7–17.0%
of reference N1 throughput.

MX5 is the matched high-quality control. Longer T8 decoder training preserves
99.9% of full tracking quality but reaches only 24.40 FPS at N8, a 10.5% gain
over the runtime reference. Its N1 throughput is also only 84.2% of the
reference. This confirms the central trade-off: keeping per-object temporal
state preserves quality but leaves most of the multiplex latency, while fully
sharing temporal K/V removes that latency at an unacceptable quality cost.

The controlled KD sweep provides no evidence that more training or a larger
distillation weight repairs the failure:

- Raising memory KD from 1 to 4 changes val/test J&F by only +0.1/+0.2 and
  slightly reduces N8 FPS.
- Raising mask-logit KD from 2 to 4 produces no J&F improvement and lowers N1
  FPS.
- These endpoints reproduce the v1 shared-K/V quality regime despite T8
  rollout and five epochs.

Therefore reject MX6–MX8 for deployment and stop pure shared-K/V KD-weight
sweeps. The evidence points to a representational bottleneck: fully shared
temporal K/V removes object-specific temporal information that the current
decoder cannot reconstruct. The next architecture should retain shared K/V
for the common scene context while adding an explicit low-rank per-object
temporal residual. Use the quality-preserving decoder path as the parent and
compare residual capacity under the same N1, N8, and full-SA-V gates.

That follow-up is implemented in
[`sam2_object_slots_v3.md`](sam2_object_slots_v3.md), using the completed MX5
T8 decoder checkpoint as the stronger quality parent.

MX5 is the best v2 quality endpoint, but it fails the N1 retention and
relative-latency gates. MX7 is the best shared-K/V quality endpoint, but its
0.810 minimum retention is far below the 0.95 promotion threshold. No v2
variant is promoted.
