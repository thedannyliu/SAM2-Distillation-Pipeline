# SAM2 priority-18 full-data suite

## Decision

The original 50 full-data experiments have not started. With six remaining
4×H100 nodes, run 18 of those registered variants first instead of launching
all 50. The selected checkpoints and W&B runs stay under
`sam2_full_data_50_v1`, so the other 32 variants remain resumable later.

Each node runs three long trainings sequentially. A failed experiment is
recorded and does not stop the next experiment. Every candidate uses its
original 5- or 8-epoch schedule, a fixed 64-video validation gate, and
three-repetition N=1/2/4/8 latency measurement.

## Memory and EdgeTAM-style compression

The question is whether two-layer temporal compression can retain at least
95% of the TV21 J&F reference. Standard-memory controls are necessary because
previous results showed that image metrics can remain healthy while temporal
tracking collapses.

| Node | Variant | Causal role |
|---:|---|---|
| 1 | `FD46_mem4_t8_decmem_8ep` | Functional four-layer upper bound |
| 1 | `FD47_mem2_t8_decmem_8ep` | Isolate reducing four layers to two |
| 1 | `FD48_mem2_t8_joint_logits2_8ep` | Test whether joint/logit KD recovers two layers |
| 2 | `FD04_tv21_t8_joint_logits1_8ep` | Same-horizon joint-training control |
| 2 | `FD49_edgetam2_temporal_logits2_8ep` | EdgeTAM two-layer Perceiver temporal training |
| 2 | `FD50_edgetam2_joint_img_logits2_8ep` | Joint encoder–temporal interface recovery |
| 3 | `FD01_tv21_t4_decmem_5ep` | Short-context full-data anchor |
| 3 | `FD03_tv21_t8_logits1_8ep` | Longer context plus behavior KD |
| 3 | `FD05_tv21_t12_joint_mem025_logits2_8ep` | Long-context accuracy ceiling |

Interpretation order:

1. FD46 vs FD47 identifies whether depth reduction alone is safe.
2. FD47 vs FD48 identifies recoverable optimization drift.
3. FD48/FD04 vs FD49 separates Perceiver/interface loss from two-layer loss.
4. FD49 vs FD50 tests whether the TinyViT–temporal interface must adapt.
5. FD01/FD03/FD05 tests whether additional temporal exposure produces a
   monotonic gain before attributing failure to architecture.

## Multiplex speed and quality

The runtime path already demonstrated roughly 49 FPS at eight objects, but
shared K/V retained only about 81% full J&F. These nine runs target the missing
object-specific temporal information while retaining one shared attention
pass for N≥4.

| Node | Variant | Causal role |
|---:|---|---|
| 4 | `FD11_sharedkv_all_t8_r16_8ep` | Maximum identity diversity |
| 4 | `FD12_sharedkv_dense4_t8_r16_8ep` | Diversity/multiplex balance |
| 4 | `FD13_sharedkv_dense8_t8_r16_8ep` | Maximum dense-object supervision |
| 5 | `FD18_sharedkv_r8_ptr8_8ep` | Rank-eight residual control |
| 5 | `FD19_sharedkv_r16_ptr8_8ep` | Double object-specific bandwidth |
| 5 | `FD20_sharedkv_r32_ptr8_8ep` | Test whether quality is capacity-limited |
| 6 | `FD21_sharedkv_r16_mem025_logits2_8ep` | Weaker memory KD allows plasticity |
| 6 | `FD25_sharedkv_r16_mem100_logits4_8ep` | Stronger mask behavior preservation |
| 6 | `FD30_sharedkv_r16_ptr8_recency050_obj025_8ep` | Recency-weighted identity state |

Promotion requires validation J&F retention ≥95%, learned-path mask IoU
≥0.95, and N=8 faster than the persistent bucket-4 reference. N=1 is reported
but not blocking because deployment can route N<4 through the legacy path.

## Entry point

```bash
scripts/company/68_run_sam2_priority_18.sh describe
scripts/company/68_run_sam2_priority_18.sh audit
scripts/company/68_run_sam2_priority_18.sh node 1
scripts/company/68_run_sam2_priority_18.sh summarize
```

Primary implementation paths:

- Priority queues: `scripts/company/68_run_sam2_priority_18.sh`
- Original experiment definitions: `tools/experiments/sam2_full_data_50.py`
- Full-data pipeline: `scripts/company/67_run_sam2_full_data_50.sh`
- Training and evaluation: `scripts/company/49_run_edgetam_memory_ablation.sh`
