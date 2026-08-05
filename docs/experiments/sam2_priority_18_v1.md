# SAM2 priority-18 full-data suite

## Decision

The original 50-run matrix was reduced to 18 prioritized experiments for six
4×H100 nodes. As of 2026-08-05, 11 of the 18 experiments are complete: five of
nine memory candidates and six of nine multiplex candidates. The remaining
seven have no checkpoint or evaluation artifact and remain pending. The
selected checkpoints and W&B runs stay under `sam2_full_data_50_v1`, so the
other registered variants remain runnable later.

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

## Results snapshot: 2026-08-05

These are fixed 64-video SA-V validation screens, not full-val/full-test
results. The runtime reference is the persistent bucket-4 model at 22.07 N=8
FPS. The quality reference is MX5 on the identical cohort.

### Memory and EdgeTAM-style candidates

| Variant | State | Mini J&F | Retention | N=1 FPS | N=8 FPS | N=8 gain | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| `FD01_tv21_t4_decmem_5ep` | complete | 70.0 | 0.970 | 57.24 | 23.96 | +8.6% | Best completed quality; clears the 95% screen gate |
| `FD46_mem4_t8_decmem_8ep` | complete | 69.5 | 0.963 | 56.84 | 24.09 | +9.1% | Four-layer T8 also clears the quality gate |
| `FD03_tv21_t8_logits1_8ep` | complete | 68.8 | 0.953 | 59.23 | 24.26 | +9.9% | Fastest of the quality-passing memory candidates |
| `FD47_mem2_t8_decmem_8ep` | complete | 58.9 | 0.816 | 66.98 | 34.54 | +56.5% | Two layers buy speed but lose too much temporal quality |
| `FD49_edgetam2_temporal_logits2_8ep` | complete | 46.8 | 0.648 | 22.14 | 14.99 | -32.1% | EdgeTAM temporal-only adaptation is worse in both quality and N=8 speed |
| `FD48`, `FD04`, `FD50`, `FD05` | pending | - | - | - | - | - | Required recovery and longer-context controls are not available yet |

The `Promote=0` field in the shared screen report must not be used to reject
FD01, FD03, or FD46. The report requires a learned-path mask-IoU measurement,
but standard-memory candidates do not have that verification field. Their
relevant first gate is J&F retention, which all three pass. They now warrant
full validation and test before choosing among the small 8--10% N=8 gains.

Depth reduction is a genuine Pareto trade-off rather than a free optimization:
FD47 improves N=8 from roughly 24 to 34.54 FPS, but loses about ten J&F points
relative to the four-layer controls. FD49 shows that inserting the current
EdgeTAM two-layer Perceiver path does not solve the problem; it is dominated by
FD47 on both axes. FD48 and FD50 are still important because they test whether
joint image/temporal adaptation can recover either failure.

### Shared-K/V multiplex candidates

| Variant | State | Mini J&F | Retention | Mask IoU | N=1 FPS | N=8 FPS | N=8 gain |
|---|---|---:|---:|---:|---:|---:|---:|
| `FD11_sharedkv_all_t8_r16_8ep` | complete | 58.3 | 0.807 | 1.000 | 58.68 | 46.84 | +112.2% |
| `FD12_sharedkv_dense4_t8_r16_8ep` | complete | 58.2 | 0.806 | 1.000 | 59.60 | 47.03 | +113.1% |
| `FD18_sharedkv_r8_ptr8_8ep` | complete | 59.0 | 0.817 | 1.000 | 56.44 | 46.35 | +110.0% |
| `FD19_sharedkv_r16_ptr8_8ep` | complete | 58.6 | 0.812 | 1.000 | 58.45 | 46.80 | +112.0% |
| `FD21_sharedkv_r16_mem025_logits2_8ep` | complete | 59.1 | 0.819 | 1.000 | 58.84 | 47.22 | +113.9% |
| `FD25_sharedkv_r16_mem100_logits4_8ep` | complete | 58.7 | 0.813 | 1.000 | 56.66 | 46.56 | +110.9% |
| `FD13`, `FD20`, `FD30` | pending | - | - | - | - | - | Dense-8, rank-32, and recency controls are not available yet |

FD21 is the current completed multiplex Pareto point: it has both the highest
J&F retention and highest N=8 throughput in this subset. It still retains only
81.9% of reference J&F, so none of the six completed multiplex runs is a
promotion candidate. All six have exact learned-path mask agreement but nearly
the same 58.2--59.1 J&F and 46.35--47.22 FPS band. Full-SA-V training, dense-4
sampling, residual rank 8/16, weaker memory KD, and stronger logit KD therefore
did not move the quality ceiling materially. This is evidence that the current
shared-K/V representation removes object-specific temporal information; it is
not primarily a data-volume or optimizer-budget problem.

N=1 retention is only 0.79--0.84 for these candidates. Production evaluation
should continue to route N<4 through the legacy path rather than paying the
multiplex fixed cost at low occupancy.

### Current decision

1. Promote FD01, FD03, and FD46 to full val/test evaluation, but do not select a
   deployment winner from the 64-video screen alone.
2. Keep FD47 as the two-layer speed upper bound, not as a quality candidate.
3. Do not spend another sweep on the same post-attention shared-K/V residual;
   the next multiplex model must preserve private object information inside
   the temporal reader, as in the SAM3.1-aligned bucket experiment.
4. Complete FD48 and FD50 only if diagnosing recoverability of the compressed
   and EdgeTAM interfaces remains useful. FD13/FD20/FD30 have lower priority
   because the completed multiplex results already occupy a narrow plateau.

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
