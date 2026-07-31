# SAM2 multiplex overnight v4

## Question

Can a cheap object-specific temporal residual recover at least 95% of the
MX5 tracking quality while retaining most of the fully shared-K/V N=8
speedup?

The selected deployment reference remains the persistent runtime bucket:
about 22.1 FPS at N=8 with val/test J&F 72.4/74.7. Fully shared K/V reaches
about 49 FPS at N=8 but only 63.6/60.2 J&F. MX5 preserves 72.3/74.6 J&F but
reaches only about 24.4 FPS at N=8. This screen targets the gap between
those endpoints.

## Budget and protocol

- Resources: eight independent nodes, each with 4×H100.
- Work per node: two experiments, sequentially, using four-GPU DDP.
- Common initializer: completed MX5 checkpoint.
- Training: T8 dense-object clips, three epochs, seed 250107256.
- Quality screen: the same deterministic 32-video SA-V val cohort for MX5
  and every candidate.
- Latency screen: N=1/2/4/8, one repetition, one dense-eight video, one
  isolated H100.
- Full five-epoch SA-V val/test is deferred to at most four Pareto
  candidates.

Three epochs are intentional. Sixteen five-epoch runs plus complete val and
test evaluation would spend most of the overnight budget repeating
evaluation. The fixed-cohort screen measures every hypothesis under the same
data and promotes only candidates that preserve tracking behavior.

## Experiment matrix

Each node changes one factor. Comparisons across a row are paired; MX15 and
MX18 provide the shared controls used by later rows.

| Node | Variant A | Variant B | Question |
|---:|---|---|---|
| 1 | MX13: spatial rank 2 | MX14: spatial rank 4 | How little object-specific spatial capacity is sufficient? |
| 2 | MX15: spatial rank 8 | MX16: spatial rank 16 | Does added rank still buy quality, or only latency? |
| 3 | MX17: pointer rank 4 | MX18: pointer rank 8 | How much explicit object-pointer identity is required? |
| 4 | MX19: latest-frame pooling | MX20: recency pooling, decay 0.50 | Is recent identity more useful than the default temporal mean? |
| 5 | MX21: recency decay 0.25 | MX22: recency decay 0.75 | Should the residual be short-memory or long-memory? |
| 6 | MX23: two shared buckets at N=8 (slot 4) | MX24: slot 6 | What is the quality/speed frontier for objects per shared K/V call? |
| 7 | MX25: object-pointer KD 0.25 | MX26: object-pointer KD 1.0 | Can direct identity distillation recover VOS quality? |
| 8 | MX27: share from N=2 | MX28: share from N=3 | Where should low-count requests switch to the shared path? |

All variants use memory KD 1 and mask-logit KD 2. Unless a row says
otherwise, they use slot 8, spatial rank 8, temporal mean pooling, pointer
rank 8, and shared-path minimum N=4.

## Metrics and promotion

The result table answers one question: which variants lie on the measured
speed–quality frontier?

- Primary quality: fixed-cohort J&F retention relative to MX5.
- Correctness: minimum bucket-versus-legacy mask IoU.
- Primary speed: N=8 propagation FPS and milliseconds per frame.
- Diagnostic speed: N=1/2/4 FPS and N=1 retention.
- Promotion gate: J&F retention ≥95%, mask IoU ≥0.95, and N=8 FPS above
  the persistent runtime bucket.

N=1 is reported but does not block the screen because deployment can route
N<4 through the legacy path. A promoted model must still pass the stricter
full benchmark before deployment.

The summary writes:

- `comparison/screen_results.md`
- `comparison/screen_results.csv`
- `comparison/screen_results.json`
- `comparison/promotion_candidates.txt`

## Implementation paths

- Shared K/V and residual:
  `sam2_distill/models/sam2_object_slots.py`
- Training configuration:
  `tools/train/run_sam2_task_training.py`
- Fixed-cohort quality gate:
  `tools/experiments/check_sav_memory_gate.py`
- N=1/2/4/8 benchmark:
  `tools/benchmark/benchmark_sam2_multiobject_scaling.py`
- Screen summary:
  `tools/benchmark/summarize_sam2_multiplex_screen.py`
- Eight-node runner:
  `scripts/company/64_run_sam2_multiplex_overnight_v4.sh`

## Company commands

Run one block in each of eight company terminals. Every command remains in
the foreground and records live output.

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only
mkdir -p /user-volume/sam2_multiplex_v4_logs
scripts/company/64_run_sam2_multiplex_overnight_v4.sh queue 1 2>&1 | \
tee /user-volume/sam2_multiplex_v4_logs/node1_$(date +%Y%m%d_%H%M%S).log
echo "Node 1 status: ${PIPESTATUS[0]}"
```

Use the same block on the other terminals, replacing both occurrences of
`1` with `2`, `3`, `4`, `5`, `6`, `7`, or `8`.

After all queues return:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
scripts/company/64_run_sam2_multiplex_overnight_v4.sh summarize 2>&1 | \
tee /user-volume/sam2_multiplex_v4_logs/summary_$(date +%Y%m%d_%H%M%S).log
echo "Summary status: ${PIPESTATUS[0]}"
cat /danny-dataset/sam2_distill/runs/sam2_multiplex_overnight_v4/comparison/screen_results.md
```

If the data root is mounted at `/group-volume/danny-dataset` instead, use
the `Run root` printed by the command when opening the report.

Promote only variants listed in `promotion_candidates.txt`:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
VARIANT="replace_with_one_candidate"
scripts/company/64_run_sam2_multiplex_overnight_v4.sh promote "$VARIANT" 2>&1 | \
tee "/user-volume/sam2_multiplex_v4_logs/promote_${VARIANT}_$(date +%Y%m%d_%H%M%S).log"
echo "Promotion status: ${PIPESTATUS[0]}"
```

Promotion resumes the same checkpoint and W&B run from epoch three to epoch
five, then runs full SA-V val, full SA-V test, and a three-repetition latency
benchmark.

## Deferred SA-V scale study

The extra 1–2 TB is intentionally not used in this architecture screen.
The current dense8 selector repeats eligible video IDs to reach 50,337
samples, so raising that number alone does not prove that more unique data
helps. After selecting the v4 architecture, run a matched data study with
the same optimizer updates:

1. 100k samples from the existing dense8 cohort.
2. 100k samples from a broader dense4 cohort drawn from additional SA-V
   videos.

This separates extra stochastic windows from genuinely greater video and
object diversity. Raw SA-V shards, manifests, caches, and intermediate runs
belong under `/danny-dataset`; only final selected weights belong under
`/group-volume`.
