# EventSAM2 COAST action-headroom screen v2

## Research question

Can a causal fast action produce useful current masks while skipping all four
expensive SAM2 operations on the selected frame: image encoder, memory
attention, SAM mask decoder, and memory encoding/write?

This first company experiment tests the action before training a gate. The
fast action is deliberately simple:

| Action | Computation | SAM2 memory mutation |
|---|---|---|
| `COAST` | 256 px DIS optical-flow sentinel plus recurrent transient-mask warp | none |
| `REFRESH` | frozen SAM2.1-L full observation trajectory | full baseline |

The REFRESH mask used to reset COAST comes from a complete SAM2 trajectory.
It therefore has memories that a deployed selective policy would not have.
All results must be named **optimistic oracle-reset action headroom**, not a
state-consistent policy or deployable speedup.

## Data contract

The frozen seed is `eventsam2_sav_v1`. The 50,337 train videos with a readable
matching manual annotation are hash-ranked and assigned without video overlap:

| Role | Videos | Permitted use |
|---|---:|---|
| `route_train` | 40,270 | future COAST correction-head training |
| `gate_train` | 5,034 | future counterfactual risk-head training and DAgger |
| `selection` | 2,517 | architecture and horizon selection |
| `calibration` | 2,516 | final clip-level risk calibration only |
| `sav_val` | 155 | full validation after a selected stage |
| `sav_test` | 150 | final frozen system only; untouched in v1 |

The initial screen uses a deterministic 32-video subset of `selection`. Raw
24 FPS frames are decoded from the MP4s for the screen. No 24 FPS copy of the
full train release is created. Full `sav_val` uses its prepared
`JPEGImages_24fps` and `Annotations_6fps` layout.

The screen keeps only objects with a non-empty mask on frame 0. This preserves
one standard multi-object SAM2 propagation and a meaningful frame-shared full
latency. Objects first annotated later represent an external prompt event; a
deployed controller must force REFRESH on that event, and they require a
separate delayed-prompt ablation rather than silently switching the baseline
to repeated per-object inference. Preparation records how many objects this
contract excludes in `prepare_summary.json`.

## Measurements

The baseline runner uses one foreground `torchrun` process with four ranks.
Each rank owns disjoint videos. It records both total wall time and synchronized
per-frame SAM2 propagation time. COAST speed estimates use the latter, not the
four-GPU throughput number. COAST timing profiles the first four selected
videos serially with one OpenCV thread; quality analysis then uses 16 host
workers. This avoids reporting thread-contention time as single-stream action
latency.

Every fixed refresh interval evaluates all phases. This prevents interval 4
from appearing artificially safe merely because SA-V GT is sampled every four
24 FPS frames. Report:

- mean and worst-phase J&F drop for intervals 1, 2, 4, 8, and 16;
- sentinel, multi-object transition, and total COAST compute latency;
- safe-anchor fraction for 4, 8, and 16 consecutive COAST frames;
- video-macro and 10th-percentile safe-anchor coverage;
- per-object/per-frame metrics and per-anchor maximum future regret.

At an annotated full-prefix anchor `t`, the safe-horizon branch recursively
COASTs for `H` frames. It includes every annotated frame and every object in
the branch:

\[
R_{t,H}=\max_{i,h\le H}
100\,[J\&F^{full}_{t+h,i}-J\&F^{coast}_{t+h,i}]_+.
\]

The branch is safe when `R <= 1.0` point. New objects that the transient state
cannot represent count as failures rather than being dropped.

Because GT is available at 6 FPS, v1 safe coverage is the fraction of
annotation-aligned anchors whose whole branch is safe; it is not yet the
frame-level gate coverage claimed by the final method. The learned E2 oracle
must repeat coverage at every 24 FPS decision using dense teacher targets plus
the available GT checkpoints.

## Promotion gates

| Question | Gate |
|---|---:|
| Is the fast action actually cheap? | COAST compute `<= 20%` of full propagation |
| Is aggregate temporal headroom meaningful? | safe-anchor coverage `>= 60%` |
| Does the opportunity survive difficult videos? | video 10th percentile `>= 40%` |
| Is the systems ceiling worth pursuing? | ideal periodic estimate `>= 1.7x` |

Passing these gates promotes the project to a learned correction head trained
on `route_train`, followed by same-prefix/commit-H state-consistent labels.
Failure of optical flow alone does not reject a learned COAST action; it says
how much improvement the correction head must provide. A learned route that
still fails the same gates is the actual no-go result.

## Company workflow

All stages run in the foreground. Large artifacts and W&B files live below
`/group-volume/danny-dataset`; terminal logs live below `/user-volume/log`.
The `screen` action first runs the full train preparation audit as a hard input
gate, then creates the frozen split.

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only
mkdir -p /user-volume/log/eventsam2_coast_v2

scripts/company/73_run_eventsam2_coast_screen.sh screen 2>&1 | \
  tee /user-volume/log/eventsam2_coast_v2/selection_screen32.log
echo "Selection screen status: ${PIPESTATUS[0]}"
```

Inspect:

```bash
python -m json.tool \
  /group-volume/danny-dataset/sam2_distill/runs/eventsam2_coast_v2/selection_screen32/coast_screen/summary.json
```

Only after the screen is complete should the same frozen action be evaluated
on full validation:

```bash
scripts/company/73_run_eventsam2_coast_screen.sh val 2>&1 | \
  tee /user-volume/log/eventsam2_coast_v2/sav_val.log
echo "Full sav_val status: ${PIPESTATUS[0]}"
```

The script has no `sav_test` action. Test access is reserved for the final
learned and clip-calibrated policy.

## Results

| Split | Full J&F | Full model ms | COAST ms | H8 safe | H16 safe | p10 safe | Best fixed drop/speed | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| selection-32 | pending | pending | pending | pending | pending | pending | pending | pending |
| full `sav_val` | pending | pending | pending | pending | pending | pending | pending | pending |

Primary machine-readable outputs are `summary.json`,
`fixed_policy_object_metrics.parquet`, `fixed_policy_system_metrics.parquet`,
and `safe_horizon_oracle.parquet` under each `coast_screen` directory.
