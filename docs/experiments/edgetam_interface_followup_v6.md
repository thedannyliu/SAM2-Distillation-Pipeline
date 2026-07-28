# EdgeTAM Interface Follow-up v6

## Research question

Can compressed EdgeTAM temporal behavior be recovered by supervising the
encoder-to-memory interface, and does the answer depend on whether the teacher
shares the student's TinyViT image and mask interface?

Q0 shows that the local trainer preserves a functional official EdgeTAM model.
E1 shows that a strict TinyViT encoder transplant breaks even the image path;
Q1 restores the mature TinyViT image/mask path but still cannot generalize
temporal propagation. The next experiments therefore compare teacher
interface rather than another broad learning-rate or epoch sweep.

## Deferred official-interface curriculum

| Run | Start | T/epochs | Trainable path | Teacher targets |
|---|---|---:|---|---|
| `W1_official_image_align_2ep` | strict E1 transplant | T2/2 | image encoder | image features + propagated masks |
| `W2a_official_logits_5ep` | W1 | T4/5 | image + temporal | image features + propagated masks |
| `W2b_official_memlogits_5ep` | W1 | T4/5 | image + temporal | image + memory features + propagated masks |
| `W3a_official_logits_t8_3ep` | W2a | T8/3 | temporal | propagated masks |
| `W3b_official_memlogits_t8_3ep` | W2b | T8/3 | temporal | memory features + propagated masks |

This remains the correct cross-interface experiment, but it is deferred from
the present two-node block. The second available node is assigned to the
independent TinyViT-5M SAM2.1-L pseudo-mask question documented in
`tinyvit5_pseudolabel_v1.md`.

## Active EdgeTAM node: same-interface compression

| Run | Start | T/epochs | Teacher | Targets |
|---|---|---:|---|---|
| `K1b_m0_logits_5ep` | A02 + official temporal init | T4/5 | functional TinyViT M0 | propagated masks |
| `K1c_m0_memlogits_5ep` | same controlled start | T4/5 | functional TinyViT M0 | memory features + propagated masks |
| `K2b_m0_logits_t8_2ep` | K1b | T8/2 | functional TinyViT M0 | propagated masks |
| `K2c_m0_memlogits_t8_2ep` | K1c | T8/2 | functional TinyViT M0 | memory features + propagated masks |

The completed task-only controls are K1a at 38.7 val J&F and K2a at 42.1.
K1b/c directly test whether behavior targets close that gap without crossing
the RepViT/TinyViT interface boundary.

## Controls and decisions

- All runs use one independent 4xH100 node, frozen BatchNorm, online W&B,
  deterministic initialization, and only physical `last.pt`/`best.pt`.
- Every candidate runs train -> full SA-V val -> full SA-V test. Selection uses
  val J&F only; test is descriptive.
- The core compression lane is 18 T4-equivalent epochs and four full
  evaluations, exceeding a 12-hour allocation under observed runtime.
- Object-pointer variants are not in this block because `obj_ptr` is not
  reliably emitted on the active box-prompt teacher path.
- Below 40 val J&F: reject the loss target.
- At least 60 val J&F: the compressed path is viable for latency comparison.
- Within 3 points of M0's 71.5 val J&F: promote to the production Pareto set.

## Execution

The existing weekend lane driver exposes the focused subset through
`EDGE_FOLLOWUP_MODE=core`:

```bash
EDGE_FOLLOWUP_MODE=core scripts/company/57_run_weekend_72h_lane.sh edge_compression describe
```

In `core` mode, `K2b` and `K2c` are conditional continuations. Their matching
`K1` parent must reach val `J&F >= 57.0`, `mIoU >= 0.8355`, and `AP >= 0.7117`;
otherwise the T8 child is skipped without failing the lane.

Outputs remain under:

- `runs/weekend_72h_v1/edge_compression`
- W&B project `weekend-72h-edge_compression-v1`

## 2026-07-28 18:30 UTC Status

The same-interface behavior targets have not started:

| Run | Status | Progress | val J&F | test J&F |
|---|---|---:|---:|---:|
| `K1b_m0_logits_5ep` | not started | 0% | - | - |
| `K1c_m0_memlogits_5ep` | not started | 0% | - | - |
| `K2b_m0_logits_t8_2ep` | not started | 0% | - | - |
| `K2c_m0_memlogits_t8_2ep` | not started | 0% | - | - |

The deferred official-interface rows W1, W2a/b/c, and W3a/b/c are also all
`not_started`. No new EdgeTAM conclusion should be drawn from this snapshot;
the latest causal results remain the task-only K1a/K2a controls and Q0/Q1.

### 2026-07-28 18:40 UTC launch failure

K1b and K1c stopped before their first optimizer step. Each stage had already
written a local W&B run ID, but the corresponding remote run had never been
initialized. The retry used `resume="must"`, which requires the remote run to
exist. K2b and K2c then failed only because their K1 checkpoints were absent.
These are orchestration failures and contain no research result.

The task trainer now uses `resume="allow"` when a run ID is available. It
continues the same ID when the remote run exists and initializes that ID when
only the local record exists. Rerunning the core lane will retry K1b/K1c in
their existing directories and then unlock K2b/K2c.
