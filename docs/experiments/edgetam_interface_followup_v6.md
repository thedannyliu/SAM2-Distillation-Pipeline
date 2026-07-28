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

Outputs remain under:

- `runs/weekend_72h_v1/edge_compression`
- W&B project `weekend-72h-edge_compression-v1`
