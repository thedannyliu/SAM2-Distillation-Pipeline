# Weekend 72-Hour Experiment Suite v1

## Goal

Use four independent 4xH100 company nodes for a weekend-scale experiment
block without repeating settings that the 2026-07-24 results already made
unlikely to help. The suite asks four separate questions:

1. Can behavior alignment repair the strict TinyViT-to-official-EdgeTAM
   transplant?
2. Can a same-interface M0 teacher make four-to-two-layer memory compression
   learnable?
3. How much additional full-SA-V fine-tuning headroom remains for
   TinyViT-5M/11M/21M?
4. Which subsystem limits the partially recovered RepViT model?

All candidates use W&B online, frozen BatchNorm, deterministic seeds, and
full SA-V train data. Every formal candidate executes train -> full SA-V val
-> full SA-V test. Candidate selection uses full-val J&F only; test remains
descriptive even though it is recorded for every run at the user's request.

## Runtime basis

The prior company runs took approximately 3-4 hours per T4 full-SA-V epoch.
The table converts T2 to 0.5 and T8 to 2.0 T4-equivalent epochs. This is an
observation-based allocation estimate, not a runtime guarantee.

| Node lane | T4-equivalent epochs | Formal runs | Full val/test pairs | Estimated training only |
| --- | ---: | ---: | ---: | ---: |
| `edge_official` | 34 | 7 | 7 | 102-136 h |
| `edge_compression` | 36 | 8 | 8 | 108-144 h |
| `tinyvit` | 30 | 9 | 9 | 90-120 h |
| `repvit` | 24 | 6 | 6 | 72-96 h |

Evaluation time is additional. The queue therefore intentionally exceeds
72 hours per node. Completed stages are skipped, so the same foreground
command safely resumes after interruption.

## Node 1: official EdgeTAM behavior transfer

The strict E1 transplant had 0.0200 val mIoU and 2.1 J&F. This lane first
aligns the TinyViT image representation, then holds architecture,
initialization, data, prompts, and optimization fixed while adding temporal
behavior targets.

| Run | Start | Trainable modules | T/epochs | Distillation terms |
| --- | --- | --- | --- | --- |
| W1 | E1 | image encoder | T2/2 | image + mask logits |
| W2a | W1 | image + memory + Perceiver | T4/5 | image + mask logits |
| W2b | W1 | image + memory + Perceiver | T4/5 | image + memory + mask logits |
| W2c | W1 | image + memory + Perceiver | T4/5 | image + memory + mask logits + object pointer |
| W3a | W2a | memory + Perceiver | T8/3 | mask logits |
| W3b | W2b | memory + Perceiver | T8/3 | memory + mask logits |
| W3c | W2c | memory + Perceiver | T8/3 | memory + mask logits + object pointer |

Interpretation:

- If W1 restores image mIoU but W2 remains low-J&F, the remaining failure is
  temporal rather than an encoder interface.
- W2a -> W2b isolates final memory-feature supervision.
- W2b -> W2c isolates object-pointer direction supervision.
- W3 determines whether a branch that works at T4 continues to benefit from
  longer temporal context.

## Node 2: same-interface M0 compression

This is the primary EdgeTAM production path. It preserves the healthy A02
image/prompt/mask interface, initializes the complete official temporal
stack coherently, and uses the functional TinyViT-21M M0 four-layer model
as teacher. It directly tests the hypothesis that the earlier official
teacher crossed an avoidable representation boundary.

| Fork | T4 stage | T8 continuation | Teacher terms |
| --- | --- | --- | --- |
| task control | K1a, 5 epochs | K2a, 2 epochs | none |
| logits | K1b, 5 epochs | K2b, 2 epochs | propagated mask logits |
| memory + logits | K1c, 5 epochs | K2c, 2 epochs | final memory feature + logits |
| full behavior | K1d, 5 epochs | K2d, 2 epochs | memory + logits + object pointer |

All four forks share A02, official temporal initialization, task loss, data,
and learning-rate schedule. K2 starts only from its matched K1 checkpoint,
so the T8 comparison does not mix branches.

Key decision thresholds:

- Below 40 val J&F after five T4 epochs: the loss component is not identifying
  a useful compressed temporal solution.
- At least 60 val J&F: the compressed path is viable for controlled latency
  benchmarking.
- Within 3 points of M0's 71.5 val J&F: promote to the production Pareto set.

## Node 3: TinyViT capacity ceiling

Each capacity starts from its current full-val winner:

- 5M: `tv5_F2_joint_low_1ep`, 65.8 val J&F.
- 11M: `tinyvit_max_jf_v1/tv11`, 68.5 val J&F.
- 21M: `tinyvit_max_jf_v1/tv21`, 72.4 val J&F.

For each size, two independent three-epoch T4 forks compare:

1. frozen image encoder with decoder/memory adaptation;
2. frozen-BN, low-LR joint encoder/decoder/memory adaptation.

The higher full-val-J&F fork then receives two T8 decoder/memory epochs.
This produces three formal candidates and ten T4-equivalent epochs per
capacity. The experiment measures remaining capacity headroom without
confounding it with a different Stage-1 checkpoint or trainable BatchNorm.

## Node 4: RepViT residual-gap localization

All four forks start from P3, which recovered test J&F from 37.5 to 60.1:

| Run | Trainable modules | T/epochs | Question |
| --- | --- | --- | --- |
| W1 | encoder | T2/5 | Is image representation still limiting? |
| W2 | mask decoder | T2/5 | Is prompted-mask adaptation limiting? |
| W3 | mask decoder + memory | T4/5 | Is temporal adaptation limiting? |
| W4 | encoder + decoder + memory | T4/5 | Is coupled adaptation required? |

Full-val J&F selects one fork. W5 gives that fork three T8 decoder/memory
epochs, and W6 finishes with three low-LR joint T4 epochs. The comparison is
causal because W1-W4 share the same P3 start and keep BatchNorm frozen.

## Reproducibility, retention, and reporting

- Entrypoint:
  `scripts/company/57_run_weekend_72h_lane.sh`
- Suite root:
  `/group-volume/danny-dataset/sam2_distill/runs/weekend_72h_v1`
- Foreground logs:
  `/user-volume/weekend_72h_logs/<lane>`
- W&B projects:
  `weekend-72h-<lane>-v1`
- Seeds are recorded per run. Paired backbone forks use the same seed so the
  trainable-module comparison is not confounded by a different sampling
  stream; continuation stages use a new recorded seed.
- Each stage keeps only physical `last.pt` and `best.pt`; compatibility names
  are symlinks. EdgeTAM `best.pt` points to `last.pt`.
- Evaluation predictions are deleted after scoring. Metrics, W&B metadata,
  resolved configs, initialization provenance, and summary CSVs are retained.
- The expected incremental checkpoint/log footprint is below 80 GB, excluding
  shared datasets and existing starting checkpoints.
- `tools/experiments/summarize_all_experiments.py` registers all 30 candidates,
  including their target epoch, so interrupted runs appear as incomplete
  rather than disappearing from the all-experiment report.

Use the `describe` action for a no-GPU expansion of every variant:

```bash
scripts/company/57_run_weekend_72h_lane.sh edge_official describe
scripts/company/57_run_weekend_72h_lane.sh edge_compression describe
scripts/company/57_run_weekend_72h_lane.sh tinyvit describe
scripts/company/57_run_weekend_72h_lane.sh repvit describe
```
