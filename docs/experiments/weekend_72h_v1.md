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

## Results through 2026-07-27

The 19:09 UTC all-experiment snapshot contains results for 17 of the 30
registered weekend candidates:

| Lane | Complete | Not started | Current conclusion |
| --- | ---: | ---: | --- |
| `edge_official` | 0/7 | 7 | no new evidence beyond E1 |
| `edge_compression` | 2/8 | 6 | task-only control improves but remains non-viable |
| `tinyvit` | 9/9 | 0 | complete; additional headroom is at most 0.2 val J&F |
| `repvit` | 6/6 | 0 | complete; decoder/memory fork is the val winner |

### TinyViT capacity results

| Size | Run | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5M | W1 decoder/memory | 0.7997 | 0.6425 | **66.0** | 0.8025 | 0.6552 | 67.6 |
| 5M | W2 joint | **0.8000** | **0.6436** | 65.8 | **0.8028** | **0.6559** | 67.8 |
| 5M | W3 selected T8 | 0.7996 | 0.6429 | 65.4 | 0.8027 | 0.6539 | **67.9** |
| 11M | W1 decoder/memory | 0.8141 | 0.6717 | **68.6** | 0.8161 | 0.6791 | 70.5 |
| 11M | W2 joint | **0.8142** | 0.6717 | 68.0 | **0.8163** | 0.6802 | **70.6** |
| 11M | W3 selected T8 | 0.8140 | **0.6718** | 68.2 | 0.8160 | **0.6803** | 69.7 |
| 21M | W1 decoder/memory | 0.8372 | 0.7130 | 71.7 | 0.8357 | **0.7166** | 74.5 |
| 21M | W2 joint | 0.8373 | 0.7131 | 71.5 | 0.8354 | 0.7165 | 74.4 |
| 21M | W3 selected T8 | **0.8374** | **0.7132** | **72.0** | **0.8357** | **0.7166** | **74.8** |

Validation-based conclusions:

- 5M improves from 65.8 to 66.0 val J&F. W1 is selected even though W3 has
  the higher descriptive test result.
- 11M improves from 68.5 to 68.6. W1 is selected; W2's 70.6 test J&F cannot
  override its lower 68.0 val result.
- None of the 21M candidates beats the existing 72.4-val model. W3 reaches
  74.8 test J&F, but its 72.0 val result does not replace the existing
  checkpoint.
- The ordered gains over the pre-weekend winners are only +0.2/+0.1/0.0 for
  5M/11M/21M. More epochs and T8 context therefore confirm saturation rather
  than revealing a hidden large fine-tuning gain.

### RepViT localization results

| Run | Scope | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | encoder | 0.7610 | 0.5777 | 60.9 | 0.7590 | 0.5774 | 60.2 |
| W2 | mask decoder | 0.7596 | 0.5738 | 60.4 | 0.7578 | 0.5755 | 59.2 |
| W3 | decoder + memory | 0.7596 | 0.5733 | **61.4** | 0.7581 | 0.5762 | 60.0 |
| W4 | joint | 0.7613 | 0.5780 | 60.9 | **0.7605** | **0.5789** | 60.4 |
| W5 | selected W3, T8 | 0.7598 | 0.5740 | 60.4 | 0.7580 | 0.5766 | 60.9 |
| W6 | W5, joint low LR | **0.7616** | **0.5781** | 60.9 | 0.7597 | 0.5775 | **61.1** |

W3 is the formal winner because it reaches 61.4 val J&F, +1.1 over P3.
The subsystem comparison points to temporal decoder/memory adaptation rather
than image-only or mask-decoder-only adaptation. W5 then loses 1.0 val J&F;
W6 recovers only to 60.9. Its 61.1 test result is descriptive and does not
replace W3. RepViT's val-selected result remains 4.6 points behind the new
TinyViT-5M winner.

### Same-interface EdgeTAM compression results

| Run | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K1a task-only T4/5ep | 0.8404 | **0.7165** | 38.7 | 0.8389 | 0.7190 | 39.0 |
| K2a task-only T8/2ep | **0.8404** | 0.7160 | **42.1** | **0.8389** | **0.7190** | **40.0** |

K2a adds 3.4 val and 1.0 test J&F over K1a while preserving image metrics,
so longer temporal exposure helps this coherent task-only branch. It still
trails the M0 four-layer reference by 29.4 val and 34.3 test J&F, and trails
the simple M1 two-layer truncation by 11.2/16.1. It therefore fails the
pre-registered 60-J&F viability threshold.

K1a is 23.1 val J&F above the two-epoch scratch S0 control, but this is not
a clean initialization ablation: K1a also has five rather than two T4
epochs. The only controlled new conclusion is K1a -> K2a. K1b-K1d and
K2b-K2d remain not started, so same-interface mask-logit, memory-feature,
and object-pointer behavior supervision are still the highest-value missing
experiments. All W1-W3 official-transfer runs also remain not started.

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
