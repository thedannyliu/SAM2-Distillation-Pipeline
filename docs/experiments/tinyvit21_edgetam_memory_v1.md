# TinyViT-21M EdgeTAM memory feasibility v1

## Question

Can the production TinyViT-21M image encoder support a coherent EdgeTAM temporal
stack after full-SA-V training, while preserving useful video J&F?

This is a feasibility experiment, not an exact reproduction of the EdgeTAM
training recipe. The official recipe mixes additional video and image datasets
and uses a much larger global batch. This run deliberately isolates the value of
full SA-V and EdgeTAM memory components on one 4xH100 node.

## Fixed inputs

| Item | Value |
|---|---|
| Train data | Full prepared SA-V train manifest |
| Manifest | `/group-volume/danny-dataset/sam2_distill/manifests/sav_train_6fps_full.parquet` |
| Validation | Full SA-V val |
| Test | Full SA-V test, evaluated only after validation |
| Image encoder | TinyViT-21M from the best TV21 task checkpoint |
| Temporal initialization | Official EdgeTAM attention, memory encoder, spatial Perceiver, pointer projections, and temporal embeddings |
| Hardware | One node, 4xH100 |
| Tracking | W&B `tinyvit21-edgetam-memory-v1` plus the existing TensorBoard/checkpoint directory |

The implementation reuses
[`49_run_edgetam_memory_ablation.sh`](../../scripts/company/49_run_edgetam_memory_ablation.sh)
for model construction, checkpoint resume, and evaluation. The foreground
orchestrator is
[`69_run_tinyvit21_edgetam_memory_v1.sh`](../../scripts/company/69_run_tinyvit21_edgetam_memory_v1.sh).
Official temporal tensor selection and coherent checkpoint transfer live in
[`task_finetune.py`](../../sam2_distill/models/task_finetune.py). Trainable-module
selection lives in
[`train_model.py`](../../sam2_distill/edgetam/train_model.py).

## Training curriculum

| Stage | T | Epochs | Trainable modules | Initialization | Main losses | Hypothesis |
|---|---:|---:|---|---|---|---|
| EM1 | 4 | 2 | EdgeTAM attention, memory encoder, Perceiver, pointer/temporal parameters | best TV21 task model + official EdgeTAM temporal stack | task 1, memory 0.5, logits 2 | Adapt the coherent official temporal interface without moving TinyViT |
| EM2 | 8 | 5 | TinyViT encoder + the full EdgeTAM temporal stack | EM1 | task 1, image 1, memory 0.5, logits 2 | Repair the image/temporal representation mismatch using a low encoder LR |
| EM3 | 16 | 2 | EdgeTAM temporal stack; TinyViT frozen again | EM2 | task 1, memory 0.25, logits 1 | Improve long-range memory while limiting encoder drift |

All stages use three sampled objects, point/box prompting, random correction
frames, BF16 training, global batch 4, and resumable `last.pt` checkpoints. The
object-pointer KD loss is intentionally disabled: previous runs showed that it
was not reliably exposed by every teacher path, while mask-logit and memory
distillation already provide direct temporal behavior constraints.

With 50,337 usable training videos, one epoch is approximately 12,585 optimizer
updates at global batch 4. The complete 2+5+2 curriculum therefore schedules
about 113,265 updates. Full SA-V increases the frame/clip sampling pool; it does
not change updates per epoch unless the number of manifest video records or the
dataset multiplier changes.

## Final result: 2026-08-05

All three stages trained successfully. The saved EM1, EM2, and EM3 checkpoints
were subsequently evaluated on the complete SA-V validation and test splits.

| Stage | State | Updates | Training time | Full val | Full test |
|---|---|---:|---:|---|---|
| EM1 | complete | 25,170 | 3.90 h | complete | complete |
| EM2 | complete | 62,925 | 20.84 h | complete | complete |
| EM3 | complete | 25,170 | 9.41 h | complete | complete |

The curriculum consumed 113,265 optimizer updates and approximately 34.15
training hours on one 4xH100 node. This establishes engineering feasibility:
official temporal initialization, stage-to-stage checkpoint transfer, low-LR
joint image/temporal training, T16 refinement, checkpoint resume, and full
evaluation all ran without the earlier interface or disconnected-gradient
failures.

### Accuracy

| Stage | Split | Image mIoU | Image AP | Video J&F | J | F |
|---|---|---:|---:|---:|---:|---:|
| EM1 | SA-V val | 0.8402 | 0.7161 | 27.9 | 25.0 | 30.8 |
| EM1 | SA-V test | 0.8390 | 0.7191 | 28.8 | 26.2 | 31.3 |
| EM2 | SA-V val | 0.8308 | 0.6993 | 42.3 | 39.4 | 45.2 |
| EM2 | SA-V test | 0.8313 | 0.7063 | 44.6 | 41.8 | 47.3 |
| EM3 | SA-V val | 0.8308 | 0.6985 | 44.4 | 41.6 | 47.2 |
| EM3 | SA-V test | 0.8313 | 0.7064 | 45.8 | 43.1 | 48.5 |

Using the selected TV21 references of 72.4 val and 74.7 test J&F:

| Stage | Val J&F retention | Test J&F retention | Val gap | Test gap |
|---|---:|---:|---:|---:|
| EM1 | 38.5% | 38.6% | -44.5 | -45.9 |
| EM2 | 58.4% | 59.7% | -30.1 | -30.1 |
| EM3 | 61.3% | 61.3% | -28.0 | -28.9 |

No stage reaches the 95% quality-retention promotion threshold. EM3 would need
at least 68.78 val and 70.97 test J&F to satisfy that gate; it remains 24.38
and 25.17 points below those thresholds.

### Research signal

EM1 preserves the single-image model almost exactly but collapses on video.
Its 0.8402/0.8390 mIoU and 0.7161/0.7191 AP show that TinyViT, box prompting,
and the image mask decoder remain functional. The 27.9/28.8 J&F therefore
localizes the initial failure to memory writing, temporal reading, or recurrent
object-state propagation rather than general image segmentation.

EM2 provides the strongest positive signal. Joint TinyViT and temporal-stack
adaptation at T8 improves val J&F by 14.4 and test J&F by 15.8 over EM1. J and F
rise together, so this is a real recovery of region tracking and boundary
tracking rather than a boundary-only effect. The cost is modest image drift:
val mIoU retains about 98.9% and val AP about 97.7% of EM1.

EM3 adds only 2.1 val and 1.2 test J&F over EM2 while leaving image quality
effectively unchanged. T16 frozen-encoder refinement is useful, but has entered
diminishing returns. Longer context or more repetitions of the same objective
alone are unlikely to close the remaining approximately 29-point test gap.

The ordering is consistent on val and test, and every stage scores slightly
higher on test, so these results do not show a val-specific overfitting failure.
The stages are nevertheless a sequential curriculum, not independent
ablations: the EM1-to-EM2 and EM2-to-EM3 differences also include additional
updates, changed trainable modules, learning rates, and inherited checkpoints.
They establish where recovery occurred, but do not identify a single causal
hyperparameter.

### Latency validity

The image evaluator reports approximately 416 ms `set_image` latency for EM1
and EM2 but 42--49 ms for EM3. That nearly 10x difference is not credible as an
architectural speedup from the small stage change and likely reflects different
runtime, contention, warmup, or evaluation conditions. These values must not
be used for a speed claim. A dedicated isolated benchmark with the same node,
GPU, software revision, warmup, and cohort is required, but is not currently
warranted for promotion because all three checkpoints fail the quality gate.

### Decision and next experiment

The result is a successful feasibility demonstration but a negative model
selection result. EdgeTAM-style memory can learn with TinyViT: EM2 recovers
roughly 15 J&F and EM3 adds a smaller long-context gain. The current objective,
however, does not teach a sufficiently coherent recurrent object state.

The next experiment should start from EM2 or EM3 and target the temporal state
transition directly rather than repeat this curriculum unchanged. The highest
priority supervision is per-propagation-step teacher mask-logit and object
identity/state distillation, followed by memory write/read consistency and
correction-frame robustness. Stage-wise full validation should be retained so
future encoder drift or temporal regressions are localized immediately.

## Company command

Run in the foreground on the new 4xH100 node:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only origin main
mkdir -p /user-volume/tv21_edgetam_memory_logs

GPUS=0,1,2,3 \
SAM2D_ROOT=/group-volume/danny-dataset/sam2_distill \
SAV_ROOT=/group-volume/danny-dataset/SA-V \
MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/sav_train_6fps_full.parquet \
WANDB_MODE=online \
scripts/company/69_run_tinyvit21_edgetam_memory_v1.sh all 2>&1 | \
tee "/user-volume/tv21_edgetam_memory_logs/all_$(date +%Y%m%d_%H%M%S).log"
echo "Pipeline status: ${PIPESTATUS[0]}"
```

The `all` action performs input audit, EM1 training, EM2 training, EM3 training,
full validation, and then full test. A failure stops the dependency chain; it
does not train a later stage from a missing or invalid checkpoint.

To resume the same pipeline, rerun the identical command. Completed stages are
skipped; partial training resumes from the same checkpoint, W&B run ID,
TensorBoard directory, and run directory.

## Inspection commands

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
scripts/company/69_run_tinyvit21_edgetam_memory_v1.sh status
scripts/company/69_run_tinyvit21_edgetam_memory_v1.sh summarize
```

Final metrics are expected under:

```text
/group-volume/danny-dataset/sam2_distill/runs/tinyvit21_edgetam_memory_v1/
  <EM1_t4_official_temporal_2ep |
   EM2_t8_joint_edgetam_5ep |
   EM3_t16_memory_refine_2ep>/main/
      sav_val_box_benchmark/metrics.csv
      sav_test_box_benchmark/metrics.csv
```

## Decision outcome

| Criterion | Outcome |
|---|---|
| Three-stage engineering feasibility | Pass |
| Full val/test evaluation available for every saved stage | Pass |
| EM3 exceeds the approximately 60-J&F learned-memory line | Fail |
| At least 95% TV21 J&F retention | Fail |
| Eligible for promotion latency benchmark | No |

EM2 is the most informative recovery checkpoint and EM3 is the highest-quality
checkpoint, but neither is a deployment candidate. Preserve both for the next
temporal-state supervision experiment; do not spend another long run on the
same initialization and loss schedule without changing the learning signal.
