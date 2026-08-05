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

All three stages and the intended full validation/test evaluation completed.

| Stage | State | Updates | Training time | Evaluation |
|---|---|---:|---:|---|
| EM1 | complete | 25,170 | 3.90 h | Intentionally deferred |
| EM2 | complete | 62,925 | 20.84 h | Intentionally deferred |
| EM3 | complete | 25,170 | 9.41 h | Full SA-V val and test complete |

The full curriculum consumed 113,265 optimizer updates and approximately 34.15
training hours on one 4×H100 node. This establishes engineering feasibility:
official temporal initialization, stage-to-stage checkpoint transfer, low-LR
joint image/temporal training, long-context refinement, and full evaluation all
ran without the earlier interface or disconnected-gradient failures.

The final accuracy result is negative:

| Split | Image mIoU | Image AP | Video J&F | J | F | Seconds/video |
|---|---:|---:|---:|---:|---:|---:|
| SA-V val | 0.8308 | 0.6985 | 44.4 | 41.6 | 47.2 | 30.89 |
| SA-V test | 0.8313 | 0.7064 | 45.8 | 43.1 | 48.5 | 32.72 |

Against the selected TV21 references of 72.4 val and 74.7 test J&F, EM3 retains
only about 61.3% on both splits. It also falls below the earlier EdgeTAM Q2
result by 11.2 val and 12.2 test J&F, and below the approximately 60-J&F
shared-K/V line. It therefore fails both the 60-J&F diagnostic threshold and
the 95% promotion threshold. A multiplex latency benchmark is not warranted
for EM3.

The image results narrow the cause. EM3 retains roughly 99% of the earlier
0.840/0.839 image mIoU and 97--98% of image AP, while video J&F collapses by
about 29 points. The model can still segment prompted images; it cannot carry
object state coherently through time. The primary failure is the TinyViT to
EdgeTAM temporal representation/interface and its learned memory dynamics, not
a wholesale failure of the image encoder.

Only EM3 received full evaluation, so this curriculum does not identify which
transition caused the largest regression. In particular, the available result
cannot distinguish among:

1. poor official-temporal adaptation already present after EM1;
2. encoder/temporal co-adaptation drift introduced by EM2; and
3. degradation from the final T16 frozen-encoder refinement in EM3.

The next diagnostic should evaluate the already-saved EM1 and EM2 checkpoints
on the same fixed validation cohort before starting another training run. If
EM2 is materially better than EM3, discard the T16 refinement. If all three are
near 44--46 J&F, the current initialization and loss interface are the failure
and more epochs or more SA-V data are unlikely to repair it. A subsequent
EdgeTAM experiment should add direct temporal-output/identity supervision and
stage-wise validation rather than repeat this curriculum unchanged.

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
  EM3_t16_memory_refine_2ep/main/
    sav_val_box_benchmark/metrics.csv
    sav_test_box_benchmark/metrics.csv
```

## Decision criteria

The first decision is feasibility rather than promotion:

1. All three stages finish without missing/unexpected checkpoint tensors.
2. Full SA-V val and test both complete and report video-tracking J&F.
3. EM3 improves on the earlier roughly 60 J&F shared-K/V learned-memory line and
   approaches the 72.3/74.6 TV21 quality reference.
4. Only after quality is established should EM3 receive the full N=1/2/4/8
   multiplex latency benchmark. This experiment does not claim a speed gain by
   construction.

If EM1 cannot reach useful validation quality, the next experiment should test
interface alignment rather than add epochs. If EM1 succeeds but EM2 regresses,
the encoder LR or image-distillation weight is the primary ablation. If EM2
succeeds but EM3 regresses, retain EM2 as the candidate and reduce T16 temporal
LR before trying more data.
