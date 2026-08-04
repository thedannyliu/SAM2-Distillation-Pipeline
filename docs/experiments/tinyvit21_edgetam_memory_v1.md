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

## Progress snapshot: 2026-08-04

| Stage | Observed state | Persisted progress | Trainable parameters | Evaluation |
|---|---|---:|---:|---|
| EM1 | Training complete | epoch 2; 25,170 updates | 4,775,392 / 29,800,722 | Intentionally deferred |
| EM2 | Active; confirmed still running by the operator | 25,170 updates in `checkpoint.pt`, equivalent to the epoch-2 boundary of 5 | 25,561,040 / 29,800,722 | Pending |
| EM3 | Not started | 0 / 25,170 planned updates | Pending construction | Pending |

EM1 took 14,039.96 seconds, or about 3.90 hours, for its two epochs. At least
50,340 of the curriculum's 113,265 updates have therefore been persisted across
EM1 and EM2. This is a lower bound because a running process may have advanced
beyond its most recent checkpoint.

The result is positive at the engineering-feasibility level:

1. The best TinyViT-21M task checkpoint accepted the coherent official EdgeTAM
   temporal initialization and completed two epochs with no recorded checkpoint
   loading or backward-pass error.
2. EM2 successfully loaded EM1 through `current_full`, enabled gradients for the
   encoder plus memory stack, and reached at least 25,170 updates. This removes
   two failure modes seen in earlier experiments: broken temporal-call
   interfaces and losses disconnected from trainable parameters.
3. There is not yet a research-quality result. No validation, test, J&F, or
   latency metric exists, so the experiment currently demonstrates trainability
   rather than accuracy or speed.

The `.full_eval_required` markers on EM1 and EM2 are expected. Intermediate
stages run with evaluation disabled; the orchestrator performs full validation
and test only after EM3. The original resolved configs also showed
`scratch.max_num_objects: 2` while the effective dataset sampler and experiment
summary both showed 3. Runtime sampling was therefore three objects, but the
scratch field was misleading. Commit following this snapshot keeps the scratch
and sampler fields synchronized for subsequent resolved configs.

No restart or resume action is warranted at this snapshot. The recent error
scan was empty, EM2 has a valid in-progress checkpoint, and the operator
confirmed that experiments are still running. Observe the current jobs without
changing their code or process state unless a real error appears.

The next decision point is the completed EM3 full validation. If its val J&F is
below 60, the official temporal path has not surpassed the earlier shared-K/V
line and should not receive a latency benchmark. A result from 60 to below the
95% quality-retention threshold is useful but remains an accuracy/speed tradeoff.
At or above 68.8 val J&F (95% of the 72.4 reference), proceed to full test and
N=1/2/4/8 latency measurement.

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
