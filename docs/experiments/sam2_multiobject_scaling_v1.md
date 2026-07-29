# SAM2 Multi-Object Shared-Session Scaling v1

## Research question

Can the current distilled SAM2 tracker handle more simultaneously prompted
objects without the near-linear FPS loss caused by expanding the object axis?

This project concerns the cost of point-selecting many objects in one video.
It does not attempt to transfer SAM3.1's text concept detector. The deployment
contract remains SAM2-compatible:

```text
one video state + N object IDs + N point prompts -> N propagated mask tracks
```

The first experiment is diagnostic and is runnable now. It measures the
current TinyViT-21M student and the unmodified SAM2.1-L teacher under exactly
the same shared-session workload. A trainable multiplexer is gated on those
measurements so that architecture work targets the measured bottleneck.

## Why FPS still falls in a shared SAM2 session

SAM2 already shares the image encoder across objects in a video state.
However, prompt decoding, object pointers, memory attention, memory encoding,
mask output, and post-processing still carry an object dimension. A useful
latency model is:

```text
T_frame(N) = T_shared_image
           + T_object_attention(N)
           + T_mask_output(N)
```

The final output cost cannot be constant because N masks must be produced.
The research target is therefore sublinear or bucketed growth, not perfectly
constant latency.

SAM3.1 is evidence that the object-axis computation can be reorganized:
the official release reports multiplex tracking for many objects, and the
implementation groups objects into fixed-capacity buckets while sharing image
features. We use the idea as an architecture reference, not as a claim that
SAM3.1 weights or its semantic detector can be copied into this student:

- [SAM3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
- [Official multiplex base implementation](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_multiplex_base.py)

## v1 hypotheses

- **H1 — object-path bottleneck:** TinyViT-21M and SAM2.1-L have similar
  relative latency slopes versus object count. If true, further encoder
  distillation cannot solve multi-object scaling.
- **H2 — encoder-dominated baseline:** TinyViT-21M has a meaningfully flatter
  slope than SAM2.1-L. If true, shared image encoding remains a substantial
  part of frame latency and absolute FPS must be reported alongside slope.
- **H3 — data support:** SA-V validation contains enough videos with at least
  16 non-empty masks on the same annotated frame to form a fixed, reproducible
  cohort. If false, use 1/2/4/8 objects and separately curate internal
  high-density videos before training a 16-object design.

## Runnable experiment matrix

| ID | Model / input | Changed variable | Output | Question answered |
|---|---|---|---|---|
| `MO-D0` | SA-V val annotations | none | density summary + fixed cohort | Is a 16-object benchmark supported by current data? |
| `MO-L0` | best TinyViT-21M task checkpoint | objects = 1/2/4/8/16 | latency, FPS, memory | What is the deployed student's scaling curve? |
| `MO-L1` | official SAM2.1-L | same cohort/counts | latency, FPS, memory | Is the scaling slope caused by the shared encoder or object path? |

All prompted objects are non-empty on one common frame. Each count uses a
prefix of the same sorted object IDs, every count sees the same videos, and
measurement order is seed-shuffled. This removes three common confounders:
different videos, different prompt frames, and different propagation lengths.

The benchmark separately records:

- state initialization seconds;
- all prompt-insertion seconds;
- propagation milliseconds per video frame and FPS;
- end-to-end FPS;
- emitted object masks per second;
- peak allocated GPU memory;
- per-video rows, medians, p90 latency, and ratios relative to one object.

One isolated H100 is intentional. DDP would change the runtime being measured,
and colocating another process on the same GPU invalidates latency. A 4-H100
node may run another job on GPUs 1–3, but this benchmark owns its selected GPU.

## Twelve-hour training matrix: four independent 4xH100 nodes

The first trainable intervention is deliberately smaller than a full
SAM3.1-style multiplexer: reduce the standard SAM2 memory-attention stack from
four layers to two, then recover quality by same-interface behavior
distillation. This can improve absolute multi-object FPS now, and it tells us
whether memory attention is enough of the object-axis cost to justify a later
fixed-bucket implementation. It does not claim constant FPS or sublinear
scaling by construction.

All four lanes start from the same selected TinyViT-21M checkpoint
(`tinyvit_max_jf_v1/tv21/main/checkpoints/best.pt`, full-val J&F 72.4). They
use the same T4 dense-object cohort, frozen image encoder and BatchNorm,
matched 50/50 point/box prompt mix without iterative correction clicks,
per-GPU batch one, global batch four, and five epochs. Correction clicks are
disabled so the company container does not depend on OpenCV `libGL` during
training. The only causal changes are memory depth and KD terms.

| Node / run | Memory | Objective | Research question |
|---|---:|---|---|
| `MO0_mem4_task_dense8_5ep` | standard 4-layer | task | Does dense eight-object continuation change the selected model? |
| `MO1_mem2_task_dense8_5ep` | standard 2-layer | task | What speed and quality are caused by depth reduction alone? |
| `MO2_mem2_logits_dense8_5ep` | standard 2-layer | task + propagated-mask logit KD | Can the 4-layer teacher recover tracking behavior? |
| `MO3_mem2_memlogits_dense8_5ep` | standard 2-layer | task + mask-logit KD + 0.5 memory KD | Does matching temporal state recover more J&F than logits alone? |

The training cohort contains videos with at least eight visible objects on at
least four annotated frames. It is selected deterministically from manual
SA-V JSON and repeated to 50,337 samples so all lanes receive the same update
budget. `max_num_objects=8` is therefore backed by dense data rather than
being only a sampler ceiling.

Each independent node executes this fixed pipeline:

```text
5-epoch train on 4 H100
  -> full SA-V validation on 4 H100
  -> full SA-V test on 4 H100
  -> fixed-cohort N=1/2/4/8 point-prompt latency on one isolated H100
  -> W&B + per-run CSV + central comparison CSV
```

Training checkpoints, TensorBoard, W&B run metadata, and evaluation artifacts
stay under `/danny-dataset/sam2_distill/runs/sam2_multiobject_training_v1`
when that mount is available. The runner can discover legacy selected weights
under `/group-volume`, but it does not copy intermediate outputs there.
Training is resumable in the same checkpoint directory and W&B run ID.

Five epochs are held constant for causal comparison. Based on the preceding
five-epoch temporal runs, online teacher cost, full evaluation, and latency
suite, budget **8–12 hours per node**. This is an empirical estimate, not a
timeout; shared-storage throughput can move it.

### Company commands

Use one block in each of four independent 4xH100 terminals. All commands run
in the foreground and preserve live output with `tee`.

Pull once before starting the four terminals because `/user-volume/repo` is
shared:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only
echo "Repository sync status: $?"
```

Node 1:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
mkdir -p /user-volume/sam2_multiobject_training_logs
GPUS=0,1,2,3 \
scripts/company/60_run_sam2_multiobject_training.sh \
  run MO0_mem4_task_dense8_5ep 2>&1 | \
tee /user-volume/sam2_multiobject_training_logs/MO0_mem4_task_dense8_5ep.log
echo "MO0 status: ${PIPESTATUS[0]}"
```

Node 2:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
mkdir -p /user-volume/sam2_multiobject_training_logs
GPUS=0,1,2,3 \
scripts/company/60_run_sam2_multiobject_training.sh \
  run MO1_mem2_task_dense8_5ep 2>&1 | \
tee /user-volume/sam2_multiobject_training_logs/MO1_mem2_task_dense8_5ep.log
echo "MO1 status: ${PIPESTATUS[0]}"
```

Node 3:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
mkdir -p /user-volume/sam2_multiobject_training_logs
GPUS=0,1,2,3 \
scripts/company/60_run_sam2_multiobject_training.sh \
  run MO2_mem2_logits_dense8_5ep 2>&1 | \
tee /user-volume/sam2_multiobject_training_logs/MO2_mem2_logits_dense8_5ep.log
echo "MO2 status: ${PIPESTATUS[0]}"
```

Node 4:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
mkdir -p /user-volume/sam2_multiobject_training_logs
GPUS=0,1,2,3 \
scripts/company/60_run_sam2_multiobject_training.sh \
  run MO3_mem2_memlogits_dense8_5ep 2>&1 | \
tee /user-volume/sam2_multiobject_training_logs/MO3_mem2_memlogits_dense8_5ep.log
echo "MO3 status: ${PIPESTATUS[0]}"
```

The latency phase intentionally uses only GPU 0 after the four-GPU
train/val/test phases. Running four latency workers would measure distributed
throughput rather than the deployed shared-session FPS.

### Training result table

This table answers whether the two-layer intervention produces a useful
quality–latency frontier. Select on validation J&F and N=8 latency only; test
remains a final held-out report.

| Run | Val J&F | Test J&F | N=1 FPS | N=8 FPS | N=8 latency / N=1 | Peak MB at N=8 | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| MO0 standard4 task | pending | pending | pending | pending | pending | pending | control |
| MO1 standard2 task | pending | pending | pending | pending | pending | pending | pending |
| MO2 standard2 + logits | pending | pending | pending | pending | pending | pending | pending |
| MO3 standard2 + memory/logits | pending | pending | pending | pending | pending | pending | pending |

The central table is:

```text
/danny-dataset/sam2_distill/runs/sam2_multiobject_training_v1/summary.csv
```

Promotion requires a real N=8 FPS gain over MO0 and a full-val J&F loss small
enough for the deployment target. If all two-layer variants keep nearly the
same relative N-slope, memory compression improves absolute speed but does
not solve multiplex scaling; proceed to fixed-capacity buckets rather than
adding more KD terms.

## Initial performance targets

These are promotion gates for a future multiplexed model, not assumptions
about the current baseline.

| Objects | Maximum propagation latency relative to N=1 |
|---:|---:|
| 2 | 1.10x |
| 4 | 1.25x |
| 8 | 1.60x |
| 16 | 2.20x |

A later architecture is only successful if it also satisfies:

- full SA-V val J&F drop no more than 0.3 at 4 objects, 0.5 at 8, and 1.0 at
  16 relative to the same task checkpoint;
- image mIoU/AP regression no larger than 0.005;
- no object-ID swaps on a persistent-ID diagnostic set;
- peak memory remains bounded by the selected bucket capacity.

Existing quality anchor: the current TinyViT-21M selected checkpoint has
full-val J&F 72.4 and full-test J&F 74.7. The latency suite does not generate a
new model and must not be used to revise that validation-selected checkpoint.

## Company runbook

Run from the company repository. Commands stay in the foreground, stream to
the terminal, write a persistent log with `tee`, and create W&B runs in
`sam2-multiobject-scaling-v1`.

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only
scripts/company/59_run_sam2_multiobject_scaling.sh describe
scripts/company/59_run_sam2_multiobject_scaling.sh all
echo "Multi-object scaling status: $?"
```

The default is point prompting on 1/2/4/8/16 objects, eight fixed videos, two
measured repetitions, and one discarded warmup video. To lower the density
requirement when MO-D0 finds too few 16-object videos:

```bash
OBJECT_COUNTS=1,2,4,8 \
MAX_VIDEOS=16 \
scripts/company/59_run_sam2_multiobject_scaling.sh all
echo "Eight-object scaling status: $?"
```

To run the two model curves on separate nodes after generating the shared
cohort once:

```bash
# Node A
GPU=0 scripts/company/59_run_sam2_multiobject_scaling.sh tv21
echo "TinyViT-21M scaling status: $?"
```

```bash
# Node B
GPU=0 scripts/company/59_run_sam2_multiobject_scaling.sh sam21l
echo "SAM2.1-L scaling status: $?"
```

Set `PROMPT_KIND=box` for the matched box-prompt curve. Set `SKIP_DONE=0` only
when intentionally repeating a completed measurement. The driver discovers
the current company data root from `/danny-dataset`, `/group-volume`, or the
legacy `/mnt/data` mount and fails before model loading when a required
checkpoint is missing.

Default outputs:

```text
<sam2_distill>/runs/sam2_multiobject_scaling_v1/
  density_audit_n16/
    cohort.txt
    per_video.csv
    summary.json
  tv21_best/point_n1-2-4-8-16/
    per_video.csv
    aggregate.csv
    summary.json
  sam21l/point_n1-2-4-8-16/
    per_video.csv
    aggregate.csv
    summary.json
```

## Reading the result

1. First inspect `density_audit_n16/summary.json`. If no 16-object cohort exists,
   rerun at 8 objects; do not silently benchmark different videos per count.
2. Compare absolute `median_propagation_fps` for deployment.
3. Compare `relative_latency_vs_1` between MO-L0 and MO-L1 to localize the
   scaling bottleneck.
4. Use p90 and per-video rows to check whether the median hides long clips or
   crowded scenes.
5. Do not claim acceleration from object-masks/second alone; deployment FPS is
   video frames/second.

## Gated architecture sequence after v1

The four memory-depth lanes may run alongside MO-L0/MO-L1 because they reuse
an already validated topology intervention. Do not implement the larger
fixed-bucket/object-slot architecture until the baseline slopes and these
four quality–latency points are complete.

| ID | Change | Control | Promotion signal |
|---|---|---|---|
| `MO-MX0` | exact current shared-session inference | MO-L0 | correctness/runtime control |
| `MO-MX1` | fixed-capacity object buckets, capacity 4 | MX0 | 4-object latency <= 1.25x |
| `MO-MX2` | capacity 8 with shared projected memory K/V | MX1 | 8-object latency <= 1.60x and val J&F drop <= 0.5 |
| `MO-MX3` | object-slot attention + demultiplexed mask heads | MX2 | flatter 16-object slope without ID swaps |
| `MO-MX4` | SAM2.1-L behavior distillation on dense clips | MX3 | recover any multi-object J&F loss |
| `MO-MX5` | compile/export fixed 4/8/16 buckets | MX4 | deployment latency reproduces benchmark gain |

Training data should sample a dense-clip stratum rather than only increasing
the existing `max_num_objects=2`. Each batch must preserve object IDs across
frames and mix 1/2/4/8/16-object examples, otherwise a model may improve dense
scenes while regressing the dominant one-object case. MX1 is the minimum
credible implementation target; distillation is a recovery objective, not
the source of the speedup.

## Result table

Populate this table from each `aggregate.csv`; selection must use validation
quality plus measured latency, never SA-V test quality alone.

| Run | Objects | Median FPS | Relative latency | p90 ms/frame | Peak MB | Gate |
|---|---:|---:|---:|---:|---:|---|
| `MO-L0` TinyViT-21M | 1/2/4/8/16 | pending | pending | pending | pending | pending |
| `MO-L1` SAM2.1-L | 1/2/4/8/16 | pending | pending | pending | pending | diagnostic |
