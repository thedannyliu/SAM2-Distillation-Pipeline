# SAM2.1-L two-clock stale-observation reuse v1

## Status and documentation contract

**Status:** Wave-1 implementation complete on the research branch; local CPU
contracts pass. Company input audit and 4xH100 E smoke are the remaining hard
gates before the six formal jobs.

This file is the authoritative design, execution plan, and result ledger for
the experiment. Update it after every smoke, training run, validation, and
promotion decision. Each recorded run must include the task, seed, data split,
manifest or video root, GPU type/count, checkpoint/output directory, W&B
project/run ID, command, result, and decision. Machine-readable metrics remain
under the company run root; this document records only the concise evidence
needed to answer each research question.

The design was approved on 2026-08-14. Formal GPU training remains gated by
the repo-owned `audit` and `smoke` actions below.

### Design-review resolution (2026-08-14)

| Review issue | Locked resolution |
|---|---|
| double-zero temporal residual | remove scalar gates; zero-init only the final residual projections |
| discrete age versus stride 2/4 | v1 is contiguous raw 24 FPS only; remove variable-stride evaluation |
| A control attribution | compare A to O2 at matched R2--R6; use O2 minus O1 for generic fine-tuning transfer |
| FiLM leaking into memory writes | immutable `F_raw_cache`; FiLM creates read-only `F_read`; A/B/C MemEnc uses raw cache |
| recency/freshness bundled in C | call Wave-1 C the combined `C-rf`; conditionally add C-r after a positive C screen |
| D mechanism overclaim | claim stale-write suppression only; no causal spatial-misalignment claim |
| E teacher advantage | call it privileged full-refresh state distillation, including denser teacher memory |
| gate horizon bias | fixed `k..k+2`; execute future forced refreshes instead of truncating |
| heterogeneous gate target | use one 0--1 soft-Dice cost for GT and teacher targets |
| checkpoint ambiguity | select every O2/A--E checkpoint by full SA-V val R4 J&F |
| DAVIS reuse | use SA-V val for every choice; run DAVIS once only after freeze |
| single-seed threshold | one-seed screening, then two confirmation seeds plus paired video bootstrap |
| growing pointer history | lock official spatial capacity 7 and model-visible pointer capacity 16 |

## Research claims

The primary claim is deliberately narrower than future prediction:

> SAM2.1-L can reuse stale image embeddings more effectively when it explicitly
> models current feature age, memory recency, and memory freshness, while
> suppressing spatial-memory writes made from stale observations.

Two stronger claims require separate evidence:

1. **Predictive latent dynamics** is claimed only if the E+ spatial transition
   improves E, with a larger gain at ages 3--5 than at ages 1--2.
2. **Learn when to observe** is claimed only if a gate trained on a frozen
   parent with multi-step refresh value beats fixed, random, and age-only
   policies at matched encoder refresh rates.

Encoder skipping or a learned gate alone is not the novelty. Efficient-SAM2
and Lean-SAM2 already route current-frame computation, while Deep Feature Flow
established keyframe feature propagation. The intended contribution is the
combination of a two-clock representation, stale-safe memory, fresh-to-stale
state supervision, and compute-matched adaptive observation.

## Fixed model, data, and storage contract

### Model weights

Student and teacher use the same official architecture and initializer:

- Config: official SAM2.1 Hiera Large.
- Checkpoint:
  `/group-volume/danny-dataset/sam2_distill/checkpoints/sam2.1/sam2.1_hiera_large.pt`.
- Student: trainable SAM2.1-L copy.
- Teacher: frozen SAM2.1-L copy, always fresh, excluded from DDP, optimizer,
  checkpoints, and gradient graphs.
- Prompt encoder: frozen in the student.
- BatchNorm: every existing BatchNorm module stays in evaluation mode with
  frozen affine parameters; no new module uses BatchNorm.

Using the same model on both sides removes capacity and representation
mismatch and permits direct post-memory feature and object-pointer losses
without an adapter. It does **not** isolate fresh current observation in E:
the teacher also writes fresh spatial memory every frame, whereas the D/E
student writes spatial memory only on refresh frames. E therefore distills a
privileged full-refresh state containing both a fresh current observation and
a denser, fresher memory history.

### Data

- Raw train release: `/group-volume/danny-dataset/SA-V/sav_train`.
- Reference manifest:
  `/group-volume/danny-dataset/sam2_distill/manifests/sav_train_6fps_full.parquet`.
- Training population: all 50,337 videos with a readable matching manual JSON;
  leave `TASK_VIDEO_IDS_FILE` empty.
- Validation: full `/group-volume/danny-dataset/SA-V/sav_val`.
- Test: full `/group-volume/danny-dataset/SA-V/sav_test`, accessed only after
  the checkpoint and all policies are frozen from validation.
- Frozen external diagnostic: DAVIS 2017 val under
  `/group-volume/danny-dataset/DAVIS/2017` after the user prepares it and after
  all architecture/policy choices are frozen from SA-V val.
- SA-V automatic JSON is not a hard target in v1. The manual SA-V annotation
  is the only hard label; frozen-teacher outputs supervise unannotated 24 FPS
  frames.

Version 1 has one fixed temporal unit: one raw 24 FPS frame. It does not train
or evaluate decimated strides 2/4, because the discrete current-age and
freshness tables have support only for raw-frame ages 0--5. Variable-FPS or
larger-gap evaluation requires a separately designed continuous-time encoder
and is explicitly out of scope for v1.

SA-V training frames are decoded online from MP4. Do not create a full 24 FPS
JPEG copy, cropped-image cache, teacher-feature cache, or pseudo-mask cache.
Full SA-V val/test already use `JPEGImages_24fps` with `Annotations_6fps`.

### Artifact locations

- Run root:
  `/group-volume/danny-dataset/sam2_distill/runs/sam21l_two_clock_reuse_v1`.
- TensorBoard root:
  `/group-volume/danny-dataset/sam2_distill/logs/sam21l_two_clock_reuse_v1`.
- Terminal logs: `/user-volume/log/sam21l_two_clock_reuse_v1`.
- W&B project: `sam2-two-clock-reuse-v1`.
- Final selected weights:
  `/group-volume/danny-dataset/sam2_distill/checkpoints/final_weights/sam21l_two_clock_reuse_v1`.
- Screening seed: `250107256` for training, sampling, policy controls, and
  evaluation sharding. A schedule is a deterministic function of seed, epoch,
  and video ID so O2 and A--E see matched clips and augmentations.
- Confirmation seeds: `250107257` and `250107258`, used only for positive
  mechanism comparisons that are retained as final claims.

## Prompt and 24 FPS sampling protocol

### T8 causal clip

Each dataset item selects one valid manual-annotation anchor and decodes eight
consecutive raw 24 FPS frames:

```text
SA-V MP4
   |
   +-- select an annotation-aligned anchor t0 with a visible object
   +-- decode [t0, t0+1, ..., t0+7] in one sequential read
```

- One clip is sampled per usable video per epoch.
- Up to three objects visible at `t0` are sampled.
- The clip is always forward causal; reverse-time sampling is disabled.
- If the requested range exceeds the video, resample an earlier valid anchor.
- Raw 24 FPS indices survive collation and drive every temporal calculation.
- T8 is the formal first-wave horizon. T12/T16 is not mixed into the main
  ablation.

### Anchor prompt mixture

The same prompt plan and exact coordinates/mask are passed to teacher and
student:

| Prompt | Probability | Construction |
|---|---:|---|
| mask | 0.50 | exact SA-V manual object mask |
| box | 0.25 | tight GT box with shared, clipped +/-5% side jitter |
| positive point | 0.25 | one point sampled inside the GT object mask |

There are no later GT corrections. Later manual masks are loss targets only;
they never become conditioning frames or teacher/student memory inputs.

- A mask-prompt anchor is excluded from segmentation loss because the official
  mask-input path can return the mask directly.
- A box/point anchor receives normal high-weight GT loss because the prediction
  is not a copied target.
- Primary model selection uses first-frame GT-mask VOS. Box and point are
  secondary robustness checks and never select a checkpoint.

## Teacher and supervision pipeline

The teacher receives every decoded frame and maintains its own on-policy,
full-refresh SAM2 memory:

```text
I_k -> frozen Hiera-L encoder -> fresh F_k^T
    -> official teacher memory attention -> F_M,k^T
    -> official mask decoder -> logits, pointer, score, predicted IoU
    -> official memory encoder -> teacher-only memory bank
```

The teacher receives only the shared anchor prompt. Later GT never corrects
it. Teacher outputs are detached immediately. Student rollout never consumes
teacher memory, teacher masks, or teacher pointers as state.

For an object/frame pair, manual SA-V GT takes precedence:

\[
L_{GT}=20L_{focal}+L_{dice}+L_{IoU}+L_{object}.
\]

When a manual target is unavailable or `None`, use soft online output-level
distillation:

\[
L_{PL}=BCEWithLogits(z^S,\sigma(z^T))
       +SoftDice(\sigma(z^S),\sigma(z^T)).
\]

Losses are group-normalized, not weighted per frame and then pooled:

\[
L_{sup}=mean(L_{GT})+0.25\,mean(L_{PL}),
\]

with an absent group contributing zero. This makes the entire GT group four
times more important than the pseudo group regardless of how many
unannotated frames occur in a clip. On a GT frame, the teacher mask target is
not also applied. A--D therefore already contain output-level fresh-to-stale
distillation; E is named **privileged full-refresh state distillation**, not
the first distillation experiment.

## Causal refresh trajectories and real encoder skipping

The anchor is always a refresh. For A--E:

- 25% of clips refresh every frame.
- 75% draw a reuse-run length uniformly from `{1,2,3,4,5}` after each refresh,
  then refresh and draw again.
- Maximum permitted feature age is five raw 24 FPS frames. A decision that
  would produce age six is forced to refresh.
- The schedule is per video frame, shared by all tracked objects, because an
  image-encoder call is also shared.

Example:

```text
frame       0 1 2 3 4 5 6 7
action      R - - - R - - R
source      0 0 0 0 4 4 4 7
age         0 1 2 3 0 1 2 0
```

The implementation must encode only the unique refresh images, then gather
all three cached feature scales through a source-frame map. Computing every
frame and replacing tensors afterward is invalid. Every run records
`encoder_calls / video_frames` and asserts that it matches the realized
refresh schedule.

For fixed-policy evaluation, `R_K` means one encoder refresh every `K`
consecutive raw 24 FPS frames starting from the prompted anchor. Thus R1 is
full refresh, R4 has ages 0--3, and R6 has ages 0--5. O1, O2, and A--E use the
exact same cache/gather inference implementation at each R2--R6 setting; only
their learned weights and registered mechanisms differ.

### Raw cache versus conditioned read features

The encoder output and the age-conditioned read path are conceptually and
programmatically separate tensors:

```text
refresh I_s -> Hiera -> F_raw_cache[s] at all three scales
                         |
                         +-> age FiLM -> F_read[k]
                         |                 +-> memory-attention query
                         |                 +-> decoder high-resolution skips
                         |
                         +-> MemEnc(F_raw_cache[s_k], mask_k) in A/B/C
```

- `F_raw_cache` is never modified in place by FiLM.
- A has `F_read == F_raw_cache`; B--E construct `F_read` from the raw cache.
- Current-age FiLM never conditions image features passed to `MemEnc`.
- A, B, and C therefore all write the same type of raw stale visual content;
  only their read paths differ.
- D, E, and E+ use the raw current feature for refresh-frame writes and skip
  `MemEnc` entirely on reuse frames.
- Vision positional encodings remain official and unmodified on the query
  side. Only the memory positional residuals defined below change in C onward.

This ownership rule is a hard ablation invariant. An implementation that
overwrites a shared feature dictionary with FiLM output is invalid.

## Temporal conditioning module and fusion operations

The time module is small and independent of Hiera. It has three semantically
separate branches:

\[
a_k=k-s_k,\qquad r_{kj}=k-j,\qquad f_j=j-s_j.
\]

Here `k` is prediction time, `s_k` is the current cached observation time, `j`
is a memory prediction time, and `s_j` is the observation used to create that
memory.

### Current feature age

`a in {0,...,5}` uses `Embedding(6,64)`. Inputs outside this range are an
assertion failure rather than clipped or bucketed. Four independent heads have
`Linear(64,64) -> SiLU -> Linear(64,2C)` for channels 32, 64, 256, and the
post-memory decoder input at 256 channels. Each last linear is zero-initialized.

For a feature `F_l`:

\[
F_l^{age}=F_l\odot(1+m(a)\gamma_l(a))+m(a)\beta_l(a),
\qquad m(a)=\mathbf 1[a>0].
\]

`gamma` and `beta` have shape `[B,C,1,1]` and are broadcast spatially. This is
channel-wise FiLM, not concatenation. The explicit bypass guarantees the
official path bit-for-bit at age zero, independent of later learned biases.

FiLM is applied to all read-path feature scales:

| Feature | Shape | Consumer |
|---|---|---|
| high-resolution level 0 | `[B,32,256,256]` | mask-decoder read skip |
| high-resolution level 1 | `[B,64,128,128]` | mask-decoder read skip |
| image embedding | `[B,256,64,64]` | memory-attention read query |
| post-memory state | `[B_obj,256,64,64]` | mask decoder |

The decoder therefore receives age through both its high-resolution skips and
a separate post-memory FiLM. No time token is concatenated to prompt tokens,
and official projection dimensions remain unchanged.

### Memory recency

Recency is unbounded and therefore uses a 64-dimensional fixed sinusoidal
encoding followed by `Linear(64,64) -> SiLU -> Linear(64,64)`. The delta is the
actual raw-frame difference, not a memory-slot rank. Elapsed seconds are also
recorded as `r/24` for analysis.

### Memory freshness

Freshness is bounded by the hard age policy and uses a distinct
`Embedding(6,64)` followed by `Linear(64,64) -> SiLU -> Linear(64,64)`.
Age and freshness tables are not shared because current-query staleness and
historical-memory staleness have different semantics.

### Memory injection

Temporal information does not alter spatial memory content. It is added as a
residual to official positional encodings:

\[
PE'_j=PE^{official}_j+P_r(e_r)+P_f(e_f).
\]

The 64-channel residual is broadcast to each spatial location in that memory
entry. Object-pointer positional tokens receive separate 256-dimensional
projections by element-wise addition. Only the last linear in each residual
projection `P_r`, `P_f`, `P_r^ptr`, and `P_f^ptr` is zero-initialized; there is
no additional zero-initialized scalar multiplier. Official slot PE and pointer
PE remain present.

This initialization is identity-preserving but trainable on the first
backward pass: at initialization the residual is zero while the loss gradient
with respect to each final projection weight is generally nonzero. Initializing
both a scalar multiplier and its projection to zero is forbidden because it
would make the whole branch stationary at zero.

### Fusion summary

| Signal | Destination | Operation |
|---|---|---|
| feature age | three image-feature scales | channel FiLM, spatial broadcast |
| feature age | post-memory decoder state | separate channel FiLM |
| memory recency/freshness | spatial-memory PE | residual element-wise addition |
| pointer recency/freshness | pointer positional token | token-wise addition |
| teacher mask/state/pointer | student | loss only; never concatenated |
| gate images/difference/mask | new gate CNN | channel concatenation |
| gate visual/time/state vectors | new gate MLP | vector concatenation |

Concatenation is reserved for the new gate because it has no pretrained
channel contract. Pretrained SAM2 tensors use identity-initialized FiLM or
positional residuals.

## Controlled experiment matrix

### Phase 0: establish the phenomenon and fine-tuning control

| ID | Training | Observation policy | Purpose | Status |
|---|---|---|---|---|
| O0 | none | official full refresh | untouched accuracy/latency reference | planned |
| O1 | none | frozen naive reuse, R1--R6 | true unadapted age degradation | planned |
| O2 | full matched fine-tune | always refresh | isolate gains from extra SA-V fine-tuning | planned |

O2 uses the same clips, prompts, teacher, GT/pseudo loss, optimizer, trainable
scope, epochs, and seed as A--E, but every age is zero. It contains no temporal
module. After training, O2 is evaluated not only at R1 but also through the
same naive-reuse R2--R6 inference path as O1 and A; otherwise O2 cannot control
generic fine-tuning transfer under stale inference.

### Phase 1: survive stale observations

| ID | Added mechanism | Spatial write policy | State KD | Question | Status |
|---|---|---|---|---|---|
| A | schedule exposure only | every frame, including stale | no | what does stale-trajectory training alone recover? | planned |
| B | A + current-age FiLM | every frame | no | does knowing query age help? | planned |
| C (`C-rf`) | B + recency and freshness PE | every frame | no | does combined actual-time metadata help? | planned |
| D | C + stale-write suppression | prompt/refresh frames only | no | does suppressing stale spatial writing help? | planned |
| E | D + privileged full-refresh state distillation | prompt/refresh frames only | yes | can privileged teacher state supervise stale student state? | planned |

All six trained first-wave runs O2 and A--E start independently from the same
official checkpoint. They are not checkpoint continuations of one another.

### A: schedule-trained naive reuse

The cached feature enters official memory attention and mask decoder without
time conditioning. Every frame runs the official memory encoder, including
`MemEnc(F_raw_cache[s], mask_k)` when `s < k`.

The controls are interpreted at the same fixed R2--R6 inference schedules:

- O1 measures the unadapted official checkpoint's naive-reuse degradation.
- O2 minus O1 measures how much generic matched SA-V fine-tuning transfers to
  naive reuse without stale-trajectory training.
- A minus O2 measures the additional effect of explicit stale-trajectory
  training.

`A - O1` is not called schedule-exposure gain because it also includes generic
SA-V fine-tuning.

### B: current-age conditioning

B applies the four age-FiLM heads described above only on `F_read` and the
post-memory state. It retains official memory time behavior and writes raw
stale spatial memory every frame from `F_raw_cache`. B minus A therefore
isolates current-query/decoder age information rather than silently changing
memory-write content.

### C: two-clock memory

C records `(j,s_j)` on every entry and adds both actual recency and freshness
residuals to spatial and pointer positional encodings. It still writes raw
stale spatial memory. C minus B tests the **combined** metadata addition; it
does not isolate which clock caused a gain.

If screening C exceeds B on the pre-registered R4 selection metric, add one
matched `C-r` run with actual recency only and no freshness encoder. Then:

- `C-r - B` measures the contribution of replacing slot-rank-only timing with
  actual raw-frame recency;
- `C-rf - C-r` is the required evidence for memory freshness, the second
  clock.

If C does not exceed B, `C-r` is not launched and no standalone freshness
claim is made.

### Model-visible memory capacity and selection invariant

All C/D/E/E+ comparisons lock memory-attention capacity and pointer selection:

- spatial memory keeps official `num_maskmem=7` capacity;
- object-pointer attention keeps official
  `max_obj_ptrs_in_encoder=16` capacity;
- selected conditioning-frame pointers are considered first, followed by the
  nearest available causal non-conditioning pointers, exactly as the official
  ordering; at most 16 pointer frames are visible to attention;
- every pointer carries `(j,s_j)` for recency/freshness conditioning, but
  timestamps retained for audit outside the selected set are logging only and
  never create additional attention tokens;
- C, D, E, and E+ use the same pointer emission, cap, ordering, and selection
  code. D changes only spatial-write eligibility.

The implementation must assert both selected spatial-entry and selected
pointer counts on every frame. A per-frame pointer history may be retained in
the rollout dictionary, but model-visible pointer attention must never grow
with video length.

### D: stale-safe memory

D keeps the C read path and changes only writes:

```text
prompt or refresh: run memory encoder and write spatial memory + pointer
reuse:             skip memory encoder/spatial write; retain pointer/state
```

The spatial bank holds the prompted anchor plus the most recent refresh
anchors up to official `num_maskmem=7`. A pointer is emitted every frame, but
the model-visible pointer bank follows the fixed 16-frame cap and selection
policy above. D minus C tests whether **suppressing stale spatial writes**
helps. It does not by itself prove spatial misalignment is causal, because D
also changes spatial-memory density and redundancy. A matched-write-count
control would be required before making a stronger poisoning-mechanism claim;
that control is deferred unless the result warrants it.

### E: privileged full-refresh state distillation

E adds losses on reuse frames only:

\[
L_E=L_{sup}+0.25L_{FM}+0.10L_{ptr}+0.10L_{score},
\]

where each auxiliary term is averaged over its valid reuse object/frames:

\[
L_{FM}=MSE(F_M^S,F_M^T),\quad
L_{ptr}=1-\cos(p^S,p^T),
\]

\[
L_{score}=BCEWithLogits(q^S,\sigma(q^T)).
\]

No loss attempts to reconstruct a fresh raw image embedding from a stale raw
embedding. E minus D isolates intermediate-state/pointer/behavior
distillation beyond the output-level pseudo supervision shared by A--D. It
does not isolate fresh current observation alone: the teacher has both fresh
per-frame visual features and denser fresh spatial memory. A future
matched-memory fresh-teacher control is required if that narrower mechanism
claim becomes important.

## Phase 2: E+ predictive latent transition

E+ is an independent matched run from the official checkpoint, not extra
epochs appended to E. It uses E's complete architecture and budget plus one
spatial residual predictor after memory attention. This is necessary before
using the phrase predictive latent dynamics.

For reuse frames:

\[
\hat Z_k=Z_k^{stale}+P_\theta(Z_k^{stale},p_{k-1},e_a).
\]

The predictor is:

```text
Z_stale [B_obj,256,64,64]
  -> depthwise 3x3 conv -> GroupNorm(32) -> SiLU
  -> FiLM from concat(age embedding, Linear(pointer))
  -> depthwise 3x3 conv -> GroupNorm(32) -> SiLU
  -> pointwise 1x1 conv, zero initialized -> delta_Z
  -> residual add: Z_pred = Z_stale + delta_Z
```

The pointer projection is used only to produce FiLM parameters; it is not
spatially concatenated to `Z`. Age zero explicitly bypasses the predictor.
The mask decoder consumes `Z_pred`. E's state loss is applied to `Z_pred`
rather than also penalizing pre-predictor `Z_stale`, keeping the total loss
weights matched:

\[
L_{pred}=MSE(Z_k^{pred},Z_k^T),\qquad \lambda_{pred}=0.25.
\]

E+ is launched only if E first passes the state-distillation screening gate in
the pre-registered criteria below. E+ is promoted as the gate parent only if
all conditions hold on full SA-V val:

1. E+ improves E by at least 0.5 J&F at R4.
2. Mean E+ minus E gain at ages 3--5 exceeds the gain at ages 1--2.
3. E+ R1 is no more than 1.0 J&F below O2.

Otherwise E remains the parent and the paper uses the stale-aware, not
predictive, claim. These single-seed thresholds are resource-allocation gates,
not final statistical evidence. DAVIS is not consulted for launch, promotion,
checkpoint selection, or gate calibration.

## Phase 3: frozen-parent adaptive observation

F starts only after E/E+ selection. The selected parent is frozen, so parent
versus F differs only in refresh policy. No joint gate/student co-adaptation is
part of the paper comparison.

### Gate inputs and fusion

Before an encoder call, resize current RGB, last-refresh RGB, and previous
union mask to 128 square pixels. Concatenate channels:

```text
current RGB        3
last-refresh RGB   3
absolute RGB diff  3
previous union     1
--------------------
gate input        10 channels
```

A three-block depthwise/pointwise CNN uses GroupNorm and SiLU, then global
average pooling to a 128-vector. Concatenate that vector with a 16-dimensional
age embedding and six normalized scalars: minimum object score, minimum
predicted IoU, maximum absolute area change, maximum centroid displacement x
and y, and current age/5. A two-layer MLP predicts scalar refresh value. No
gate layer uses BatchNorm or a full SAM2 feature.

### H=2 multi-step value of refresh

From one identical frozen-parent state at decision time `k`, compare:

- branch R: refresh at `k`, then follow reuse-until-forced for two frames;
- branch U: reuse at `k`, then follow the same reuse-until-forced policy for
  two frames.

Each branch updates only its own pointer/spatial state according to the frozen
parent. If either branch reaches the hard maximum age at `k+1` or `k+2`, that
branch executes the required refresh; this future consequence is part of the
value and is never truncated away. Sample a gate-training decision only when
`k+2` exists, so every target has the same three-frame horizon.

Targets use GT when available and the detached online-teacher mask otherwise,
but the quality cost has one common 0--1 definition for both groups. For each
frame, average soft Dice across valid objects and define:

\[
d_{k+h}^{branch}=1-mean_o\,SoftDice(\hat M_{k+h,o}^{branch},T_{k+h,o}).
\]

The oracle target is:

\[
V_k=\Delta d_k+0.8\Delta d_{k+1}+0.64\Delta d_{k+2},
\]

\[
\Delta d_{k+h}=d^{U-rollout}_{k+h}-d^{R-rollout}_{k+h}.
\]

Gate training excludes decisions at `k` that are already forced by prompt
events or the hard maximum age. It never uses the heterogeneous segmentation
training objective as refresh value. `V_k` is detached, and the gate minimizes
Smooth L1. The teacher, counterfactual branches, and parent weights are absent
at inference.

### Matched-compute policies

Validation produces a J&F-versus-refresh-rate curve. At each learned-gate
operating point, compare policies within 0.5 percentage points of its realized
refresh rate:

| Policy | Definition |
|---|---|
| full | refresh every frame |
| fixed-K | periodic R2--R6 schedule starting from the prompted anchor |
| random | exact per-video refresh budget sampled with frozen seed |
| age-only | same value-regression protocol using only feature age |
| learned | full gate inputs |
| oracle | rank true multi-step `V_k` under the same per-video budget |

The oracle is explicitly non-deployable and measures routing headroom. If it
does not beat fixed scheduling, content-adaptive refresh has little value. If
oracle is strong but learned is weak, the scheduling hypothesis survives but
the gate predictor fails.

The selected operating point maximizes measured end-to-end speed subject to
no more than 1.0 J&F loss from the same frozen parent at R1.

## Six-node execution schedule

### Preflight, before formal jobs

1. Audit all required paths and the 50,337-video manual-annotation contract.
2. Run the official SAM2 Torch 2.4 compatibility import/forward smoke inside
   the company container; stop if it fails rather than upgrading Torch.
3. Verify sequential 24 FPS MP4 decoding and raw-index/GT-validity collation.
4. Run one T8 forward/backward with paired teacher/student prompts.
5. Run a 4-GPU, 64-video DDP smoke for O2, D, and E.
6. Assert teacher freeze/exclusion, age-zero identity, nonzero first-backward
   gradients on zero-output temporal projections, raw-cache immutability,
   MemEnc raw-feature ownership, temporal metadata, refresh-only spatial
   writes, fixed memory/pointer capacity, actual encoder-call count,
   checkpoint resume, and W&B/TensorBoard directory reuse.
7. Lock per-GPU video batch to 1 (global batch 4). D/E have ragged spatial
   banks because reuse frames omit writes; v1 deliberately avoids an
   unvalidated per-sample memory-padding mask. Record the smoke HBM and step
   time, but do not promote batch 2 in Wave 1.

### Wave 1: launch together on six 4xH100 nodes

| Node | Run | Initializer |
|---:|---|---|
| 1 | O2 full-refresh fine-tune | official SAM2.1-L |
| 2 | A schedule-trained reuse | official SAM2.1-L |
| 3 | B + age FiLM | official SAM2.1-L |
| 4 | C + two-clock memory | official SAM2.1-L |
| 5 | D + stale-safe memory | official SAM2.1-L |
| 6 | E + privileged full-refresh state KD | official SAM2.1-L |

O0/O1 are evaluation-only and run during preflight or on the first released
node; they do not consume a formal training lane.

All Wave-1 runs use:

- T8, five complete SA-V epochs, BF16, AdamW, weight decay 0.1.
- Same clips, prompt plan, augmentation RNG, and optimizer-update budget.
- Consistent horizontal flip; consistent square resize to 1024; the prior
  modest consistent affine, color jitter, and grayscale `p=0.05` recipe.
- First 1,000 updates: frozen image encoder; train decoder, memory modules, and
  any new temporal modules.
- Remaining updates: unfreeze the student image encoder at low LR.
- LR groups: image encoder `1e-6 -> 1e-7`; memory/decoder/pointer
  `5e-6 -> 5e-7`; new temporal modules `5e-5 -> 5e-6`.
- 5% linear warmup followed by cosine decay.
- Gradient clipping at norm 1.0, with pre-clip norm and clip fraction logged.
  Do not reuse the previously saturated 0.1 threshold silently.
- Weight decay zero for bias and normalization parameters.
- `checkpoint_N.pt` after every epoch plus resumable `checkpoint.pt` and a
  convenience `last.pt`; `best.pt` for every O2/A--E run is selected by
  the same full SA-V val **R4 J&F** metric. R4 starts at the prompted anchor,
  uses only real GT for scoring, and uses the common naive cache/gather path.
  R1 and age-stratified metrics are diagnostics and never select Wave-1
  checkpoints.
- Resume reuses W&B run ID, TensorBoard directory, and checkpoint directory.

### Wave 2

After Wave 1, launch only the follow-ups whose screening gates pass:

1. If C exceeds B at R4, run `C-r` independently with the exact C budget and
   seed.
2. If E passes its state-distillation gate relative to D, run E+
   independently with the exact E budget and seed.

Neither C-r nor E+ initializes from a predecessor; both start from the
official SAM2.1-L checkpoint because extra parent training would confound the
mechanism comparison.

### Wave 3

Freeze the promoted E or E+ best checkpoint and train the F gate. Gate training
does not update any SAM2 parameter. Use full train videos, online targets, and
the H=2 counterfactual objective. Threshold calibration uses SA-V val only.

### Statistical confirmation after single-seed screening

Wave 1 uses one seed to screen mechanisms and allocate compute. A 0.5 J&F
screening threshold is not treated as a causal conclusion. Before a positive
pair is retained as a paper claim, rerun both sides with confirmation seeds
`250107257` and `250107258`, keeping clips, augmentations, prompts, and
evaluation shards paired within each seed.

For every retained comparison, report:

- all three per-seed deltas and their mean plus standard deviation;
- a paired, video-level bootstrap 95% confidence interval with 10,000
  resamples for each seed;
- a paired bootstrap interval over seed-averaged per-video deltas.

A final directional mechanism claim requires all three seed deltas to have the
same sign and the seed-averaged paired interval to exclude zero. Comparisons
that fail this confirmation remain exploratory even if the screening seed
crossed 0.5 J&F.

### Conditional long-horizon continuation

All checkpoints are evaluated on complete videos, regardless of T8 training.
If the selected parent loses at least 2.0 J&F between the first and later
refresh cycles on SA-V val, run one T16 continuation epoch with the image
encoder frozen and half temporal/memory/decoder LR. Report the pre/post result
as a confirmation, not as part of A--E.

Company commands remain foreground with live output and `tee`; no `nohup`,
background process, detached shell, or charged `inferno` job is permitted.

## Evaluation plan

### Fixed refresh and feature-age curves

Evaluate O0/O1/O2/A--E/E+ at R1--R6. Report:

- J, F, and J&F;
- encoder calls / frames and realized age histogram;
- J&F at feature age 0--5;
- loss/quality versus memory freshness;
- first-cycle versus later-cycle full-video quality.

SA-V age evaluation never scores against teacher pseudo-labels. For an
annotation-aligned target `k` and desired age `a`, phase-align the last actual
encoder refresh to `k-a`, then score only the real GT at `k`. This yields
GT-only SA-V age strata even though consecutive intermediate frames are not
annotated. After every architecture, checkpoint, and policy threshold is
frozen using SA-V val, DAVIS supplies one dense external diagnostic for
per-frame prediction between observations. DAVIS is run once and cannot cause
model or policy reselection.

### Temporal-scope boundary

All v1 training and evaluation use contiguous raw 24 FPS frames and ages
0--5. Stride 2/4 evaluation is removed because it would produce ages outside
`Embedding(6,64)`. A later variable-gap study must replace the discrete age
and freshness tables with a continuous elapsed-time encoding and receive a
new experiment version; clipping an age such as 8 into bucket 5 is invalid.

### Failure-mode strata

Use only GT-scored frames:

- object motion: centroid displacement normalized by object scale, split into
  dataset tertiles;
- occlusion/reappearance: GT visibility gaps and reappearance events;
- camera motion: median low-resolution DIS background-flow magnitude outside
  the GT union mask, split into tertiles;
- deformation: one minus IoU after translating the previous GT mask by its
  centroid displacement, split into tertiles.

These are deterministic diagnostics, not new training inputs. Report gate
refresh frequency and J&F in each stratum.

### Systems metrics

On one H100 with batch-one streaming inference, measure after warmup:

- mean, median, and P95 model latency;
- mean, median, and P95 end-to-end latency including frame decode and gate;
- FPS, peak HBM, and component times for encoder, memory attention, decoder,
  memory write, temporal module, and gate;
- encoder calls / total frames and spatial-memory writes / total frames.

Use actual skipped calls, not FLOP estimates. The teacher and counterfactual
oracle are excluded from deployment timing.

### Test access

Freeze checkpoint, policy, threshold, prompt protocol, and all reporting code
from validation first. Run full SA-V test only for:

1. O0 official full-refresh reference;
2. selected fixed parent at R1;
3. selected fixed-budget policy;
4. selected learned gate at its frozen operating point.

DAVIS val is diagnostic and does not alter the SA-V test policy after
selection.

## Pre-registered interpretation criteria

| Claim | Required evidence |
|---|---|
| naive reuse is a real phenomenon | O1 degrades monotonically with age relative to O0 |
| generic SA-V fine-tuning transfers to reuse | O2 exceeds O1 at matched R2--R6 inference |
| schedule adaptation helps | A exceeds O2 at matched R2--R6 inference, especially the pre-selected R4 metric |
| current feature age helps | B exceeds A on R4 and the GT-only age curve |
| combined memory time helps | C-rf exceeds B; this alone does not isolate freshness |
| memory freshness helps | after a positive C screen, C-rf exceeds matched C-r |
| stale-write suppression helps | D exceeds C, especially at ages 3--5; do not claim misalignment is proven causal |
| privileged state KD helps | E exceeds D by at least 0.5 J&F at R4 or ages 3--5 mean |
| predictive transition helps | E+ passes all three E+ promotion conditions |
| adaptive observation helps | frozen-parent learned gate beats fixed, random, and age-only by at least 0.5 J&F at one matched nontrivial budget and is non-dominated across the curve |
| deployment claim | accuracy, refresh rate, mean/P95 latency, and FPS all measured on the target H100 path |

The 0.5 J&F entries above are single-seed screening gates. Every positive
direction retained as a final mechanism claim must additionally satisfy the
three-seed and paired-bootstrap confirmation rule.

Negative outcomes remain research results. In particular, a weak oracle
rejects content-adaptive routing headroom; a strong oracle with a weak learned
gate rejects the gate predictor; E+ failing E rejects the predictive claim but
does not reject stale-aware reuse.

## Implementation boundaries and verification

Implementation adds repo-owned modules rather than editing the official SAM2
checkout. Wave-1 delivered components are:

- a 24 FPS SA-V dataset/sampler and GT-validity-aware collation path;
- a stale-aware SAM2.1-L training subclass with real refresh-only encoding,
  two-clock metadata, and safe-bank selection;
- temporal conditioning, group-normalized loss, and frozen online-teacher
  extensions (E+ remains conditional on E passing its registered gate);
- a matching video-predictor subclass for fixed/gated inference;
- a matching fixed-policy video-predictor subclass for R1--R6 inference;
- a company runner with `audit`, `smoke`, `train`, `select`, `curves`,
  `controls`, `run`, and `status` actions;
- full-val rank merging, official SA-V evaluation, actual encoder-call
  telemetry, and GT-only realized-age tables. Forced-age, failure-strata,
  E+/gate, Pareto, and deployment profiling remain post-Wave-1 work.

Minimum verification before formal launch:

1. Unit tests for deterministic trajectories, raw 24 FPS indices,
   age/freshness bounds with out-of-range rejection, group-normalized loss,
   and fixed-horizon gate-value discounting.
2. Age-zero outputs exactly match the official model before training.
3. Every zero-output temporal projection has a nonzero gradient after the
   first nontrivial backward pass; no residual has a second zero scalar gate.
4. `F_raw_cache` remains bitwise unchanged after constructing `F_read`, and
   A/B/C `MemEnc` receives the same raw cached tensor for a matched rollout.
5. Encoder invocations equal unique refresh frames; reuse does not call Hiera.
6. D/E/E+ reuse frames never execute or store spatial memory, but do emit
   object pointers and timestamps.
7. Spatial selection never exceeds `num_maskmem=7`; model-visible pointer
   selection never exceeds 16 and is identical across C/D/E/E+.
8. Teacher/student prompts match exactly and teacher parameters receive no
   gradients or optimizer/checkpoint entries.
9. E state/pointer/score losses select only reuse frames and detach targets.
10. E+ age-zero bypass and zero-initialized residual preserve parent behavior.
11. Gate counterfactual branches start from identical immutable prior state,
   always contain `k..k+2`, and execute rather than truncate forced refreshes;
   deployment runs one branch only.
12. O1/O2/A evaluated at a given Rk use identical refresh actions and cache
   gathering; all O2/A--E checkpoint selectors consume only `val/R4/J&F`.
13. Training and inference share one temporal/safe-memory policy
   implementation so evaluation cannot silently restore official per-frame
   writes.
14. Single-GPU forward/backward, four-GPU DDP, checkpoint resume, full-val
    inference, and evaluator identity smokes pass.

The design and implementation live on
`research/sam21l-two-clock-reuse-v1`. Local verification currently includes
17 deterministic core/data/loss tests, official SAM2 import, Hiera-L student
and predictor Hydra construction, and optimizer parameter-group coverage.
Company Torch 2.4 forward/backward, four-rank DDP, and resume behavior must be
recorded by the smoke before formal launch. Keep generated results out of code
commits and merge only after the smoke and first result-table update pass.

## Result ledger

### Training and fixed-policy accuracy

| Run | Seed | T | Epochs | GPUs | W&B ID | Best checkpoint | R1 J&F | R4 J&F | Age 3--5 mean | Status/decision |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---|
| O0 | n/a | full | 0 | pending | pending | official | pending | pending | pending | planned |
| O1 | n/a | full | 0 | pending | pending | official | pending | pending | pending | planned |
| O2 | 250107256 | 8 | 5 | 4xH100 | pending | pending | pending | pending | pending | planned |
| A | 250107256 | 8 | 5 | 4xH100 | pending | pending | pending | pending | pending | planned |
| B | 250107256 | 8 | 5 | 4xH100 | pending | pending | pending | pending | pending | planned |
| C | 250107256 | 8 | 5 | 4xH100 | pending | pending | pending | pending | pending | planned |
| C-r | 250107256 | 8 | 5 | 4xH100 | pending | pending | pending | pending | pending | conditional on C > B |
| D | 250107256 | 8 | 5 | 4xH100 | pending | pending | pending | pending | pending | planned |
| E | 250107256 | 8 | 5 | 4xH100 | pending | pending | pending | pending | pending | planned |
| E+ | 250107256 | 8 | 5 | 4xH100 | pending | pending | pending | pending | pending | blocked on Wave 1 |

### Adaptive-policy Pareto

| Parent | Policy | Target refresh | Actual refresh | J&F | Mean ms | P95 ms | FPS | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pending | fixed | 25% | pending | pending | pending | pending | pending | planned |
| pending | random | 25% | pending | pending | pending | pending | pending | planned |
| pending | age-only | 25% | pending | pending | pending | pending | pending | planned |
| pending | learned | 25% | pending | pending | pending | pending | pending | blocked on parent |
| pending | oracle | 25% | pending | pending | n/a | n/a | n/a | blocked on parent |

## References

- [SAM 2 paper](https://arxiv.org/html/2408.00714v2)
- [Official SAM2 implementation](https://github.com/facebookresearch/sam2)
- [DAVIS 2017](https://arxiv.org/abs/1704.00675)
- [Deep Feature Flow](https://arxiv.org/abs/1611.07715)
- [MoSAM](https://arxiv.org/abs/2505.00739)
- [Efficient-SAM2](https://arxiv.org/abs/2602.08224)
- [Lean-SAM2](https://arxiv.org/abs/2607.19811)
- [StreamDAM](https://arxiv.org/abs/2608.03912)
