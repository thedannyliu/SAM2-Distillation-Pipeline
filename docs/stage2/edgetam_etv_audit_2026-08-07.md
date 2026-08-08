# ETV1/ETV2 EdgeTAM Audit (2026-08-07)

## Question

Do the ETV1/ETV2 loss spikes indicate unstable optimization, and did ETV2
actually instantiate and train the released EdgeTAM temporal stack on the full
SA-V training release?

The audit separates four possible causes:

1. unstable gradients;
2. an incomplete dataset or unfrozen BatchNorm;
3. wrong EdgeTAM topology/checkpoint loading;
4. a semantically incompatible student/teacher interface or loss target.

## Immediate findings

| Check | Evidence | Result |
|---|---|---|
| Loss spikes imply failure | Functional Q0 official EdgeTAM reaches 68.3/69.5 val/test J&F despite a maximum raw loss/frame of 117.48. ETV2's maximum is 31.35. | Rejected |
| Gradient clipping absent | Resolved base config uses L2 clipping with `max_norm=0.1`; upstream trainer clips after backward and before the optimizer step. | Rejected |
| BatchNorm updated | W&B records `freeze_batchnorm=true`; `EdgeTAMTrain.train()` reapplies eval/frozen state to every BatchNorm and every wholly frozen module. | Rejected |
| Small training subset | ETV1/ETV2 logs report 50,337 training videos, excluding only the 116 records without readable manual JSON. Optimizer steps exactly match full-data batch arithmetic. | Rejected |
| Wrong temporal tensor count | Released `edgetam.pt` has 148 selected temporal tensors / 4,775,392 parameters. ETV2 reports exactly 148 trainable tensors / 4,775,392 trainable parameters. | Rejected |
| Wrong temporal graph | The local temporal model matches released EdgeTAM: two memory-attention layers, `RoPEAttentionv2`, 256 global + 256 2D latents, 64-channel memory, and official pointer/layout flags. | Rejected |
| Raw W&B curve is a global loss | Direct W&B logging records rank-0 current meters, not a four-rank average. | False; monitoring limitation |
| Hiera-L `F_M` is interface-compatible with frozen TV21 decoder | Shapes match, but representation compatibility is not established. The loss also includes conditioning frames where memory attention is bypassed. | High-risk hypothesis |
| Teacher and student receive identical prompts | Their complete forwards independently sample PyTorch point/box coordinates and correction clicks. NumPy route choices start synchronized, but exact coordinates are not shared. | Confirmed code risk |
| Reverse-time augmentation is enabled | The source and resolved sampler set `reverse_time_prob: 0.5`. The artifact audit now checks this value explicitly. | Pass; no correction required |

## W&B loss audit

### A functional counterexample

Q0 uses the exact released RepViT-M1 EdgeTAM graph and checkpoint. It remains
functional after one local temporal epoch:

| Run | val/test J&F | median loss/frame | p95 | maximum |
|---|---:|---:|---:|---:|
| Q0 official identity | 68.3 / 69.5 | 0.183 | 3.737 | **117.475** |
| ETV2 T4 | 47.3 / pending | 0.655 | 5.735 | 31.349 |

Raw loss spikes therefore cannot diagnose tracking failure by themselves.
Q0 is a direct counterexample under the same trainer family and SA-V task
loss.

### ETV2 spike source

Across the 140 logged ETV2 rows, correlation with total loss/frame is:

| Component | Correlation |
|---|---:|
| focal/mask loss | **0.997** |
| class/presence loss | 0.395 |
| Dice loss | 0.047 |
| IoU loss | -0.076 |
| present object frames | -0.021 |

The mask focal term has weight 20. The largest events are isolated hard-mask
events, not a persistent rise in Dice or IoU:

| Step | total/frame | focal/mask | logit KD | Dice | IoU |
|---:|---:|---:|---:|---:|---:|
| 2041 | 31.349 | 5.838 | 6.808 | 0.400 | 0.119 |
| 3571 | 23.532 | 4.425 | 4.208 | 0.516 | 0.236 |
| 601 | 18.027 | 3.338 | 3.085 | 0.568 | 0.257 |

The next logged rows return close to the previous range. Gradient clipping
cannot remove these forward-loss events; it only limits the parameter update
that follows them. The missing diagnostic is pre-clip norm and clip rate, not
a lower clip threshold.

W&B currently logs rank-0 current meters. ETV2 therefore displays one local
batch of six clips rather than a synchronized global batch of 24. ETV1 has
only 14 W&B samples over 393 optimizer updates, so its apparent
`1 -> 0.8 -> 1.9` path is not a dense optimization trajectory.

## Data and exposure audit

| Stage | Videos/pass | Per-GPU batch | Global batch | T | Passes | Updates |
|---|---:|---:|---:|---:|---:|---:|
| ETV1 | 50,337 | 32 | 128 | 1 | 1 | 393 |
| ETV2 | 50,337 | 6 | 24 | 4 | 2 | 4,194 |

This is the full available SA-V train release, but it is not paper-scale
exposure. ETV2 sees approximately `4194 * 24 * 4 = 402,624` sampled frame
slots. A 130K-update, global-256, T8 stage would expose about 266 million
frame slots before accounting for its broader dataset mixture. ETV2 is about
0.15% of that exposure. Full dataset cardinality does not imply full temporal
coverage because the sampler draws one short random clip per video per pass.

## Released module and checkpoint audit

The pinned EdgeTAM source commit is
`7711e012a30a2402c4eaab637bdb00a521302c91`. The locally verified released
checkpoint has SHA-256:

`ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df`

Its ETV `official_temporal` selection contains:

| Prefix/parameter | Tensors |
|---|---:|
| `memory_attention.*` | 54 |
| `memory_encoder.*` | 40 |
| `spatial_perceiver.*` | 44 |
| `obj_ptr_proj.*` | 6 |
| temporal/no-memory parameters | 4 |
| **Total** | **148** |

These tensors contain 4,775,392 parameters, exactly matching ETV2's
trainable model summary. The initializer checks every source tensor and shape
and calls `load_state_dict(..., strict=True)`. ETV2 therefore did train the
released temporal stack; it did not silently train a random substitute.

This does not make ETV2 an unmodified EdgeTAM model. Its actual graph is:

```text
TV21 encoder + TV21-aligned prompt/mask decoder
  -> released EdgeTAM memory encoder + 2D Perceiver + two attention blocks
  -> frozen TV21-aligned mask decoder
```

Q0 proves that the exact official graph, loader, trainer, and evaluator can
retain EdgeTAM behavior. ETV2 instead tests a cross-interface hybrid.

## Module-by-module connection review

| Boundary | Mechanical contract | Semantic contract | Finding |
|---|---|---|---|
| TinyViT -> FPN | 32/64/256 channels at stride 4/8/16; final feature is 256 x 64 x 64 | Features must remain compatible with the retained decoder and temporal stack | Shape/load contract passes; semantic alignment is only established for the retained TV21 decoder |
| Predicted mask -> memory encoder | High-resolution logits and 256-channel image feature enter official memory encoder | Written memory must preserve object identity and mask confidence | Official path is used; no direct memory-token or poison audit exists |
| Memory encoder -> Perceiver | 64-channel map becomes 256 global + 256 2D tokens with matching positional encodings | Compressed tokens must preserve information needed by the reader | Exact official implementation and weights pass |
| Perceiver memory -> memory attention | 512 tokens per stored frame; RoPE repeats per spatial-memory frame; pointers excluded from spatial RoPE | Pointer and spatial-token semantics must remain coherent | Exact official layout passes mechanically |
| Memory attention -> TV21 decoder | Output is reshaped to 256 x 64 x 64 | Output distribution must match the frozen TV21 mask decoder | **Not established; primary interface risk** |
| Student/teacher prompts | Same probabilities and frame-count controls | Distillation requires comparable conditioning inputs | Route choices use separate model RNGs; PyTorch point/box coordinates and correction trajectories are not shared |
| Hiera-L teacher `F_M` -> TV21 student `F_M` | Tensor shapes match | The Hiera-L representation must be a valid target for a frozen TV21 decoder | **Not established and potentially conflicting** |
| KD scaling across frames | Feature/logit terms are averaged across frames | Their weight relative to task loss should remain stable with T | Task loss sums frames while KD averages frames; effective KD strength decreases as T grows |
| Conditioning-frame memory KD | Both expose `F_M` | Target should train the compressed read path | Initial conditioning frames bypass memory attention; their `F_M` loss mainly trains `no_mem_embed` |
| BatchNorm | All BN modules are repeatedly forced to eval/frozen | Running statistics must not drift | Pass |
| Reverse-time sampling | Sampler supports `reverse_time_prob` | Recipe calls for bidirectional clip sampling | Enabled at `0.5`; the earlier missing-value diagnosis was incorrect |

## Highest-value fast audits

The next actions should answer specific questions before another full
curriculum:

| Priority | Audit | Cost | Decision |
|---:|---|---:|---|
| 1 | Parse ETV2 `initialization_summary.json`, resolved config, and `loss_outliers.jsonl` | Seconds, CPU | Confirm run-level provenance and identify whether spikes concentrate on tiny masks or specific videos |
| 2 | Compare company checkpoint SHA with the verified released SHA | Seconds, CPU | Reject a corrupted/wrong checkpoint immediately |
| 3 | Log pre-clip gradient norm and clipping fraction for a 50--100-step smoke | Minutes | Only adjust LR/clip if norms are non-finite or almost every step is clipped |
| 4 | T4 100--500-step exact-mask-prompt diagnostic with no correction clicks | Short GPU smoke | If KD spikes disappear, prompt-trajectory mismatch is material |
| 5 | Split memory/logit KD into initial-conditioning versus propagated frames | Short GPU smoke | Determine whether the current target mostly supervises the bypass path |
| 6 | Evaluate J&F by propagation age on the existing ETV2 checkpoint | Existing predictions/short eval | Frame-1 failure implies interface failure; progressive decay implies recurrent-state failure |

Do not lower clip below 0.1 or launch ETV3 until priorities 1--3 are read.
Do not interpret a smooth loss curve as success; the promotion metric remains
full SA-V validation J&F and its temporal-age breakdown.

## Company artifact command

Run from the company repository after pulling the audit tool:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
AUDIT_LOG_ROOT=/user-volume/log/edgetam_tv21_sam21l_v1/audit_20260807
ETV2_DIR=/group-volume/danny-dataset/sam2_distill/runs/edgetam_tv21_sam21l_v1/formal/ETV2_t4_bootstrap_2ep/main
EDGETAM_CKPT=/group-volume/danny-dataset/sam2_distill/checkpoints/edgetam/edgetam.pt
mkdir -p "${AUDIT_LOG_ROOT}"
python tools/edgetam/audit_etv_run.py \
  --run-dir "${ETV2_DIR}" \
  --edgetam-checkpoint "${EDGETAM_CKPT}" \
  --expected-sha256 ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df \
  --out-json "${AUDIT_LOG_ROOT}/etv2_artifact_audit.json" \
  2>&1 | tee "${AUDIT_LOG_ROOT}/etv2_artifact_audit.log"
AUDIT_STATUS=${PIPESTATUS[0]}
echo "ETV2 artifact audit status: ${AUDIT_STATUS}"
echo "EdgeTAM commit: $(git -C /user-volume/repo/EdgeTAM rev-parse HEAD)"
echo "SAM2 training commit: $(git -C /user-volume/repo/facebookresearch-sam2 rev-parse HEAD)"
```

The audit output is a small foreground log under `/user-volume/log`; it does
not duplicate checkpoints or modify the existing run.
