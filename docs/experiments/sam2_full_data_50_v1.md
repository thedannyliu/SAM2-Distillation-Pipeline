# SAM2 full-data 50-run suite v1

## Objective

Use the complete 6 fps SA-V train manifest to find a better speed/quality
Pareto frontier for SAM2 multiplex tracking. The suite runs 50 long trainings
on ten independent 4×H100 nodes. Each node owns five sequential variants.

This is not a one-epoch screen. Every run trains for 5–8 epochs, evaluates on
the same fixed 64-video SA-V val gate, and measures N=1/2/4/8 propagation
latency with three repetitions. The seed is fixed so each table isolates one
research question.

Primary code paths:

- Matrix and hypotheses: `tools/experiments/sam2_full_data_50.py`
- Ten-node runner: `scripts/company/67_run_sam2_full_data_50.sh`
- Training/evaluation pipeline: `scripts/company/49_run_edgetam_memory_ablation.sh`
- Learned slots/shared K/V: `sam2_distill/models/sam2_object_slots.py`
- Full SA-V dataset adapter: `sam2_distill/data/sav_task_dataset.py`
- Result aggregation: `tools/benchmark/summarize_sam2_multiplex_screen.py`

## Data and controls

- Manifest: `/group-volume/danny-dataset/sam2_distill/manifests/sav_train_6fps_full.parquet`
- All-video lanes: nodes 1, 2, 10, and the FD11 data control.
- Multiplex lanes compare deterministic all-video, dense-4, and dense-8
  cohorts from the full manifest. Dense cohorts are repeated to 50,337 samples
  per epoch so diversity/density comparisons keep the update budget fixed.
- Quality reference: MX5 on the identical fixed gate cohort.
- Runtime reference: persistent bucket-4 TV21 result.
- Training batch: one video per GPU, global batch four.
- Checkpoints, W&B IDs, logs, locks, and result files are isolated by variant.

The full dataset adds temporal windows, not new video identities: it is most
likely to help T8/T12/T16 memory learning, correction curricula, and identity
preservation. It should not be interpreted as a pure image-diversity scaling
experiment.

## Node 1 — Does full data raise the TV21 accuracy ceiling?

| Variant | Main change | Epochs |
|---|---|---:|
| `FD01_tv21_t4_decmem_5ep` | T4 decoder+memory control | 5 |
| `FD02_tv21_t8_decmem_5ep` | T8 temporal context | 5 |
| `FD03_tv21_t8_logits1_8ep` | T8 plus mask-logit KD | 8 |
| `FD04_tv21_t8_joint_logits1_8ep` | Low-LR encoder joint tuning | 8 |
| `FD05_tv21_t12_joint_mem025_logits2_8ep` | T12 plus memory/logit KD | 8 |

## Node 2 — Can TV11 cross 95% TV21 quality at lower latency?

| Variant | Main change | Epochs |
|---|---|---:|
| `FD06_tv11_t4_decmem_5ep` | TV11 T4 control | 5 |
| `FD07_tv11_t8_decmem_5ep` | TV11 T8 | 5 |
| `FD08_tv11_t8_logits1_8ep` | TV21 mask-logit KD | 8 |
| `FD09_tv11_t8_joint_logits1_8ep` | Very-low-LR encoder joint tuning | 8 |
| `FD10_tv11_t12_joint_mem025_logits2_8ep` | T12 behavior KD | 8 |

## Node 3 — How should the new data trade diversity for object density?

| Variant | Cohort | Frames | Epochs |
|---|---|---:|---:|
| `FD11_sharedkv_all_t8_r16_8ep` | All train videos | 8 | 8 |
| `FD12_sharedkv_dense4_t8_r16_8ep` | Dense-4 | 8 | 8 |
| `FD13_sharedkv_dense8_t8_r16_8ep` | Dense-8 | 8 | 8 |
| `FD14_sharedkv_dense4_t12_r16_8ep` | Dense-4 | 12 | 8 |
| `FD15_sharedkv_dense8_t12_r16_8ep` | Dense-8 | 12 | 8 |

## Node 4 — What low-rank residual capacity restores object identity?

| Variant | Spatial rank | Pointer rank | Epochs |
|---|---:|---:|---:|
| `FD16_sharedkv_r2_ptr8_8ep` | 2 | 8 | 8 |
| `FD17_sharedkv_r4_ptr8_8ep` | 4 | 8 | 8 |
| `FD18_sharedkv_r8_ptr8_8ep` | 8 | 8 | 8 |
| `FD19_sharedkv_r16_ptr8_8ep` | 16 | 8 | 8 |
| `FD20_sharedkv_r32_ptr8_8ep` | 32 | 8 | 8 |

## Node 5 — Which memory/logit KD balance preserves J&F?

| Variant | Memory KD | Logit KD | Epochs |
|---|---:|---:|---:|
| `FD21_sharedkv_r16_mem025_logits2_8ep` | 0.25 | 2 | 8 |
| `FD22_sharedkv_r16_mem050_logits2_8ep` | 0.5 | 2 | 8 |
| `FD23_sharedkv_r16_mem100_logits2_8ep` | 1 | 2 | 8 |
| `FD24_sharedkv_r16_mem200_logits2_8ep` | 2 | 2 | 8 |
| `FD25_sharedkv_r16_mem100_logits4_8ep` | 1 | 4 | 8 |

## Node 6 — How should pointer identity be pooled over time?

| Variant | Pointer path | Temporal pool | Epochs |
|---|---|---|---:|
| `FD26_sharedkv_r16_ptr0_mean_8ep` | Disabled | Mean | 8 |
| `FD27_sharedkv_r16_ptr4_mean_obj010_8ep` | Rank 4, KD 0.1 | Mean | 8 |
| `FD28_sharedkv_r16_ptr8_mean_obj025_8ep` | Rank 8, KD 0.25 | Mean | 8 |
| `FD29_sharedkv_r16_ptr8_latest_obj025_8ep` | Rank 8, KD 0.25 | Latest | 8 |
| `FD30_sharedkv_r16_ptr8_recency050_obj025_8ep` | Rank 8, KD 0.25 | Recency 0.5 | 8 |

## Node 7 — What training horizon gives the best long-video identity?

| Variant | Frames | Epochs |
|---|---:|---:|
| `FD31_sharedkv_t4_r16_ptr8_8ep` | 4 | 8 |
| `FD32_sharedkv_t6_r16_ptr8_8ep` | 6 | 8 |
| `FD33_sharedkv_t8_r16_ptr8_8ep` | 8 | 8 |
| `FD34_sharedkv_t12_r16_ptr8_8ep` | 12 | 8 |
| `FD35_sharedkv_t16_r16_ptr8_8ep` | 16 | 8 |

## Node 8 — Which prompt/correction curriculum trains robust multiplexing?

| Variant | Main change | Epochs |
|---|---|---:|
| `FD36_sharedkv_pointheavy_8ep` | Point-heavy prompts | 8 |
| `FD37_sharedkv_mixedprompt_8ep` | Balanced control | 8 |
| `FD38_sharedkv_boxheavy_8ep` | Box-heavy prompts | 8 |
| `FD39_sharedkv_correct2x3_8ep` | Two corrected frames, three clicks | 8 |
| `FD40_sharedkv_correct3x7_8ep` | Three corrected frames, seven clicks | 8 |

## Node 9 — Where should deployment switch into the shared route?

| Variant | Slots | Shared-route threshold | Epochs |
|---|---:|---:|---:|
| `FD41_sharedkv_slot4_min4_r16_8ep` | 4 | N=4 | 8 |
| `FD42_sharedkv_slot6_min4_r16_8ep` | 6 | N=4 | 8 |
| `FD43_sharedkv_slot8_min2_r16_8ep` | 8 | N=2 | 8 |
| `FD44_sharedkv_slot8_min3_r16_8ep` | 8 | N=3 | 8 |
| `FD45_sharedkv_slot8_min4_gt025_r16_8ep` | 8 | N=4, GT sampling 0.25 | 8 |

## Node 10 — Can reduced-depth or EdgeTAM memory train coherently at scale?

| Variant | Memory path | Trainable scope | Epochs |
|---|---|---|---:|
| `FD46_mem4_t8_decmem_8ep` | Standard four-layer control | Decoder+memory | 8 |
| `FD47_mem2_t8_decmem_8ep` | Standard two-layer | Decoder+memory | 8 |
| `FD48_mem2_t8_joint_logits2_8ep` | Standard two-layer plus KD | Encoder+decoder+memory | 8 |
| `FD49_edgetam2_temporal_logits2_8ep` | EdgeTAM Spatial Perceiver + two layers | Temporal modules | 8 |
| `FD50_edgetam2_joint_img_logits2_8ep` | EdgeTAM plus image/logit KD | Encoder+temporal modules | 8 |

## Decision metrics

Do not rank by FPS alone. The primary view is a Pareto plot/table over:

1. Fixed-cohort J&F retention relative to MX5.
2. N=8 propagation FPS and milliseconds per frame.
3. N=1 retention, to expose low-object overhead.
4. Bucket verification mask IoU for learned routes.
5. Peak GPU memory.

Promote full evaluations only when J&F retention is at least 95%, learned-route
mask IoU is at least 0.95, and N=8 is faster than the persistent bucket
reference. Accuracy anchors and legacy memory variants remain useful controls
even though they cannot pass the learned-route gate.

## QAT boundary

Quantization-aware training is intentionally not mislabeled as one of these 50
runs. The current repository has no Thor/TensorRT-aligned fake-quant operator
policy, scale granularity, or export validator. First select the FP Pareto
models here; then define INT8/FP8 operator coverage and calibration with the
deployment engineer, add numerical-equivalence tests, and run a separate QAT
matrix against the actual Thor engine.
