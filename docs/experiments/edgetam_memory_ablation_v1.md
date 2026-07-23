# EdgeTAM memory ablation v1

## Question

After the TinyViT-21M hybrid has been trained end to end with SA-V mask GT, does EdgeTAM's compressed spatial memory improve video segmentation, and is any change caused by compression or by the official EdgeTAM initialization?

The starting point for every row is `A02_e2e_t4_official_prompt`. The M rows freeze the image encoder and mask decoder to isolate memory architecture. The R rows then reproduce the EdgeTAM video-training signals while keeping the TinyViT-21M encoder.

## Controlled experiment table

| Run | Memory tokens | Attention | Initialization | Question | Lane | Status |
|---|---:|---:|---|---|---|---|
| `M0_sam2_mem4` | uncompressed 64x64 | 4 layers | current A02 | Continued-training control | `memory1` | complete |
| `M1_sam2_mem2` | uncompressed 64x64 | 2 layers | first two A02 layers | Is depth reduction alone responsible? | `memory2` | complete |
| `M2a_edgetam_hybrid2_official` | 256 global + 256 2D | 2 layers | official Perceiver + official attention pair | Full EdgeTAM memory transfer | `memory1` | complete |
| `M2b_edgetam_hybrid2_current` | 256 global + 256 2D | 2 layers | official Perceiver + first two A02 attention layers | Separate compression from attention initialization | `memory2` | complete |

This is a 2x2 causal sequence rather than a broad hyperparameter sweep: compare M0 vs M1 for depth, M1 vs M2b for compression, and M2b vs M2a for initialization.

## TinyViT-21M reproduction ladder

| Run | Frames | Trainable scope | Image KD | Memory KD | Question | Lane | Status |
|---|---:|---|---:|---:|---|---|---|
| `R0_edgetam_e2e_t4_task` | 4 | full student except prompt encoder | 0 | 0 | Does full task tuning unlock the transferred memory? | `memory1` | complete |
| `R1_edgetam_e2e_t4_imgkd` | 4 | full student except prompt encoder | 1 | 0 | What is gained by EdgeTAM image-feature distillation? | `memory2` | complete |
| `R2_edgetam_e2e_t4_imgmemkd` | 4 | full student except prompt encoder | 1 | 1 | What is additionally gained by memory-output distillation? | `memory2` | complete |
| `R3_edgetam_e2e_t8_imgmemkd` | 8 | full student except prompt encoder | 1 | 1 | Does the official eight-frame horizon matter? | `memory1` | complete |

The official EdgeTAM video stage uses task loss plus unit-weight image and memory MSE, eight frames, two memory-attention blocks, and 256 global plus 256 2D latents. It also uses a much larger mixed dataset and 130K-iteration schedule. These R rows reproduce the model, prompt, and loss method with the available SA-V data; they do not claim to reproduce the original compute or dataset mixture. The frozen online teacher is SAM2.1 Hiera-L and consumes the same frames and prompt simulation as the TinyViT student.

## Shared protocol

- Data: all usable SA-V train videos (currently 50,337), full SA-V val and test. R3 uses the audited eight-frame-eligible subset.
- Hardware: four H100s per run; one epoch. M rows use batch 2/GPU; R rows use batch 1/GPU because the full student and online teacher are resident together.
- M-row prompt: exact first-frame box with no correction click, matching the deployed box-prompt evaluation path.
- R-row prompt: point probability 0.5, conditional box probability 0.5, GT-click probability 0.1, two randomly chosen correction frames, seven iterative correction points, and up to three objects.
- M-row scope: TinyViT-21M encoder, neck, prompt encoder, and mask decoder frozen; train memory attention, memory encoder, object-pointer memory parameters, and the Perceiver for M2a/M2b.
- R-row scope: train the full student except the prompt encoder; the teacher is eval-only and excluded from DDP, optimizer state, and checkpoints. BatchNorm remains frozen.
- Learning rates: Perceiver `1e-5 -> 1e-6`, memory attention `3e-6 -> 3e-7`, other memory parameters `1e-6 -> 1e-7`; 5% linear warmup, cosine decay, gradient clipping at 0.1.
- R-row encoder learning rate: `3e-7 -> 3e-8` with 10% warmup. Image/memory MSE is computed in fp32.
- Tracking: online W&B project `edgetam-memory-ablation-v1`, TensorBoard fallback, raw and EMA losses, LR groups, normalized loss, object-frame count, and samples whose total loss is at least 20.
- Pipeline: train -> full val -> select best -> full test -> append central CSV. Training resumes the same checkpoint, TensorBoard directory, and W&B run ID.
- Storage: each run keeps one physical `last.pt`; `best.pt` and the compatibility `checkpoint.pt` are symlinks. The teacher is never checkpointed and evaluation predictions are deleted after scoring. All eight runs add well below 10 GB, excluding shared data and starting checkpoints.

Run root:

`/group-volume/danny-dataset/sam2_distill/runs/edgetam_memory_ablation_v1`

Central result table:

`/group-volume/danny-dataset/sam2_distill/runs/edgetam_memory_ablation_v1/summary.csv`

## Results

All eight pipelines completed train -> full SA-V val -> full SA-V test. The
2026-07-23 all-experiment report is the source for this table.

| Run | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `M0_sam2_mem4` | 0.8405 | 0.7167 | **71.5** | 0.8391 | 0.7191 | **74.3** |
| `M1_sam2_mem2` | **0.8406** | 0.7167 | 53.3 | 0.8391 | **0.7197** | 56.1 |
| `M2a_edgetam_hybrid2_official` | 0.8405 | 0.7166 | 15.6 | 0.8391 | 0.7190 | 12.8 |
| `M2b_edgetam_hybrid2_current` | **0.8406** | 0.7167 | 13.2 | 0.8391 | 0.7191 | 10.6 |
| `R0_edgetam_e2e_t4_task` | 0.8369 | 0.7096 | 23.0 | 0.8364 | 0.7137 | 21.5 |
| `R1_edgetam_e2e_t4_imgkd` | **0.8379** | **0.7121** | 23.6 | 0.8373 | 0.7165 | 21.7 |
| `R2_edgetam_e2e_t4_imgmemkd` | 0.8377 | 0.7117 | **25.3** | **0.8374** | **0.7167** | **23.2** |
| `R3_edgetam_e2e_t8_imgmemkd` | 0.8374 | 0.7114 | 21.9 | 0.8367 | 0.7157 | 19.1 |

Latency is retained as descriptive evidence because the runs were evaluated
under different machine loads.

| Run | val image s | val video s/video | test image s | test video s/video |
| --- | ---: | ---: | ---: | ---: |
| `M0_sam2_mem4` | 0.1178 | 38.1633 | 0.1020 | 38.8936 |
| `M1_sam2_mem2` | 0.1113 | 37.2029 | 0.0873 | 38.8279 |
| `M2a_edgetam_hybrid2_official` | 0.0867 | 36.8407 | 0.0817 | 38.4183 |
| `M2b_edgetam_hybrid2_current` | 0.0846 | 38.8273 | 0.0646 | 40.4279 |
| `R0_edgetam_e2e_t4_task` | 0.0581 | 33.3275 | 0.0581 | 40.1494 |
| `R1_edgetam_e2e_t4_imgkd` | 0.0698 | 34.2270 | 0.0562 | 35.0037 |
| `R2_edgetam_e2e_t4_imgmemkd` | 0.0623 | 30.0210 | 0.0610 | 31.8370 |
| `R3_edgetam_e2e_t8_imgmemkd` | 0.0483 | 28.9706 | 0.0457 | 30.2030 |

## Interpretation and decision

- M0 is the only viable model from this suite. Relative to A02, it gains
  0.0031 val mIoU and 0.0038 val AP, loses 0.5 val J&F, and gains 0.2 test
  J&F. Keep its checkpoint as an uncompressed continued-memory baseline, not
  as evidence for EdgeTAM compression.
- Reducing the four-layer SAM2 memory stack to its first two layers is not a
  functional warm start: M1 loses 18.2 J&F on both val and test versus M0.
- Both compressed M2 variants preserve image mIoU/AP but collapse tracking.
  This localizes the failure to the temporal memory path rather than the
  TinyViT image encoder or mask decoder. Official attention initialization is
  only 2.4 val and 2.2 test J&F better than retaining the A02 attention
  layers; neither is usable.
- Full end-to-end task tuning does not repair the transferred module. Image KD
  adds 0.6 val and 0.2 test J&F over R0. Adding memory KD adds another 1.7 val
  and 1.5 test J&F, so the signal is directionally useful but far too small.
  Increasing the horizon from four to eight frames then loses 3.4 val and 4.1
  test J&F.

Do not extend the current M2/R starting point with a broad learning-rate,
frame-count, or KD-weight sweep. Before another full SA-V run:

1. Audit a fixed 20-50-video subset without training. Record first-frame
   quality, per-frame J/F decay, memory feature and object-pointer norms, and
   attention-output statistics for M0, M1, and M2.
2. Test a coherent temporal-stack initialization instead of mixing modules:
   transfer the official memory encoder, memory attention, Perceiver, and
   associated temporal/object-pointer parameters together.
3. Distill the compressed two-layer memory path from functional M0 outputs
   while the image encoder and mask decoder are frozen. Require the mini-val
   gate to recover at least 60 J&F before full SA-V training.
4. Only after the gate passes, run staged task tuning: memory alignment first,
   then task plus image/memory KD. A learned four-to-two-layer distillation is
   preferable to truncating the first two layers.

These controls are implemented in
`docs/experiments/edgetam_memory_recovery_v2.md`.

## Decision rule

Primary selection uses SA-V val J&F. Test is reported only after the val result exists. Also report image mIoU/AP and both image/video latency so a J&F gain is not accepted blindly if the compressed module regresses prompt segmentation or speed.

Treat changes smaller than 0.3 J&F as inconclusive without a repeat seed. Keep EdgeTAM memory only if M2a or M2b improves val J&F over both M0 and M1 without a material image-quality regression. If M2b beats M2a, preserve A02 attention initialization; if M2a wins, the official attention pair is part of the useful transfer.

For the reproduction ladder, compare R1-R0 for image KD, R2-R1 for memory KD, and R3-R2 for the longer horizon. R2 is the primary compact reproduction candidate. R3 is retained only if its additional training cost produces a clear video gain without degrading image-prompt metrics.

## Implementation

- Single-run driver: `scripts/company/49_run_edgetam_memory_ablation.sh`
- Two-node lanes: `scripts/company/50_run_edgetam_memory_lane.sh`
- Training topology and initialization: `tools/train/run_sam2_task_training.py` and `sam2_distill/models/task_finetune.py`
- Full SA-V evaluation: `scripts/company/25_benchmark_stage1_sav_test.sh`

The driver validates the EdgeTAM checkout, serializes official-checkpoint download, audits SA-V paths, uses per-variant pipeline locks, performs strict checkpoint loading, and records incomplete or failed stages in the same summary table.
