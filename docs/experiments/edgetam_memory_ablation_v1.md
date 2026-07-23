# EdgeTAM memory ablation v1

## Question

After the TinyViT-21M hybrid has been trained end to end with SA-V mask GT, does EdgeTAM's compressed spatial memory improve video segmentation, and is any change caused by compression or by the official EdgeTAM initialization?

The starting point for every row is `A02_e2e_t4_official_prompt`. The image encoder and mask decoder are frozen. Training updates the memory attention, memory encoder, object-pointer memory parameters, and, when present, the spatial Perceiver.

## Controlled experiment table

| Run | Memory tokens | Attention | Initialization | Question | Lane | Status |
|---|---:|---:|---|---|---|---|
| `M0_sam2_mem4` | uncompressed 64x64 | 4 layers | current A02 | Continued-training control | `memory1` | pending |
| `M1_sam2_mem2` | uncompressed 64x64 | 2 layers | first two A02 layers | Is depth reduction alone responsible? | `memory2` | pending |
| `M2a_edgetam_hybrid2_official` | 256 global + 256 2D | 2 layers | official Perceiver + official attention pair | Full EdgeTAM memory transfer | `memory1` | pending |
| `M2b_edgetam_hybrid2_current` | 256 global + 256 2D | 2 layers | official Perceiver + first two A02 attention layers | Separate compression from attention initialization | `memory2` | pending |

This is a 2x2 causal sequence rather than a broad hyperparameter sweep: compare M0 vs M1 for depth, M1 vs M2b for compression, and M2b vs M2a for initialization.

## Shared protocol

- Data: all usable SA-V train videos (currently 50,337), four sampled frames per example, full SA-V val and test.
- Hardware: four H100s per run; batch 2/GPU, global batch 8; one epoch.
- Prompt: exact first-frame box; no correction click. This matches the deployed box-prompt evaluation path.
- Frozen: TinyViT-21M encoder, neck, prompt encoder, and mask decoder. BatchNorm is frozen.
- Trainable: memory attention, memory encoder, object-pointer memory parameters, and the Perceiver for M2a/M2b.
- Learning rates: Perceiver `1e-5 -> 1e-6`, memory attention `3e-6 -> 3e-7`, other memory parameters `1e-6 -> 1e-7`; 5% linear warmup, cosine decay, gradient clipping at 0.1.
- Tracking: online W&B project `edgetam-memory-ablation-v1`, TensorBoard fallback, raw and EMA losses, LR groups, normalized loss, object-frame count, and samples whose total loss is at least 20.
- Pipeline: train -> full val -> select best -> full test -> append central CSV. Training resumes the same checkpoint, TensorBoard directory, and W&B run ID.
- Storage: each run keeps one physical `last.pt`; `best.pt` and the compatibility `checkpoint.pt` are symlinks. Evaluation predictions are deleted after scoring. Four runs add well below 10 GB, excluding the shared SA-V data and starting checkpoints.

Run root:

`/group-volume/danny-dataset/sam2_distill/runs/edgetam_memory_ablation_v1`

Central result table:

`/group-volume/danny-dataset/sam2_distill/runs/edgetam_memory_ablation_v1/summary.csv`

## Decision rule

Primary selection uses SA-V val J&F. Test is reported only after the val result exists. Also report image mIoU/AP and both image/video latency so a J&F gain is not accepted blindly if the compressed module regresses prompt segmentation or speed.

Treat changes smaller than 0.3 J&F as inconclusive without a repeat seed. Keep EdgeTAM memory only if M2a or M2b improves val J&F over both M0 and M1 without a material image-quality regression. If M2b beats M2a, preserve A02 attention initialization; if M2a wins, the official attention pair is part of the useful transfer.

## Implementation

- Single-run driver: `scripts/company/49_run_edgetam_memory_ablation.sh`
- Two-node lanes: `scripts/company/50_run_edgetam_memory_lane.sh`
- Training topology and initialization: `tools/train/run_sam2_task_training.py` and `sam2_distill/models/task_finetune.py`
- Full SA-V evaluation: `scripts/company/25_benchmark_stage1_sav_test.sh`

The driver validates the EdgeTAM checkout, serializes official-checkpoint download, audits SA-V paths, uses per-variant pipeline locks, performs strict checkpoint loading, and records incomplete or failed stages in the same summary table.
