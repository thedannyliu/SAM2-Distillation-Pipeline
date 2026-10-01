# Training and evaluation

Run commands from the repository root. Set paths for your environment rather
than relying on the company defaults embedded in historical recipes.

## Environment and data

The company image is `ngc24.06/ub22/py3.10/cu12.5/cudnn9.1/pytorch2.4`.
Use its Python directly. Setup installs SAM2 with `--no-deps` and verifies
compatibility; do not solve a failed import by silently replacing torch.
A local SAM2 checkout must include upstream training code, not only inference.

```bash
export SAM2_UPSTREAM=/path/to/sam2
export SAM2_TRAINING_ROOT="$SAM2_UPSTREAM"
export SAM2D_ROOT=/path/to/sam2_distill
export SAV_ROOT=/path/to/SA-V
bash scripts/core/setup_env.sh
bash scripts/core/download_weights.sh --out "$SAM2D_ROOT/checkpoints"
```

Stage 1's online-teacher recipe uses SA-1B; Stage 2 uses SA-V. Provisioning
one does not provision the other. Keep datasets read-only and place caches,
runs and checkpoints on the designated data volume. Company runs use
`/group-volume/danny-dataset`; small terminal logs use `/user-volume/log`.

Prepare SA-V using the [data runbook](stage2/sav_full_memory_train_data.md).
Before training, run:

```bash
bash scripts/core/data_prepare_sav.sh audit
```

The full-data contract is 50,453 MP4 records, of which 50,337 have usable
matching manual annotations, sampled at the 6 FPS annotation cadence.
Do not substitute a small repeated cohort and describe it as full-data training.

## Stage 1: image encoder

```bash
GPUS=0,1,2,3 bash scripts/core/stage1_distill_encoder.sh download
GPUS=0,1,2,3 bash scripts/core/stage1_distill_encoder.sh train
```

The frozen online teacher supervises `image_embed`, `high_res_s0`, and
`high_res_s1`. The cached-teacher alternative is
`tools/train/train_stage1.py`; see [Stage 1 protocols](stage1/).

## Stage 2: task fine-tuning

For a new run, use the shared driver with explicit inputs:

```bash
GPUS=0,1,2,3 \
MANIFEST="$SAM2D_ROOT/manifests/sav_train_6fps_full.parquet" \
SOURCE_STAGE1_CHECKPOINT=/path/to/stage1/checkpoints/best.pt \
RUN_ROOT="$SAM2D_ROOT/runs/my_tv21" WANDB_MODE=offline \
  bash scripts/lib/task_finetune_stages.sh all
```

See the [three-stage protocol](stage2/sam2_task_finetune_3stage.md) before
launching. Its historical driver evaluates validation and test after each
stage; repeated test evaluation must not be used to select configurations.
Select with full validation J&F and reserve final test reporting for the chosen
model. Resume with the same run identity and output/checkpoint directories.

`scripts/core/stage2_finetune_tinyvit.sh` reproduces a recorded run and requires
previous baseline artifacts plus four GPUs. It is not a fresh-machine quickstart.

## Evaluate and export

- VOS predictions: `tools/eval/run_sam2_vos_prompt_dataset.py`.
- Official J/F scoring: `tools/eval/run_sav_evaluator.py`.
- Prompted-image metrics: `tools/benchmark/benchmark_sav_prompt_masks.py`.
- Validation-based selection: `tools/train/select_task_checkpoint_by_val.py`.
- Experiment ledger: `bash scripts/core/report_experiments.sh`.

Run each Python CLI with `--help` for its required paths. Do not infer video
quality from image mIoU or compare multi-worker seconds/video with isolated FPS.

A task checkpoint is not a standalone model. Reconstruct it with the matching
SAM2 config, teacher skeleton checkpoint and student initializer, then load
non-image state strictly. `sam2_distill/models/task_finetune.py` and
`stage1_checkpoint.py` implement this contract. Transfer the complete bundle
and its hashes to the deployment repository; build engines on the target GPU.

## CPU regression checks

In a separate CPU development environment (not the company training container):

```bash
python -m pip install -e ".[test]"
python -m pytest tests
```

These checks cover small tensors and tooling. They do not establish convergence,
full-dataset quality, CUDA performance or Thor compatibility.
