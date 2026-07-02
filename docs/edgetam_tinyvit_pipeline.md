# EdgeTAM TinyViT Pipeline

This repo keeps the EdgeTAM reproduction pipeline open-source friendly:

- Source, configs, docs, and small manifests are tracked.
- Data, checkpoints, run outputs, overlays, third-party checkouts, and PDFs are ignored.
- PACE smoke subsets are capped at 500 images or frames per dataset.
- Full training is intended for the company cluster, with datasets and checkpoints under `/danny-dataset`.

## Local PACE Smoke

Prepare and validate small real-data subsets:

```bash
scripts/pace/06_run_edgetam_tinyvit_smoke.sh all-cpu
```

Prepare the repo-local conda env when the active shell does not have SAM2/TinyViT dependencies:

```bash
scripts/setup/prepare_edgetam_env.sh
```

Submit single-GPU smoke tasks independently:

```bash
sbatch --export=TASK=probe-tinyvit scripts/pace/slurm_edgetam_smoke.sbatch
sbatch --export=TASK=write-config scripts/pace/slurm_edgetam_smoke.sbatch
sbatch --export=TASK=stage1-feature-smoke scripts/pace/slurm_edgetam_smoke.sbatch
```

When reusing an existing dependency environment for a smoke run, point Slurm at
that prefix:

```bash
sbatch --partition=gpu-l40s \
  --account=gts-agarg35-ideas_l40s \
  --qos=embers \
  --export=ALL,TASK=stage1-feature-smoke,ENV_PREFIX=/path/to/venv \
  scripts/pace/slurm_edgetam_smoke.sbatch
```

Default smoke inputs are existing PACE-local SA-1B, SA-V, and COCO mirrors. Outputs go to ignored repo-local paths:

```text
data/edgetam_smoke
runs/edgetam_smoke
```

## Upstreams

SAM2 and EdgeTAM should be checked out under ignored `third_party/` paths:

```bash
scripts/setup/clone_upstreams.sh
```

Set `SAM2_REF` and `EDGETAM_REF` to pin exact commits for reproducible runs.

## TinyViT Config

Generate an EdgeTAM TinyViT config from timm feature metadata:

```bash
python tools/edgetam/write_tinyvit_edgetam_config.py \
  --out /storage/scratch1/9/eliu354/sam2_distill/edgetam/configs/edgetam_tinyvit21m.yaml
```

The generator probes TinyViT channels and reductions instead of hardcoding FPN channels.
For the default TinyViT-21M-512 model, the tool uses a fast known-metadata path to avoid slow CPU initialization; pass `--force-probe` when validating a new timm version or model variant.

Generated TinyViT configs use `sam2_distill.edgetam.timm_backbone.TimmBackbone`
instead of the upstream EdgeTAM wrapper because the upstream wrapper hardcodes
`pretrained=True`. Smoke configs default to `pretrained: false`; company runs
should set a local `checkpoint_path` or enable pretrained loading only when the
environment has the expected Hugging Face/timm cache.

## Training Feature Contract

`sam2_distill.edgetam.train_model.EdgeTAMTrain` extends SAM2 training outputs with:

```text
distill_F16
distill_F_M
```

Teacher/student wrappers must add matching teacher keys before the loss runs:

```text
teacher_distill_F16
teacher_distill_F_M
```

`EdgeTAMMultiStepDistillationLoss` wraps the standard SAM2 multi-step mask/IoU/object loss and adds `loss_img_distill` and `loss_mem_distill` when their weights are nonzero.

`sam2_distill.edgetam.train_model.EdgeTAMTrainWithTeacher` attaches teacher
features before the upstream SAM2 trainer calls the loss. In smoke mode, the
same wrapper can use `synthetic_teacher: true`; in full runs, replace that with
a frozen teacher model config or `teacher_feature_cache_path`.

Cache-backed teacher targets are loaded by
`sam2_distill.edgetam.teacher_features.TeacherFeatureCache`. The cache is a
torch checkpoint with frame-major `teacher_distill_F16` and
`teacher_distill_F_M` tensors. This lets the upstream `Trainer` path consume
precomputed teacher features without changing the SAM2 loss call signature.

The minimal full-trainer config is:

```text
configs/edgetam/tinyvit_video_distill_smoke.yaml
```

Validate its Hydra targets and nested loss construction with:

```bash
EDGETAM_ROOT=/path/to/EdgeTAM \
SAM2_TRAINING_ROOT=/path/to/sam2 \
scripts/pace/06_run_edgetam_tinyvit_smoke.sh validate-train-config
```

Smoke the official SAM2 task loss plus EdgeTAM image/memory distillation
backward path with:

```bash
EDGETAM_ROOT=/path/to/EdgeTAM \
SAM2_TRAINING_ROOT=/path/to/sam2 \
scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-distill-loss-smoke
```

Run the minimal full upstream SAM2 trainer on the real VOS smoke subset with:

```bash
EDGETAM_ROOT=/path/to/EdgeTAM \
SAM2_TRAINING_ROOT=/path/to/sam2 \
scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-full-trainer-smoke
```

Submit it on PACE through `scripts/pace/slurm_edgetam_smoke.sbatch` with
`TASK=edgetam-full-trainer-smoke`; keep `qos=embers`.

For the 8-frame / 1024px smoke on smaller PACE GPUs, use the checkpointed image
encoder path:

```bash
TASK=edgetam-full-trainer-smoke \
EDGETAM_TRAINER_SMOKE_FRAMES=8 \
EDGETAM_TRAINER_SMOKE_IMAGE_ENCODER_BATCH=1 \
EDGETAM_TRAINER_SMOKE_IMAGE_ENCODER_CKPT=1 \
sbatch --qos=embers scripts/pace/slurm_edgetam_smoke.sbatch
```

This keeps the training behavior on the upstream `Trainer` path while reducing
TinyViT activation memory. Leave these variables unset for full-memory company
runs unless the container/GPU needs the lower-memory path.

Validate checkpoint resume in one Slurm allocation with:

```bash
TASK=edgetam-full-trainer-resume-smoke \
EDGETAM_TRAINER_SMOKE_FRAMES=2 \
sbatch --qos=embers scripts/pace/slurm_edgetam_smoke.sbatch
```

The resume smoke first writes an epoch-1 checkpoint and then restarts the
upstream trainer to epoch 2 in the same output directory. The summary records
`checkpoint_before` and `checkpoint_after` epochs and train steps.

Validate cache-backed teacher attachment with:

```bash
TASK=edgetam-full-trainer-cache-smoke \
EDGETAM_TRAINER_SMOKE_FRAMES=2 \
sbatch --qos=embers scripts/pace/slurm_edgetam_smoke.sbatch
```

The smoke task first writes a deterministic cache with
`tools/train/make_teacher_feature_cache_smoke.py`, then runs the full upstream
trainer with `teacher_feature_cache_path` instead of `synthetic_teacher`. The
smoke cache is intentionally synthetic; company runs should replace that file
with frozen SAM2.1 teacher features.

Validate teacher-cache generation from a real SAM2 trainer forward with:

```bash
TASK=edgetam-teacher-cache-smoke \
EDGETAM_TRAINER_SMOKE_FRAMES=2 \
sbatch --qos=embers scripts/pace/slurm_edgetam_smoke.sbatch
```

To validate the checkpoint-loaded video teacher path, provide a SAM2 model
config and checkpoint:

```bash
TASK=edgetam-full-trainer-forward-cache-smoke \
EDGETAM_TEACHER_MODEL_CONFIG=/path/to/sam2_hiera_l.yaml \
EDGETAM_TEACHER_CHECKPOINT=/path/to/sam2.1_hiera_large.pt \
EDGETAM_TRAINER_SMOKE_FRAMES=2 \
EDGETAM_TRAINER_SMOKE_OBJECTS=1 \
sbatch --qos=embers scripts/pace/slurm_edgetam_smoke.sbatch
```

The task first caches teacher `distill_F16` / `distill_F_M` from the checkpoint
teacher, then runs the TinyViT trainer against `teacher_feature_cache_path`.
Checkpoint loading is strict unless `EDGETAM_TEACHER_ALLOW_UNEXPECTED_KEYS=1`
is set for a known-compatible fallback with no missing keys.

Validate the complete forward-cache path with:

```bash
TASK=edgetam-full-trainer-forward-cache-smoke \
EDGETAM_TRAINER_SMOKE_FRAMES=2 \
sbatch --qos=embers scripts/pace/slurm_edgetam_smoke.sbatch
```

`tools/train/cache_edgetam_teacher_features.py` rewrites the smoke model target
to `sam2_distill.edgetam.train_model.EdgeTAMTrain`, runs one no-grad teacher
forward through the same VOS loader, and writes `teacher_distill_F16` /
`teacher_distill_F_M` tensors for `TeacherFeatureCache`. On PACE this uses the
TinyViT smoke config as the teacher model. For company runs, use the same cache
tool with a frozen SAM2.1-Hiera-L teacher config/weights and keep the generated
cache under `/danny-dataset`.

Run the SA-1B single-frame image-pretrain smoke with:

```bash
TASK=edgetam-image-trainer-smoke \
EDGETAM_IMAGE_TRAINER_SMOKE_ITEMS=2 \
EDGETAM_IMAGE_TRAINER_SMOKE_OBJECTS=4 \
sbatch --qos=embers scripts/pace/slurm_edgetam_smoke.sbatch
```

This task uses upstream `SA1BRawDataset` on real SA-1B smoke images/JSON masks,
forces `num_frames=1`, enables image feature distillation, and disables memory
feature distillation. It exercises the Phase 3 `Simg` training surface without
running a long SA-1B epoch. The runner writes a bounded file list under the run
directory so smoke jobs do not iterate over the full local subset.

Validate the SA-1B image-pretrain path with real-forward teacher cache targets:

```bash
TASK=edgetam-image-forward-cache-smoke \
EDGETAM_IMAGE_TRAINER_SMOKE_ITEMS=1 \
EDGETAM_IMAGE_TRAINER_SMOKE_OBJECTS=4 \
sbatch --qos=embers scripts/pace/slurm_edgetam_smoke.sbatch
```

This task first calls `tools/train/cache_edgetam_teacher_features.py` in
`sa1b-image` mode to write frame-major `teacher_distill_F16` /
`teacher_distill_F_M` tensors from a real image batch, then runs the upstream
trainer with `teacher_feature_cache_path`. On PACE the cache teacher is the
TinyViT smoke config; company runs should use the same cache flow with the
frozen SAM2.1-Hiera teacher config and weights.

To validate the checkpoint-loaded teacher path, provide a SAM2 model config and
checkpoint to the same task:

```bash
TASK=edgetam-image-forward-cache-smoke \
EDGETAM_TEACHER_MODEL_CONFIG=/path/to/sam2_hiera_l.yaml \
EDGETAM_TEACHER_CHECKPOINT=/path/to/sam2.1_hiera_large.pt \
EDGETAM_IMAGE_TRAINER_SMOKE_ITEMS=1 \
EDGETAM_IMAGE_TRAINER_SMOKE_OBJECTS=1 \
sbatch --qos=embers scripts/pace/slurm_edgetam_smoke.sbatch
```

Checkpoint loading is strict by default. Set
`EDGETAM_TEACHER_ALLOW_UNEXPECTED_KEYS=1` only for known-compatible smoke
fallbacks that have no missing keys and a small set of unused unexpected keys.

`sam2_distill.edgetam.compat` patches the external EdgeTAM
`PerceiverResampler.forward_2d` multi-object path from `expand().view()` to
`expand().reshape()`. The patch is applied when `EdgeTAMTrain` is imported and
is required for image batches with more than one mask/object.

Run the scaled full-trainer progressive schedule smoke with:

```bash
TASK=edgetam-progressive-full-trainer-smoke \
EDGETAM_PROGRESSIVE_FULL_FRAMES="2 4 8" \
sbatch --qos=embers scripts/pace/slurm_edgetam_smoke.sbatch
```

The smoke uses the upstream `Trainer` for each phase. Phase 1 keeps image and
memory distillation enabled; later phases set `freeze_image_encoder=true` and
`lambda_img=lambda_mem=0`, matching the EdgeTAM fine-tuning rule at smoke scale.
For company runs, set `EDGETAM_PROGRESSIVE_FULL_FRAMES="8 16 32"` after the
teacher cache/weights path is finalized.

Export an upstream `Trainer` checkpoint to a model-only checkpoint for
inference/eval with:

```bash
EDGETAM_EXPORT_TRAINER_CHECKPOINT=/path/to/run/checkpoints/checkpoint.pt \
EDGETAM_EXPORT_MODEL_CONFIG=/path/to/edgetam_tinyvit21m.yaml \
EDGETAM_EXPORT_SMOKE_OUT_DIR=/path/to/export_dir \
scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-export-checkpoint-smoke
```

The export writes `model.pt` in SAM2/EdgeTAM `{"model": state_dict}` format and
strict-loads it against the model-only YAML. This is the bridge from training
checkpoints to official-style eval or deployment scripts.

Smoke-test the exported checkpoint through `SAM2ImagePredictor` with:

```bash
TASK=edgetam-exported-image-smoke \
EDGETAM_EXPORT_SMOKE_OUT_DIR=/path/to/export_dir \
EDGETAM_EXPORT_MODEL_CONFIG=/path/to/edgetam_tinyvit21m.yaml \
sbatch --qos=embers scripts/pace/slurm_edgetam_smoke.sbatch
```

If `model.pt` is missing in `EDGETAM_EXPORT_SMOKE_OUT_DIR`, the task first runs
the export step, then predicts masks on the bounded COCO smoke image.

## Smoke Train/Eval Entry Points

Stage 1 feature smoke uses real SA-1B smoke images, forwards them through the
TinyViT SAM2 adapter, runs feature MSE/backward on synthetic teacher targets,
and writes a resumable checkpoint plus JSONL metrics:

```bash
scripts/pace/06_run_edgetam_tinyvit_smoke.sh stage1-feature-smoke
```

SA-V evaluator smoke copies ground-truth masks as predictions and can run the
official SAM2 SA-V evaluator:

```bash
SAV_EVALUATOR=/path/to/sam2/sav_dataset/sav_evaluator.py \
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh sav-eval-smoke
```

Generic DAVIS/MOSE/YTVOS-style layout smoke can be exercised with real SA-V
frames packed into indexed PNG masks:

```bash
scripts/pace/06_run_edgetam_tinyvit_smoke.sh vos-style-data
scripts/pace/06_run_edgetam_tinyvit_smoke.sh vos-style-eval-smoke
```

Real DAVIS 2017 smoke uses the official trainval 480p archive and extracts only
the bounded subset:

```bash
DAVIS_ZIP=/path/to/DAVIS-2017-trainval-480p.zip \
DAVIS_MAX_FRAMES=120 \
scripts/pace/06_run_edgetam_tinyvit_smoke.sh davis-data

DAVIS_MAX_FRAMES=120 \
scripts/pace/06_run_edgetam_tinyvit_smoke.sh davis-eval-smoke
```

The lightweight video training shell smoke checks real clip loading,
mask-supervised backward, checkpoint writing, and resume. It is intentionally
not a substitute for the full SAM2/EdgeTAM video trainer.

```bash
scripts/pace/06_run_edgetam_tinyvit_smoke.sh video-mask-train-smoke
VIDEO_SMOKE_RESUME=1 scripts/pace/06_run_edgetam_tinyvit_smoke.sh video-mask-train-smoke
```

Progressive schedule smoke runs 8/16/32-frame phases with the EdgeTAM
fine-tuning rules recorded in the summaries:

```bash
PROGRESSIVE_SMOKE_IMAGE_SIZE=64 \
PROGRESSIVE_SMOKE_MAX_CLIPS=2 \
PROGRESSIVE_SMOKE_STEPS=1 \
scripts/pace/06_run_edgetam_tinyvit_smoke.sh progressive-video-smoke
```

Official EdgeTAM VOS inference smoke is available once an EdgeTAM checkpoint is
present:

```bash
EDGETAM_ROOT=/path/to/EdgeTAM \
EDGETAM_CHECKPOINT=/path/to/edgetam.pt \
scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-vos-smoke
```

Exercise the YouTube-VOS/LVOS late-appearing-object inference flag with the
same bounded SA-V smoke data:

```bash
EDGETAM_ROOT=/path/to/EdgeTAM \
EDGETAM_CHECKPOINT=/path/to/edgetam.pt \
scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-vos-track-later-smoke
```

After predictions exist, evaluate them without rewriting the prediction root:

```bash
SAV_EVALUATOR=/path/to/sam2/sav_dataset/sav_evaluator.py \
  scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-sav-eval
```

Set `EDGETAM_VOS_OUT_DIR` when evaluating a non-default prediction directory.

The official image predictor can be smoke-tested on one real image:

```bash
EDGETAM_ROOT=/path/to/EdgeTAM \
EDGETAM_CHECKPOINT=/path/to/edgetam.pt \
scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-image-smoke
```

For speed smoke, run the official image predictor benchmark:

```bash
EDGETAM_ROOT=/path/to/EdgeTAM \
EDGETAM_CHECKPOINT=/path/to/edgetam.pt \
EDGETAM_BENCH_LIMIT=2 \
EDGETAM_BENCH_WARMUP=1 \
EDGETAM_BENCH_ITERS=2 \
scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-image-benchmark
```

## Experiment Tracking

Record every smoke run in:

```text
docs/experiments/edgetam_smoke.md
```

Use the helper when running jobs:

```bash
python tools/experiments/record_experiment.py \
  --task "TinyViT shape probe" \
  --data "no data" \
  --command "scripts/pace/06_run_edgetam_tinyvit_smoke.sh probe-tinyvit" \
  --result "pass"
```
