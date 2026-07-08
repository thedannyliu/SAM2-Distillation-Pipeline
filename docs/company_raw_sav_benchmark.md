# Raw SA-V Shard Benchmark

This document describes the quick benchmark used for raw company SA-V train shards, such as:

```text
/mnt/dataset/data/danny-dataset/SA-V/sav_train/sav_030
```

The benchmark has two modes:

- image segmentation with box and point prompts
- video object segmentation with SAM2 memory, initialized from box and point prompts

The main entrypoint is `scripts/company/15_benchmark_raw_sav_shard_suite.sh`.

## Dataset Preparation

Raw SA-V train shards contain MP4 videos and train JSON masklets, not the SA-V val/test PNG layout. The preparation step converts a small subset into:

```text
JPEGImages_24fps/<video>/<frame>.jpg
Annotations_6fps/<video>/<object>/<frame>.png
sav_train_benchmark.txt
```

The conversion is implemented in `tools/data/prepare_sav_train_shard_benchmark.py`:

- detects MP4 and JSON roots: `tools/data/prepare_sav_train_shard_benchmark.py:44`
- extracts 24fps JPEG frames: `tools/data/prepare_sav_train_shard_benchmark.py:66`
- decodes SA-V RLE masklets with `pycocotools`: `tools/data/prepare_sav_train_shard_benchmark.py:98`
- writes per-object sparse PNG masks every 4 frames: `tools/data/prepare_sav_train_shard_benchmark.py:142`
- writes the selected video list and preparation summary: `tools/data/prepare_sav_train_shard_benchmark.py:216`

The suite calls this through its `prepare` stage:

- defaults and input paths: `scripts/company/15_benchmark_raw_sav_shard_suite.sh:7`
- prepare wrapper: `scripts/company/15_benchmark_raw_sav_shard_suite.sh:57`

## Models Compared

The suite benchmarks:

- SAM2.1-L
- SAM2.1-B+
- Stage1 TinyViT encoder checkpoints listed in `scripts/company/15_benchmark_raw_sav_shard_suite.sh:23`

Stage1 checkpoints are encoder-only. For image mode and video mode, they are attached to the original SAM2.1-L prompt/mask/memory stack rather than loaded as full SAM2 checkpoints.

TinyViT architecture is inferred from checkpoint tensor shapes, so mislabeled run names do not silently instantiate the wrong backbone:

- model inference from `projections.image_embed.weight`: `sam2_distill/models/stage1_checkpoint.py:38`
- checkpoint filename resolution: `sam2_distill/models/stage1_checkpoint.py:47`

## Image Segmentation Mode

For image segmentation, each annotated object/frame pair is treated as one promptable segmentation example.

The benchmark uses:

- box prompt from the GT mask bbox
- point prompt from the GT mask centroid-nearest foreground pixel
- SAM2 mask decoder prediction
- IoU and AP metrics against the GT mask

Implementation:

- loads SAM2.1 image predictor: `tools/benchmark/benchmark_sav_prompt_masks.py:113`
- loads Stage1 TinyViT encoder and SAM2.1-L decoder stack: `tools/benchmark/benchmark_sav_prompt_masks.py:124`
- injects Stage1 features into `SAM2ImagePredictor._features`: `tools/benchmark/benchmark_sav_prompt_masks.py:170`
- builds object/frame records from `Annotations_6fps`: `tools/benchmark/benchmark_sav_prompt_masks.py:210`
- derives box and point prompts: `tools/benchmark/benchmark_sav_prompt_masks.py:269`
- runs prompt prediction: `tools/benchmark/benchmark_sav_prompt_masks.py:318`
- computes mIoU, AP, and latency: `tools/benchmark/benchmark_sav_prompt_masks.py:413`

The suite invokes this for every model and both prompt types:

- SAM2 image calls: `scripts/company/15_benchmark_raw_sav_shard_suite.sh:81`
- Stage1 image calls: `scripts/company/15_benchmark_raw_sav_shard_suite.sh:105`
- image loop over models/prompts: `scripts/company/15_benchmark_raw_sav_shard_suite.sh:134`

Image output metrics:

- `mIoU`
- `AP`
- `AP50`
- `AP75`
- `mean_set_image_seconds`
- `mean_prompt_seconds`
- `mean_total_object_seconds`

## Video Tracking Mode

For video tracking, each object is initialized on its first annotated frame, not necessarily frame 0. This is important for raw SA-V train shards because objects may appear later in the video.

For each object:

- find the first available GT mask
- create either a box prompt or a point prompt from that mask
- call `SAM2VideoPredictor.add_new_points_or_box`
- run `propagate_in_video`
- save one PNG prediction per object/frame
- evaluate with the official SAM2 SA-V evaluator

Implementation:

- prompt VOS runner args: `tools/eval/run_sam2_vos_prompt_dataset.py:23`
- bf16 autocast matching official SAM2 VOS inference behavior: `tools/eval/run_sam2_vos_prompt_dataset.py:47`
- Stage1 encoder is patched into `forward_image`: `tools/eval/run_sam2_vos_prompt_dataset.py:67`
- Stage1 `forward_image` returns SAM2-compatible `backbone_fpn` and `vision_pos_enc`: `tools/eval/run_sam2_vos_prompt_dataset.py:91`
- SAM2 video predictor construction: `tools/eval/run_sam2_vos_prompt_dataset.py:122`
- first GT mask per object: `tools/eval/run_sam2_vos_prompt_dataset.py:182`
- box/point prompt creation and `add_new_points_or_box`: `tools/eval/run_sam2_vos_prompt_dataset.py:200`
- video propagation and PNG writing: `tools/eval/run_sam2_vos_prompt_dataset.py:234`

The suite invokes video tracking for every model and both prompt types:

- SAM2 VOS calls: `scripts/company/15_benchmark_raw_sav_shard_suite.sh:148`
- Stage1 VOS calls: `scripts/company/15_benchmark_raw_sav_shard_suite.sh:181`
- VOS loop over models/prompts: `scripts/company/15_benchmark_raw_sav_shard_suite.sh:220`

Video tracking output metrics:

- `J&F`
- `J`
- `F`
- `elapsed_sec`
- `sec_per_video`

The official evaluator wrapper cleans progress-bar output and parses the global metrics:

- output cleanup and metric parsing: `tools/eval/run_sav_evaluator.py:26`
- official evaluator command: `tools/eval/run_sav_evaluator.py:58`

## Overlay Artifacts

The benchmark can generate visual artifacts after VOS prediction PNGs exist. This is a post-processing step and does not rerun model inference.

Image mode artifacts are written by `tools/benchmark/benchmark_sav_prompt_masks.py` when `--save-video-frame-artifacts` is enabled. The company suite passes this through `IMAGE_ARTIFACT_VIDEOS`:

- image artifact selection for first/middle/last annotated frames: `tools/benchmark/benchmark_sav_prompt_masks.py:316`
- image masks and overlays are saved under `<OUT_ROOT>/image/<model>/<prompt>/frame_artifacts/`
- suite wiring: `scripts/company/15_benchmark_raw_sav_shard_suite.sh:106`

Video tracking overlays are generated from saved VOS prediction PNGs:

- scans all prediction roots matching `<OUT_ROOT>/vos/*/*/pred`: `scripts/company/15_benchmark_raw_sav_shard_suite.sh:240`
- writes one full-timeline MP4 per selected video: `tools/eval/make_vos_overlay_artifacts.py:95`
- uses all frames from `JPEGImages_24fps`, not only sparse GT annotation frames: `tools/eval/make_vos_overlay_artifacts.py:52`
- copies GT and predicted masks under `<OUT_ROOT>/vos/<model>/<prompt>/artifacts/masks/`

Overlay colors:

- on frames with GT annotations:
  - yellow: GT and prediction overlap
  - green: GT only, missed by prediction
  - red: prediction only, false positive area
- on frames without GT annotations:
  - blue: prediction-only visualization; this is not treated as a false positive

To generate or regenerate overlay MP4s for every completed VOS prediction root in an existing run:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline

RAW_SHARD_ROOT=/mnt/dataset/data/danny-dataset/SA-V/sav_train/sav_030 \
MAX_VIDEOS=10 \
MAX_OBJECTS_PER_VIDEO=2 \
IMAGE_ARTIFACT_VIDEOS=3 \
VOS_OVERLAY_VIDEOS=3 \
VOS_OVERLAY_FRAMES=0 \
OUT_ROOT=/group-volume/danny-dataset/sam2_distill/runs/raw_sav030_stage1_video_suite_10vid_artifacts \
scripts/company/15_benchmark_raw_sav_shard_suite.sh artifacts
```

The MP4s are written to:

```text
<OUT_ROOT>/vos/<model>/<prompt>/artifacts/*_overlay.mp4
```

`VOS_OVERLAY_FRAMES=0` means write the full video timeline. `VOS_OVERLAY_VIDEOS=3` means save overlay MP4s for the first three videos in `sav_train_benchmark.txt`. Increase this if you want MP4s for more videos.

## Summary Table

All per-model results are collected into:

```text
<OUT_ROOT>/benchmark_summary.csv
<OUT_ROOT>/benchmark_summary.json
```

The summary code:

- image rows: `tools/benchmark/summarize_sav_benchmark_suite.py:25`
- video tracking rows: `tools/benchmark/summarize_sav_benchmark_suite.py:61`
- CSV/JSON output fields: `tools/benchmark/summarize_sav_benchmark_suite.py:101`

The suite calls summarization at `scripts/company/15_benchmark_raw_sav_shard_suite.sh:234`.

## Company Command

Run all benchmark stages:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline

RAW_SHARD_ROOT=/mnt/dataset/data/danny-dataset/SA-V/sav_train/sav_030 \
MAX_VIDEOS=2 \
MAX_OBJECTS_PER_VIDEO=2 \
MAX_IMAGE_OBJECTS=200 \
OUT_ROOT=/group-volume/danny-dataset/sam2_distill/runs/raw_sav030_stage1_video_suite \
scripts/company/15_benchmark_raw_sav_shard_suite.sh all
```

Print the final table:

```bash
column -s, -t /group-volume/danny-dataset/sam2_distill/runs/raw_sav030_stage1_video_suite/benchmark_summary.csv | less -S
```

To resume after a crash, rerun the same command. Completed `summary.json` and `eval_summary.json` files are skipped by default:

- skip completed image runs: `scripts/company/15_benchmark_raw_sav_shard_suite.sh:87`
- skip completed VOS runs: `scripts/company/15_benchmark_raw_sav_shard_suite.sh:156`

To force a full rerun in the same output directory:

```bash
SKIP_DONE=0 scripts/company/15_benchmark_raw_sav_shard_suite.sh all
```

Using a fresh `OUT_ROOT` is safer when comparing against older failed runs.
