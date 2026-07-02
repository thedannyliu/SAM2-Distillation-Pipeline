# EdgeTAM Reproduction Status

This document tracks the implementation state for the TinyViT-21M EdgeTAM
reproduction. Detailed paper-to-code notes remain in `docs/EdgeTAM/`; this file
is the concise engineering runbook.

## Scope

- Base method: EdgeTAM-style SAM2 training and evaluation.
- Student image encoder: TinyViT-21M from timm/Hugging Face.
- Smoke data cap: at most 500 images or frames per dataset.
- PACE usage: smoke validation only, single GPU when needed, `embers` QOS.
- Generated data, checkpoints, logs, and run outputs stay ignored by git.

## Implemented Entry Points

| Area | Entry point | Validation |
| --- | --- | --- |
| Data subset | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh prepare-data` | Copies SA-1B, SA-V, and COCO smoke subsets under `data/edgetam_smoke`. |
| Data validation | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh validate-data` | Decodes capped image/frame subsets and checks paths. |
| TinyViT metadata | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh probe-tinyvit` | Reports TinyViT feature reductions/channels using known metadata for default model. |
| EdgeTAM config | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh write-config` | Writes `runs/edgetam_smoke/configs/edgetam_tinyvit21m.yaml`. |
| Training config validation | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh validate-train-config` | Loads `configs/edgetam/tinyvit_video_distill_smoke.yaml` and instantiates the nested SAM2 task + EdgeTAM distillation loss. |
| Stage 1 feature smoke | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh stage1-feature-smoke` | Real SA-1B images through TinyViT adapter, MSE/backward, checkpoint. GPU smoke passed. |
| SA-V evaluator smoke | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh sav-eval-smoke` | GT-as-pred identity smoke; official SAM2 evaluator passed with J&F/J/F 100.0. |
| DAVIS-style VOS data | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh vos-style-data` | Packs real SA-V per-object masks into indexed DAVIS-style masks for generic VOS smoke. |
| DAVIS-style VOS eval | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh vos-style-eval-smoke` | GT-as-pred indexed-mask identity smoke passed with mean object IoU 1.0. |
| DAVIS 2017 data | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh davis-data` | Extracts a bounded subset directly from the official DAVIS 2017 trainval 480p zip. |
| DAVIS 2017 eval | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh davis-eval-smoke` | GT-as-pred indexed-mask identity smoke passed on the real DAVIS subset. |
| Video train shell | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh video-mask-train-smoke` | Real video clip loader, BCE+dice backward, checkpoint, and resume smoke passed. |
| EdgeTAM distillation loss smoke | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-distill-loss-smoke` | Official SAM2 multi-step task loss plus image/memory distillation backward passed. |
| Full SAM2 trainer smoke | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-full-trainer-smoke` | Runs upstream `training.trainer.Trainer` on the real VOS smoke subset with synthetic teacher features. 2-frame and 8-frame GPU smokes passed. |
| Full trainer resume smoke | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-full-trainer-resume-smoke` | Runs epoch 1 then resumes to epoch 2 in one Slurm allocation; checkpoint resume passed. |
| Full trainer cache-backed smoke | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-full-trainer-cache-smoke` | Runs upstream `Trainer` with frame-major teacher features loaded from a torch cache instead of synthetic teacher targets. |
| Teacher forward cache smoke | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-teacher-cache-smoke` | Instantiates `EdgeTAMTrain`, runs a no-grad real forward on a VOS smoke batch, and writes frame-major teacher feature cache tensors. |
| Full trainer forward-cache smoke | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-full-trainer-forward-cache-smoke` | Writes teacher cache from a real smoke-model forward, then consumes it in the upstream `Trainer`. |
| Image trainer smoke | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-image-trainer-smoke` | Runs upstream `Trainer` on real SA-1B smoke images/JSON masks as single-frame image pretraining. |
| Image forward-cache trainer smoke | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-image-forward-cache-smoke` | Writes real-forward teacher features from a SA-1B image batch, then trains the image-pretrain path against that cache. |
| Trainer checkpoint export smoke | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-export-checkpoint-smoke` | Exports an upstream `Trainer` checkpoint to model-only `{"model": ...}` format and strict-loads it with the TinyViT model config. |
| Progressive full trainer smoke | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-progressive-full-trainer-smoke` | Runs scaled progressive 2/4/8-frame phases through upstream `Trainer`, freezing the image encoder and disabling distillation after phase 1. |
| Official EdgeTAM image smoke | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-image-smoke` | Passed with the existing official EdgeTAM checkpoint on one COCO smoke image. |
| Official EdgeTAM image benchmark | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-image-benchmark` | Passed with A100 image predictor latency/FPS/peak-memory summary. |
| Official EdgeTAM VOS smoke | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-vos-smoke` | Passed with official EdgeTAM checkpoint on the SA-V smoke video. |
| Official EdgeTAM VOS late-object flag smoke | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-vos-track-later-smoke` | Passed with upstream `--track_object_appearing_later_in_video` enabled on the SA-V smoke video. |
| Official EdgeTAM SA-V eval | `scripts/pace/06_run_edgetam_tinyvit_smoke.sh edgetam-sav-eval` | Passed on existing `edgetam_vos_pred` with official SAM2 SA-V evaluator. |

## Code Modules

| Module | Purpose |
| --- | --- |
| `sam2_distill.models.tinyvit_adapter` | TinyViT feature adapter that emits SAM2 Stage 1 targets. |
| `sam2_distill.edgetam.config` | Generates EdgeTAM TinyViT YAML from timm feature metadata. |
| `sam2_distill.edgetam.timm_backbone` | SAM2-compatible timm feature wrapper with offline smoke support and optional local checkpoint path. |
| `sam2_distill.edgetam.compat` | Applies local compatibility patches for external EdgeTAM/SAM2 code without editing third-party checkouts. |
| `sam2_distill.edgetam.train_model` | SAM2Train subclass exposing `distill_F16` and `distill_F_M`. |
| `sam2_distill.edgetam.teacher_features` | Attaches detached teacher feature tensors to SAM2 per-frame outputs, including frame-major cache-backed targets. |
| `sam2_distill.edgetam.distillation_losses` | Adds EdgeTAM image/memory MSE distillation to SAM2 task loss. |
| `configs/edgetam/tinyvit_video_distill_smoke.yaml` | Minimal full-trainer Hydra config for TinyViT video distillation smoke/adaptation. |
| `tools/edgetam/validate_training_config.py` | Validates Hydra target paths and nested loss instantiation. |
| `tools/train/make_teacher_feature_cache_smoke.py` | Writes deterministic frame-major teacher feature caches for cache-backed trainer smoke tests. |
| `tools/train/cache_edgetam_teacher_features.py` | Instantiates the SAM2 trainer model/data path and caches real-forward `distill_F16` / `distill_F_M` features. |
| `tools/edgetam/export_trainer_checkpoint.py` | Converts upstream SAM2 `Trainer` checkpoints to model-only EdgeTAM/SAM2 checkpoint payloads and validates strict model loading. |
| `tools/train/smoke_stage1_features.py` | Minimal real-image feature training smoke. |
| `tools/eval/sav_identity_smoke.py` | SA-V prediction layout and official evaluator smoke. |
| `tools/eval/run_edgetam_vos_smoke.py` | Thin wrapper around official EdgeTAM `tools/vos_inference.py`. |
| `tools/eval/run_edgetam_image_smoke.py` | Thin wrapper around official EdgeTAM image predictor API. |
| `tools/eval/run_sav_evaluator.py` | Thin wrapper for official SAM2 `sav_evaluator.py` on existing predictions. |
| `tools/benchmark/benchmark_edgetam_image_predictor.py` | Official EdgeTAM image predictor latency/FPS/peak-memory benchmark. |
| `tools/data/make_vos_smoke_subset.py` | Builds bounded DAVIS-style VOS subsets, including SA-V per-object to packed-mask conversion. |
| `tools/eval/vos_identity_smoke.py` | DAVIS-style indexed-mask identity prediction and object IoU sanity check. |
| `tools/train/smoke_video_masks.py` | Lightweight real-video training shell smoke with checkpoint resume. |

## Current Smoke Results

The authoritative table is `docs/experiments/edgetam_smoke.md`.

- Data subset validation passed for SA-1B 400 images, SA-V 461 frames, and COCO 500 images.
- TinyViT config metadata passed for reductions `[4, 8, 16, 32]` and channels `[96, 192, 384, 576]`.
- SA-V identity evaluator smoke passed with official SAM2 evaluator.
- DAVIS-style VOS layout smoke passed on real SA-V frames packed to indexed PNG masks.
- DAVIS 2017 trainval 480p smoke passed on 2 videos / 120 frames; the 795MB archive was removed after extracting a 21MB subset.
- Video mask train shell smoke passed on 4 real VOS clips and resumed from `last.pt`.
- Training config validation passed: Hydra instantiated `EdgeTAMMultiStepDistillationLoss` wrapping official `MultiStepMultiMasksAndIous`.
- Training model instantiate validation passed: Hydra instantiated `EdgeTAMTrainWithTeacher` with the repo-owned TinyViT backbone, 30,007,218 parameters.
- EdgeTAM distillation loss smoke passed: official SAM2 task loss plus `loss_img_distill` and `loss_mem_distill` produced gradients for mask, IoU, object-score, F16, and memory features.
- Full upstream SAM2 trainer smoke passed on PACE:
  - `10669421`: `gpu-rtx6000`, `embers`, completed in 36s.
  - Real SA-V-derived VOS smoke batch, 2 sampled frames, upstream `Trainer`, TinyViT EdgeTAM model, synthetic teacher features, optimizer step, TensorBoard/log/checkpoint directories.
  - Earlier jobs `10669402` and `10669391` failed before `pandas` was added to the smoke environment; `pandas>=2.2` is now in `requirements-edgetam.txt`.
- Full trainer checkpoint resume smoke passed:
  - `10669524`: `gpu-rtx6000`, `embers`, completed in 55s.
  - In one allocation, wrote an epoch-1 checkpoint, resumed, and advanced to checkpoint epoch 2 / train step 2.
- 8-frame full trainer smoke passed:
  - `10669637`: `gpu-rtx6000`, `embers`, completed in 36s.
  - Real SA-V-derived VOS smoke batch, 8 sampled frames at 1024px, upstream `Trainer`, TinyViT EdgeTAM model, synthetic teacher features, optimizer step, checkpoint epoch 1 / train step 1.
  - Used `image_encoder_forward_batch_size=1` and `image_encoder_activation_checkpoint=true`.
  - Non-checkpointed 8-frame RTX 6000 job `10669444` failed with CUDA OOM, so the checkpointed image encoder path is the PACE low-memory smoke path.
- Cache-backed full trainer smoke passed:
  - `10669723`: `gpu-rtx6000`, `embers`, completed in 34s.
  - Real SA-V-derived VOS smoke batch, 2 sampled frames, upstream `Trainer`, TinyViT EdgeTAM model, deterministic frame-major teacher feature cache, optimizer step, checkpoint epoch 1 / train step 1.
  - This validates the trainer-side cache attachment path; the cache contents are synthetic smoke tensors, not frozen SAM2.1 teacher outputs yet.
- Teacher forward-cache smoke passed:
  - `10669771`: `gpu-rtx6000`, `embers`, completed in 52s.
  - The cache step instantiated `EdgeTAMTrain`, ran a no-grad real forward on the 2-frame VOS smoke batch, and wrote `teacher_distill_F16` / `teacher_distill_F_M` tensors with shape `[2, 1, 256, 64, 64]`.
  - The trainer step consumed that cache in the upstream `Trainer`, produced nonzero image/memory distillation losses, and checkpointed epoch 1 / train step 1.
  - This validates cache generation from a real model forward; the smoke teacher is the TinyViT EdgeTAM smoke config, not a SAM2.1-Hiera-L checkpoint.
- VOS checkpoint-loaded Hiera teacher smoke passed:
  - `10669972`: `gpu-rtx6000`, `embers`, completed in 44s.
  - Used `sam2_hiera_t.yaml` plus `sam2.1_hiera_tiny.pt` as a local smoke fallback for the video teacher cache path.
  - Loaded 471 tensors with 0 missing keys and 3 unused unexpected keys, wrote 2-frame cache tensors with shape `[2, 1, 256, 64, 64]`, then trained the TinyViT video path for 1 step.
  - The trainer consumed the Hiera Tiny teacher cache with `lambda_img=0.5`, `lambda_mem=0.25`, `loss_img_distill=0.7882`, and `loss_mem_distill=1.1933`.
  - This validates the video teacher config+checkpoint cache interface; the paper-scale teacher remains SAM2.1-Hiera-Large on the company cluster.
- SA-1B image trainer smoke passed:
  - `10669790`: `gpu-rtx6000`, `embers`, completed in 32s.
  - Used upstream `SA1BRawDataset` on 2 real SA-1B smoke images with up to 4 masks per image, `num_frames=1`, image distillation enabled, memory distillation disabled, and checkpointed epoch 1 / train step 2.
  - Earlier job `10669780` exposed an upstream EdgeTAM `PerceiverResampler.forward_2d` `expand().view()` bug for multi-object batches; `sam2_distill.edgetam.compat` patches that path to use `reshape`.
- SA-1B image forward-cache trainer smoke passed:
  - `10669885`: `gpu-rtx6000`, `embers`, completed in 48s.
  - The cache step used `SA1BRawDataset` on 1 real image with up to 4 masks and wrote real-forward `teacher_distill_F16` / `teacher_distill_F_M` tensors with shape `[1, 4, 256, 64, 64]`.
  - The trainer step consumed that cache, kept `lambda_img=1.0` and `lambda_mem=0.0`, produced `loss_img_distill=0.7571`, and checkpointed epoch 1 / train step 1.
  - This validates the image-pretrain cache plumbing; the smoke teacher is the TinyViT EdgeTAM smoke config, not a SAM2.1-Hiera-L checkpoint.
- SA-1B image checkpoint-loaded Hiera teacher smoke passed:
  - `10669946`: `gpu-rtx6000`, `embers`, completed in 43s.
  - Used `sam2_hiera_t.yaml` plus `sam2.1_hiera_tiny.pt` as a local smoke fallback; strict load first failed in job `10669927` because the checkpoint had 3 unused unexpected keys and no missing keys.
  - The rerun explicitly allowed unexpected keys, loaded 471 tensors with 0 missing keys, wrote cache tensors with shape `[1, 1, 256, 64, 64]`, then trained the TinyViT image-pretrain path for 1 step with `loss_img_distill=0.7906`.
  - This validates the config+checkpoint teacher cache interface; the paper-scale teacher remains SAM2.1-Hiera-Large/Base Plus on the company cluster.
- Progressive full trainer smoke passed:
  - `10669805`: `gpu-rtx6000`, `embers`, completed in 1m13s.
  - Ran scaled phases `2/4/8` frames on the real VOS smoke subset through upstream `Trainer`.
  - Phase 1 used image/memory distillation weights `0.5/0.25`; phases 2 and 3 set `freeze_image_encoder=true` and `lambda_img=lambda_mem=0`.
  - Each phase checkpointed epoch 1 / train step 1.
- Trainer checkpoint export smoke passed:
  - CPU smoke converted `runs/edgetam_smoke/edgetam_vos_hiera_tiny_cache_trainer_smoke/checkpoints/checkpoint.pt` to `runs/edgetam_smoke/edgetam_export_checkpoint_smoke/model.pt`.
  - The exported model-only payload contains 598 tensors plus metadata from epoch 1 / train step 1.
  - Strict-load validation against `runs/edgetam_smoke/configs/edgetam_tinyvit21m.yaml` instantiated `SAM2Base` with 0 missing and 0 unexpected keys.
- Existing official EdgeTAM checkpoint found at `/storage/project/r-agarg35-0/eliu354/projects/efficientsam3-benchmark/external/EdgeTAM/checkpoints/edgetam.pt`.
- Official EdgeTAM checkpoint metadata smoke passed: `torch.load(..., weights_only=True)` found a `model` state dict with 982 tensors and `edgetam.yaml` exists.
- Stage 1 feature train smoke passed on PACE:
  - `10669130`: `gpu-a100`, `embers`, completed in 52s.
  - Real SA-1B smoke images flowed through TinyViT adapter, synthetic teacher feature targets, backward, AdamW step, and checkpoint writing.
  - Duplicate pending jobs `10669025` and `10669030` were cancelled to avoid redundant smoke runs.
- Official EdgeTAM image smoke passed on PACE:
  - `10669202`: `gpu-a100`, `embers`, completed in 1m17s.
  - Produced 3 masks on one COCO smoke image; best mask area was 455 pixels.
- Official EdgeTAM image benchmark passed on PACE:
  - `10669213`: `gpu-a100`, `embers`, completed in 32s.
  - Mean latency 0.02787s, p95 0.03280s, mean FPS 35.88, peak memory 355.97MB.
- Official EdgeTAM VOS smoke and evaluator passed:
  - `10669147`: `gpu-a100`, `embers`, completed in 1m44s.
  - Propagated one SA-V smoke video at about 40.5 iterations/s and wrote per-object masks.
  - Official SA-V evaluator returned J&F 91.5, J 89.1, F 93.9 on `sav_011944`.
  - Duplicate pending `gpu-l40s` job `10669187` was cancelled.
- Official EdgeTAM VOS late-object flag smoke passed:
  - `10669832`: `gpu-rtx6000`, `embers`, completed in 1m09s.
  - Reused the SA-V smoke video while enabling upstream `--track_object_appearing_later_in_video`, the flag needed by YouTube-VOS/LVOS-style evaluation.
  - Official SA-V evaluator on the flag-path predictions returned J&F 91.7, J 89.2, F 94.3 on `sav_011944`.

## Remaining Work

| Phase | Next implementation | Smoke validation |
| --- | --- | --- |
| Official baseline | Run official EdgeTAM checkpoint on the SA-V smoke subset. | SA-V smoke inference, late-object flag path, and official evaluator passed; extend to full SA-V/DAVIS/MOSE/YTVOS when full datasets are available. |
| Image pretrain | Swap the local Hiera Tiny smoke fallback for frozen SAM2.1-Hiera-B+/Large image teacher features and scale from 1-image cache smoke to 100-image overfit. | SA-1B single-frame image trainer smoke passed with synthetic targets; SA-1B image forward-cache trainer smoke passed with TinyViT smoke-model and checkpoint-loaded Hiera Tiny teacher cache tensors. |
| Video train | Swap the local Hiera Tiny smoke fallback for frozen SAM2.1-Hiera-L teacher config/weights in `cache_edgetam_teacher_features.py`. | 2-frame full trainer, checkpoint resume, 8-frame low-memory trainer, deterministic-cache trainer, TinyViT real-forward-cache trainer, and checkpoint-loaded Hiera Tiny teacher-cache trainer smokes passed. |
| Progressive schedule | Run full-size `8/16/32` progressive phases on company GPUs after teacher/weights are finalized. | Scaled full upstream Trainer `2/4/8` progressive smoke passed; lightweight 8/16/32 shell smoke also passed. |
| Full eval | Add MOSE/YTVOS wrappers beside SA-V and DAVIS when those datasets are available locally. | Generic indexed-mask layout smoke and the YTVOS/LVOS late-object flag path passed on SA-V smoke data; real MOSE/YTVOS dataset smoke remains pending. |
