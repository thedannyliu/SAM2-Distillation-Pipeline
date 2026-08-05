Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Testing Guidelines

There is no formal unit-test suite. Validate with the smallest workflow that exercises your change: import/package checks for library edits, manifest generation for experiment tooling, and one smoke row or Slurm array for training/eval changes. Record task, seed, manifest path, GPU type, checkpoint/output directory, and any W&B project in PR notes.

## Commit & Pull Request Guidelines

Recent history uses imperative subjects: `Add ...` for new utilities or experiment flows, `Document ...` and `Record ...` for status/results, `Fix ...` for behavior corrections, and `Update ...` for refreshed job records. Keep commits narrow and separate code from large generated artifacts. PRs should state the research or pipeline impact, list validation commands or smoke runs, link relevant docs, and call out cluster-resource, checkpoint, dataset, or W&B implications.

Use git like machine learning engineer to update. For example, when there is new development, open a new branch a develop there then merge it back.
For company-side SAM2 distillation work, W&B connectivity has been verified and should be used for primary experiment tracking when allowed. TensorBoard remains useful as a local fallback or additional log. Make each training process continuable; when resuming, reuse the same W&B run ID and write to the same TensorBoard log directory and checkpoint directory.

## Cluster & QOS Policy

Current runs should target PACE-Phoenix by default. Use the `embers` QOS for GPU jobs because it is not charged to the account, though it has lower priority. Do not submit with `inferno` unless the user explicitly approves it; `inferno` has normal priority but incurs account charges.

Default Slurm account for H100、H200、A100 work: `gts-agarg35`.
Default Slurm account for L40S work: `gts-agarg35-ideas_l40s`.

Usually current node is home node, and don't have GPUs. GPU must be accessed from submitting a job.
Don't do any GPU job, env setup at $HOME, keep it clean.
Do not create a venv for company container runs unless explicitly requested; use the container Python directly so the preinstalled PyTorch remains visible.

## 5. AI Research

As a senior & rigorous AI researcher, always check with fact. Come up with hypothesis -> design experiments -> record metrics -> gain research signal -> further shape our research idea and directions.

Please document and keep updating experiments tables, each table should answer it's own question.

Design documentation system under docs/ and record how it's designed on this doc.

Focus on TOP AI venue's best papers' criteria, do mearningful work. Have research novelty. Keep idea tide and elegant, don't just integrate different ideas and think it'll be novel. Good idea should shape people's perspective. Do a great and thorough literature review and document it. Always keep updating but keep it concise and precise.

## 6. Collaboration with Company's Development Environment

You are currently running on Georgia Tech's PACE GPU cluster, and this repo is intended to develop for company's use. So when writing documentation on instructions of running on company's dev env, we should follow company's data storage system:

catagory | path | note
code | /user-volume/repo/<repo> | code should be store here
env | container image | reproducibility
small personal files | /user-volume | 50G, personal use; terminal logs go under `/user-volume/log/<experiment>`
shared project root | /group-volume/danny-dataset | Danny's project area in the shared volume
large datasets and runs | /group-volume/danny-dataset | datasets, teacher cache, W&B/TensorBoard files, runs, intermediate checkpoints, and exported weights

## 7. SAM2 Stage 1 Company Defaults

Stage 1 means encoder-only distillation from SAM2.1 teacher image features to a TinyViT-21M student image encoder with projection/adapters. This stage prepares teacher embeddings and a shape-compatible student feature interface; it does not train box-prompt masks or video memory.

Company runtime:
- Container image: `ngc24.06/ub22/py3.10/cu12.5/cudnn9.1/pytorch2.4`
- Keep the container PyTorch 2.4 runtime by default. Do not silently upgrade torch in setup scripts.
- Official current SAM2 may require torch >= 2.5.1, so every environment setup must run a SAM2 compatibility smoke test. If torch 2.4 fails, report it and either pin a compatible SAM2 commit or request a separate torch 2.5.1 company image.

Default paths:
- Company code root: `/user-volume/repo/SAM2-Distillation-Pipeline`
- Company GitHub repo: `https://github.com/thedannyliu/SAM2-Distillation-Pipeline.git`
- Official SAM2 upstream checkout: `/user-volume/repo/facebookresearch-sam2`
- Company data/checkpoint/cache root: `/group-volume/danny-dataset/sam2_distill`
- Company TensorBoard root: `/group-volume/danny-dataset/sam2_distill/logs`
- Company terminal log root: `/user-volume/log`
- Company final exported weight root: `/group-volume/danny-dataset/sam2_distill/checkpoints/final_weights`
- Company W&B project default: `sam2-distill-stage1`
- PACE scratch simulation root: `/storage/scratch1/9/eliu354/sam2_distill`
- Storage policy: datasets, teacher embeddings, TensorBoard/W&B run files, `last.pt`, `best.pt`, step checkpoints, and final exported weights live under `/group-volume/danny-dataset`. Small foreground terminal logs live under `/user-volume/log/<experiment>`.

PACE policy:
- PACE is for script, import, manifest, and tiny smoke validation only.
- Do not run full SA-1B teacher embedding cache or full Stage 1 training on PACE; put those compute-heavy jobs on the company cluster.

Multi-GPU teacher cache policy:
- Teacher embedding cache uses shard-level data parallelism, not gradient DDP.
- Use `scripts/company/03_cache_teacher_embeddings.sh --gpus 0,1,2,3` for single-node multi-GPU cache jobs.
- Use `--shard-ids 0-127` to assign an explicit shard range to one job.
- Use `tools/cache/plan_cache_shards.py --manifest <manifest> --shard-size 512 --num-jobs N` to compute shard ranges for multiple jobs/nodes.
- Under `torchrun`, each rank writes only shards where `shard_id % WORLD_SIZE == RANK`.
- For Slurm arrays, use `--start-shard $SLURM_ARRAY_TASK_ID --num-shards 1`.
- Cache writes use per-shard lock directories, so overlapping assignments skip locked shards instead of concurrently writing the same zarr shard.

COCO Stage 1 pilot:
- Company pilot root: `/group-volume/danny-dataset/sam2_distill`.
- Use exactly 1000 COCO train images and 100 COCO val images for the quick pilot.
- Use `docs/stage1/coco_pilot_2xh100.md` as the step-by-step runbook.
- Use `scripts/company/04_run_coco_stage1_pilot.sh prepare|cache|train|benchmark|all` on the company cluster.
- Store overlay mask benchmark results under `/group-volume/danny-dataset/sam2_distill/runs/stage1_coco_pilot/benchmark_val/overlays`.

Large-scale Stage 1 MSE speed run:
- Company root: `/group-volume/danny-dataset/sam2_distill`.
- Use `docs/stage1/large_scale_mse_8xh100.md` as the step-by-step runbook.
- Use `scripts/company/05_run_stage1_large_mse_8xh100.sh manifest|plan-cache|cache|train|all`.
- Default GPUs: `0,1,2,3,4,5,6,7`.
- Default objective: MSE on final `image_embed` plus MSE on high-resolution SAM2 features, with L1 and cosine disabled.
- Default train split is `train`; default validation split for SA-1B manifest is `val_sa1b`.
- Default stability settings: 2000 projection-only warmup steps, 2000 LR warmup steps, bf16 AMP, and grad clipping at norm 1.0.

Stage 1 data defaults:
- For the 8xH100 large-scale MSE run, use a deterministic fixed 3% SA-1B downloaded shard subset.
- Default SA-1B link list: `/group-volume/danny-dataset/SA-1B/sa1b_links.txt`
- Default 3% image root: `/group-volume/danny-dataset/SA-1B/images_3pct`
- Use `scripts/company/02_download_sa1b_subset.sh` to select/download/extract the 3% subset.
- The SA-1B downloader can fetch the link-list file directly with `SA1B_LINK_URL='<current Meta/fbcdn txt URL>'`; use `REFRESH_LINK_FILE=1` when replacing an expired link list.
- Default downloader selection: `SA1B_DOWNLOAD_PERCENT=3`, `SA1B_SELECTION_MODE=hash`, `KEEP_ARCHIVES=0`, `EXTRACT_ANNOTATIONS=0`.
- The downloader removes compressed archives after successful extraction and keeps reproducibility metadata under `/group-volume/danny-dataset/SA-1B/manifests/`.
- Manifest name: `sa1b_3pct_v1.parquet`
- Sampling seed: `sam2_stage1_sa1b_3pct_v1`
- Manifest should keep `SAMPLE_PERCENT=100` for the downloaded 3% image root.
- Validation split: `VAL_FRACTION=0.1`, producing roughly 90% `train` and 10% `val_sa1b`.
- Manifest fields: `sample_id`, `source`, `image_path`, `height`, `width`, `sha256`, `split`.
- Training checkpoints: `checkpoints/last.pt` is the resume checkpoint, and `checkpoints/best.pt` is selected by lowest `val/loss_stage1_total`.
- Intermediate training checkpoints stay under `/group-volume/danny-dataset/sam2_distill/runs/...`; copy the final selected checkpoint to `/group-volume/danny-dataset/sam2_distill/checkpoints/final_weights`.
- Stage 1 terminal/W&B/TensorBoard progress should include train/val image counts, global batch size, `train/images_seen`, `train/epoch`, `train/progress_pct`, `train/eta_hours`, `train/avg_wall_sec_per_step`, LR, grad norm, image MSE, high-res MSE, total loss, and validation loss.
- Stage 1 MSE losses use PyTorch mean reduction over all tensor elements. `loss_high_res_mse` is the sum of the separately averaged `high_res_s0` and `high_res_s1` MSE terms.

Stage 1 weights:
- Teacher primary: SAM2.1 Hiera Large checkpoint `sam2.1_hiera_large.pt`.
- Teacher smoke/fallback: SAM2.1 Hiera Base Plus checkpoint `sam2.1_hiera_base_plus.pt`.
- Student init: TinyViT-21M 512 distillation checkpoint from timm/Hugging Face, preferably downloaded with a no-login direct URL or manually mirrored if company blocks Hugging Face.
- Hugging Face login works on the company cluster. Minimal test:
  - `python -m pip install -U huggingface_hub`
  - `huggingface-cli login`
  - `huggingface-cli whoami`
- TinyViT HF fallback download:
  - `hf_hub_download(repo_id="timm/tiny_vit_21m_512.dist_in22k_ft_in1k", filename="model.safetensors")`
  - Copy the downloaded file to `/group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors`.

W&B company smoke test:
- `python -m pip install -U wandb`
- `wandb login`
- `wandb status`
- Minimal online run:
  - `python -c 'import wandb; r=wandb.init(project="sam2-distill-smoke", name="company-wandb-smoke"); wandb.log({"ok": 1}); r.finish()'`
- If network is blocked for a run, use `WANDB_MODE=offline` and later `wandb sync wandb/offline-run-*`.
- For repeatable HF/W&B login on new company nodes, follow `docs/company_auth_hf_wandb.md`.

Teacher cache defaults:
- SAM2 input size is 1024.
- Cache post-neck/post-`no_mem_embed` teacher features:
  - `image_embed`: fp16 `[N, 256, 64, 64]`
  - `high_res_s0`: fp16 `[N, 32, 256, 256]`
  - `high_res_s1`: fp16 `[N, 64, 128, 128]`
- Cache backend: zarr shards plus parquet shard indexes.

TinyViT student defaults:
- Use `timm.create_model("tiny_vit_21m_512.dist_in22k_ft_in1k", features_only=True, pretrained=False, checkpoint_path=<local safetensors>)`.
- Add projection/adapters so TinyViT feature dims and spatial sizes match the three teacher targets above.
- Stage 1 trainable modules: TinyViT encoder plus projection/adapters.
- Frozen modules: teacher all modules, SAM2 prompt encoder, SAM2 mask decoder, and memory modules.

## 8. Full SA-V Stage 2 Data Defaults

The current full-data memory, temporal, and multiplex experiments use the
complete prepared SA-V train release at its 6 FPS annotation cadence. This is
different from the older controlled cache that retained only 16 frames per
video.

Current company artifacts:
- Data Lake source: `s3://sdp-ril/danny-dataset/SA-V/sav_train`
- Audited raw train release: `/group-volume/danny-dataset/SA-V/sav_train`
- Full 6 FPS frame cache: `/group-volume/danny-dataset/SA-V/sav_train_6fps_full/JPEGImages`
- Full training manifest: `/group-volume/danny-dataset/sam2_distill/manifests/sav_train_6fps_full.parquet`
- Cohort root: `/group-volume/danny-dataset/sam2_distill/cohorts/sav_train_6fps_full`
- Full validation split: `/group-volume/danny-dataset/SA-V/sav_val`
- Full test split: `/group-volume/danny-dataset/SA-V/sav_test`
- Preparation runbook: `docs/stage2/sav_full_memory_train_data.md`
- Preparation entry points: `scripts/company/65_prepare_full_sav_memory_data.sh` and `scripts/company/66_prepare_full_sav_frames_4node.sh`

Audited release cardinality:
- 50,453 MP4 files
- 50,452 manual annotation JSON files
- 48,306 automatic annotation JSON files
- 50,337 videos usable by the SAM2 task adapter because they have a readable
  matching manual annotation
- The manifest retains all 50,453 video records; the task adapter excludes the
  116 MP4 IDs without matching readable manual annotations.

Training semantics:
- Set `MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/sav_train_6fps_full.parquet` explicitly for full-data experiments.
- Leave `TASK_VIDEO_IDS_FILE` empty for general memory training over every
  usable video.
- Use `dense4_unique.txt` for a more diverse multi-object cohort and
  `dense8_unique.txt` only for strict high-density multiplex experiments.
- The full-data expansion increases temporal coverage per video; it does not
  automatically increase the number of video samples in one epoch. With
  50,337 usable videos and four GPUs at batch size one per GPU, one epoch is
  approximately `ceil(50337 / 4) = 12,585` optimizer updates. T8/T16 still
  take longer per update than T4 because each sample decodes more frames.
- Do not claim data scaling from repeated dense-cohort IDs. Record whether an
  experiment uses all unique videos, dense4, dense8, or repeated sampling.
- For the SAM3.1-aligned multiplex curriculum, run full `sav_val` image and VOS
  evaluation after SMX1, SMX2, and SMX3. Run full `sav_test` only after SMX3.

Before launching a new full-data suite, run the preparation audit and treat it
as a hard input gate:

```bash
SAV_ROOT=/group-volume/danny-dataset/SA-V \
SAM2D_ROOT=/group-volume/danny-dataset/sam2_distill \
scripts/company/65_prepare_full_sav_memory_data.sh audit
```

The `/group-volume/danny-dataset` locations above are the writable company
project root for current and new experiments. Keep small foreground terminal
logs under `/user-volume/log/<experiment>` and do not introduce an unverified
`/danny-dataset` mount into commands or reproducibility records.

### TV21 EdgeTAM v1 data and folder contract

The first company EdgeTAM adaptation validates TinyViT-21M only. It retains
the selected TV21 image encoder, projection/adapters, prompt encoder, and
matched mask decoder, and replaces the temporal path with the released
EdgeTAM two-block memory attention and 2D Perceiver architecture. The frozen
online teacher is SAM2.1 Hiera Large.

Data contract:
- Train only on the complete prepared SA-V train release at its 6 FPS
  annotation cadence.
- Training manifest:
  `/group-volume/danny-dataset/sam2_distill/manifests/sav_train_6fps_full.parquet`.
- Frame/data root: `/group-volume/danny-dataset/SA-V`.
- Leave `TASK_VIDEO_IDS_FILE` empty to use all 50,337 videos that have a
  readable matching manual annotation.
- For the image re-anchor, sample annotated SA-V frames online. Do not write
  cropped images, duplicate frames, or teacher feature caches.
- First-round training does not use SA-1B, DAVIS, MOSE, or YTVOS. Record this
  as an explicit difference from the original EdgeTAM recipe.
- Use full `/group-volume/danny-dataset/SA-V/sav_val` for model selection and
  full `/group-volume/danny-dataset/SA-V/sav_test` only for the final selected
  checkpoint.

Read-only initialization artifacts:
- Selected TV21 task checkpoint:
  `/group-volume/danny-dataset/sam2_distill/runs/tinyvit_max_jf_v1/tv21/main/checkpoints/best.pt`.
- TinyViT initializer:
  `/group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors`.
- SAM2.1 Hiera-L teacher:
  `/group-volume/danny-dataset/sam2_distill/checkpoints/sam2.1/sam2.1_hiera_large.pt`.
- Released EdgeTAM temporal initializer:
  `/group-volume/danny-dataset/sam2_distill/checkpoints/edgetam/edgetam.pt`.

Writable experiment artifacts:
- Run root:
  `/group-volume/danny-dataset/sam2_distill/runs/edgetam_tv21_sam21l_v1`.
- Each formal stage stores its resumable `last.pt`, validation-selected
  `best.pt`, resolved config, W&B files, and evaluation outputs under its run
  directory. Reuse the same W&B run ID and directories when resuming.
- TensorBoard root:
  `/group-volume/danny-dataset/sam2_distill/logs/edgetam_tv21_sam21l_v1`.
- W&B project: `sam2-edgetam-tv21-sam21l-v1`.
- Foreground terminal logs:
  `/user-volume/log/edgetam_tv21_sam21l_v1`.
- Final selected/exported weights only:
  `/group-volume/danny-dataset/sam2_distill/checkpoints/final_weights/edgetam_tv21_sam21l_v1`.
- Batch-probe summary:
  `/group-volume/danny-dataset/sam2_distill/runs/edgetam_tv21_sam21l_v1/batch_probe/v1/batch_probe_summary.csv`.

Do not create `/danny-dataset`, write large artifacts directly under
`/user-volume`, or move historical selected checkpoints merely to make paths
look uniform.

## 9. Goal Implementation Tips

 Can submit several jobs in parallel, every GPU can be used to run smoke run. Don't waste time and token on keep checking the samme job.
 Keep the codebase concise, always make sure that each srcipt is neccesary.
 Can use documentation to track each experiment. Keep documenting.

## 10. Company Terminal Commands

- Company commands must run in the foreground and keep live output visible in the current terminal.
- Use `tee` when a persistent log is needed so output remains visible while it is recorded.
- Do not use `nohup`, `disown`, background `&`, detached `bash -lc`, or similar commands that detach work from the current terminal.
- Do not include `set -Eeuo pipefail` in commands pasted into the company terminal.
- Do not include `exit` in pasted command blocks. Report command status explicitly and return control to the same terminal prompt after completion or failure.
