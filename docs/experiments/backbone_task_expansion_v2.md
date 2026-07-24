# Backbone task fine-tuning expansion v2

## Question

Given the fixed 50,337-video usable SA-V training set, what is the best
task-fine-tuned result reachable by TinyViT-5M, TinyViT-11M,
TinyViT-21M, and the completed RepViT-M0.9 distillation checkpoint? Which
parameters should be protected before low-learning-rate joint tuning?

This suite extends, rather than duplicates, `tinyvit_max_jf_v1`.
The earlier lane tunes the image encoder first. The new controlled lane
starts by freezing the distilled encoder and adapting the mask decoder and
four-block SAM2 memory, then permits one low-learning-rate joint pass.

## Evidence shaping the design

- TinyViT-21M train-BN degraded test mIoU/AP relative to frozen BN
  (`0.8347/0.7143` versus approximately `0.8355–0.8358/0.7163–0.7168`).
  TinyViT BatchNorm therefore remains frozen.
- Decoder/memory-only tuning was competitive with end-to-end tuning, so
  protecting the distilled image representation is a high-value control.
- RepViT-M0.9 distill-only achieved only `37.5` test J&F and `0.5417`
  test mIoU. It needs image-task recovery before temporal refinement.
- RepViT is convolutional and BatchNorm-heavy. A one-epoch train-BN
  branch from the same recovered encoder is retained as an
  architecture-specific diagnostic; it is not assumed to win.

## TinyViT capacity lane

All three sizes use the same prompt simulation, frozen BatchNorm, T4
clips, W&B, and full SA-V val/test. Test never selects a model.

| Size | F1 | F2 | Total SA-V passes |
| --- | --- | --- | ---: |
| 5M | decoder + memory, encoder frozen, 2 epochs | full joint, low encoder LR `2e-7`, 1 epoch | 3 |
| 11M | decoder + memory, encoder frozen, 2 epochs | full joint, low encoder LR `1.5e-7`, 1 epoch | 3 |
| 21M | decoder + memory from A02, encoder frozen, 2 epochs | full joint, low encoder LR `1e-7`, 1 epoch | 3 |

The 5M and 11M initializers combine their best completed Stage-1 student
with the SAM2.1-L prompt, decoder, and memory modules. The 21M branch uses
A02 because a coherent task checkpoint already exists.

## RepViT recovery lane

The completed RepViT-M0.9 projection checkpoint is the sole RepViT source;
the unstarted M2.3 Stage-1 run is not silently substituted.

| Run | Start | Trainable scope | T / epochs | Question |
| --- | --- | --- | --- | --- |
| `repvit_P1_encoder_recovery_3ep` | distilled M0.9 | encoder | T2 / 3 | Can mask supervision recover the weak distilled image interface? |
| `repvit_P2_joint_frozenbn_2ep` | P1 | encoder + decoder + memory, BN frozen | T4 / 2 | Main RepViT task adaptation |
| `repvit_P2b_joint_trainbn_1ep` | P1 | same, BN trainable | T4 / 1 | Does RepViT-specific BN recalibration help or destabilize? |
| `repvit_P3_decmem_t8_refine_1ep` | frozen-BN P2 | decoder + memory | T8 / 1 | Can longer temporal context improve J&F without moving the encoder? |

P2 and P2b share P1, making the BN result interpretable. P3 deliberately
continues the safer frozen-BN branch regardless of test performance.

## Protocol and storage

- Hardware: one independent 4×H100 node per lane.
- Data: full audited SA-V train split; T8 uses `eligible_t8.txt`.
- Tracking: W&B online plus local training status.
- Formal order: train -> full `sav_val` -> full `sav_test` -> selection CSV.
- Selection: maximum full-val J&F, with mIoU and AP as tie-breakers.
- Storage: each completed stage retains exactly two physical checkpoint
  files, `last.pt` (trainer/resume state) and `best.pt` (portable task
  checkpoint). Compatibility names are symlinks.

Entry point:

`scripts/company/56_run_backbone_task_expansion_lane.sh tinyvit|repvit`

Run roots:

- `/group-volume/danny-dataset/sam2_distill/runs/tinyvit_capacity_freeze_v2`
- `/group-volume/danny-dataset/sam2_distill/runs/repvit_task_finetune_v2`

W&B projects:

- `tinyvit-capacity-freeze-v2`
- `repvit-task-finetune-v2`

The TinyViT lane queues nine train passes and six full evaluations. The
RepViT lane queues seven train passes and four full evaluations. Based on
the observed company runtimes, each is intentionally longer than a
20-hour allocation; the scripts resume in place if interrupted.
