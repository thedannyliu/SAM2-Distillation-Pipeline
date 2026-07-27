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

## Results through 2026-07-27

### Best result by TinyViT capacity

Selection uses full-val J&F across both `tinyvit_max_jf_v1` and the
capacity/freeze continuation. Test is descriptive only.

| Size | Selected run | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5M | `weekend_72h_v1/tv5_W1_decmem_t4_3ep` | 0.7997 | 0.6425 | **66.0** | 0.8025 | 0.6552 | 67.6 |
| 11M | `weekend_72h_v1/tv11_W1_decmem_t4_3ep` | 0.8141 | 0.6717 | **68.6** | 0.8161 | 0.6791 | 70.5 |
| 21M | `tinyvit_max_jf_v1/tv21` | 0.8371 | 0.7127 | **72.4** | 0.8353 | 0.7162 | 74.7 |

Relative to the best pre-task Stage-1 results, fine-tuning raises val J&F
by 1.8 points for 5M (64.2 to 66.0), 1.5 for 11M (67.1 to 68.6), and
1.3 for 21M (71.1 to 72.4). Relative to the best pre-task test J&F, the
descriptive test gains are 0.5, 1.0, and 0.5 points respectively. The
capacity curve remains orderly; fine-tuning provides consistent but
modest gains because the stronger TinyViT models already start near their
current-data ceiling.

Within the controlled freeze lane:

| Comparison | val J&F change | test J&F change | Interpretation |
| --- | ---: | ---: | --- |
| TV5 F1 -> F2 | +0.7 | -0.1 | low-LR joint tuning improves the selection split |
| TV11 F1 -> F2 | +1.1 | +0.3 | joint tuning helps, but remains below the earlier max-J&F lane |
| TV21 selected -> F1 | -1.2 | -0.5 | extra frozen decoder/memory training is unnecessary |
| TV21 selected -> F2 | -0.7 | -0.3 | low-LR joint tuning does not recover the gap |

The completed `tv21_F2_joint_low_1ep` control reaches 71.7 val and 74.4 test
J&F. It confirms the earlier prediction that this branch would not beat the
existing 72.4-val model.

The longer weekend continuations add only +0.2, +0.1, and +0.0 to the
validation-selected 5M, 11M, and 21M frontiers. More epochs within the same
task objective now look lower-value than changing data, supervision, or the
temporal interface.

### RepViT recovery and BatchNorm

| Run | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| distill-only M0.9 | 0.5664 | 0.2791 | 37.1 | 0.5417 | 0.2570 | 37.5 |
| P1 encoder recovery | 0.7481 | 0.5527 | 58.6 | 0.7446 | 0.5534 | 57.4 |
| P2 joint, frozen BN | 0.7557 | 0.5675 | 59.7 | 0.7541 | 0.5687 | 59.5 |
| P2b joint, train BN | 0.7162 | 0.4993 | 58.1 | 0.7069 | 0.4944 | 57.0 |
| P3 T8 decoder/memory | 0.7565 | 0.5683 | 60.3 | 0.7549 | 0.5699 | 60.1 |
| Weekend W3 decoder/memory | 0.7596 | 0.5733 | **61.4** | 0.7581 | 0.5762 | 60.0 |
| Weekend W6 joint-low | **0.7616** | **0.5781** | 60.9 | **0.7597** | **0.5775** | **61.1** |

Task fine-tuning rescues RepViT substantially. The validation-selected
weekend W3 gains 24.3 val and 22.5 test J&F over distill-only. Most recovery
comes from P1 image-encoder training; frozen-BN joint tuning adds 1.1/2.1
val/test J&F, P3 adds another 0.6/0.6, and the longer W3 decoder/memory stage
adds 1.1 val J&F over P3. W6's 61.1 test score is descriptive only because
its 60.9 val score does not win selection.

Trainable BatchNorm is decisively worse than the same-start frozen-BN
branch: P2b loses 1.6 val and 2.5 test J&F, 0.0395/0.0472 val/test mIoU,
and 0.0682/0.0743 AP. Together with the earlier TinyViT BN result, frozen
normalization is now the default across both backbone families under the
global-batch-four video regime.

Despite the recovery, validation-selected RepViT W3 remains 4.6 val J&F and
0.0401 val mIoU below the 5M TinyViT winner. Fine-tuning can rescue a weak
distillation interface, but does not erase backbone/distillation quality
differences.

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
