# Script guide

`core/` is the main workflow; `lib/` contains shared drivers. `edgetam/` and
`deploy/` are specialized paths; `experiments/` retains ablations and run-specific
recipes. Do not launch all historical scripts as a setup sequence.

All in-repository callers have been updated. External automation must update
its paths using the map below; argument names and model behavior are unchanged.
Research outputs belong outside source control.

## Path migration

| Previous path | Current path |
| --- | --- |
| `scripts/company/00_setup_env.sh` | `scripts/core/setup_env.sh` |
| `scripts/company/01_download_weights.sh` | `scripts/core/download_weights.sh` |
| `scripts/company/02_download_sa1b_subset.sh` | `scripts/experiments/02_download_sa1b_subset.sh` |
| `scripts/company/03_cache_teacher_embeddings.sh` | `scripts/experiments/03_cache_teacher_embeddings.sh` |
| `scripts/company/04_run_coco_stage1_pilot.sh` | `scripts/experiments/04_run_coco_stage1_pilot.sh` |
| `scripts/company/05_run_stage1_large_mse_8xh100.sh` | `scripts/experiments/05_run_stage1_large_mse_8xh100.sh` |
| `scripts/company/06_download_sav_subset.sh` | `scripts/experiments/06_download_sav_subset.sh` |
| `scripts/company/07_run_davis_mask_finetune_1gpu.sh` | `scripts/experiments/07_run_davis_mask_finetune_1gpu.sh` |
| `scripts/company/08_run_sav_tinyvit_image_encoder_1h100.sh` | `scripts/experiments/08_run_sav_tinyvit_image_encoder_1h100.sh` |
| `scripts/company/09_run_sav000_005_epoch_timing.sh` | `scripts/experiments/09_run_sav000_005_epoch_timing.sh` |
| `scripts/company/10_run_sav_range_formal_image_encoder.sh` | `scripts/experiments/10_run_sav_range_formal_image_encoder.sh` |
| `scripts/company/11_run_sa1b_hf_online_teacher_stage1_21m.sh` | `scripts/core/stage1_distill_encoder.sh` |
| `scripts/company/12_benchmark_sav_val_prompts.sh` | `scripts/experiments/12_benchmark_sav_val_prompts.sh` |
| `scripts/company/13_download_hf_sa1b_archives_approx.sh` | `scripts/experiments/13_download_hf_sa1b_archives_approx.sh` |
| `scripts/company/14_benchmark_raw_sav_shard_sam2.sh` | `scripts/experiments/14_benchmark_raw_sav_shard_sam2.sh` |
| `scripts/company/15_benchmark_raw_sav_shard_suite.sh` | `scripts/experiments/15_benchmark_raw_sav_shard_suite.sh` |
| `scripts/company/16_benchmark_edgetam_bridge_raw_sav.sh` | `scripts/experiments/16_benchmark_edgetam_bridge_raw_sav.sh` |
| `scripts/company/17_download_edgetam_checkpoint.sh` | `scripts/edgetam/download_edgetam_weights.sh` |
| `scripts/company/18_prepare_sav_stage1_frame_cache.sh` | `scripts/core/data_prepare_frame_cache.sh` |
| `scripts/company/19_run_sav_stage1_ablation.sh` | `scripts/experiments/19_run_sav_stage1_ablation.sh` |
| `scripts/company/20_queue_sav_stage1_ablation_8gpu.sh` | `scripts/experiments/20_queue_sav_stage1_ablation_8gpu.sh` |
| `scripts/company/21_queue_sav_stage1_ablation_4gpu_size.sh` | `scripts/experiments/21_queue_sav_stage1_ablation_4gpu_size.sh` |
| `scripts/company/22_queue_sav_stage1_ablation_4gpu_loss.sh` | `scripts/experiments/22_queue_sav_stage1_ablation_4gpu_loss.sh` |
| `scripts/company/23_queue_sav_stage1_ablation_4gpu_adapter_teacher.sh` | `scripts/experiments/23_queue_sav_stage1_ablation_4gpu_adapter_teacher.sh` |
| `scripts/company/24_queue_sav_stage1_ablation_4gpu_extra.sh` | `scripts/experiments/24_queue_sav_stage1_ablation_4gpu_extra.sh` |
| `scripts/company/25_benchmark_stage1_sav_test.sh` | `scripts/lib/eval_sav_benchmark.sh` |
| `scripts/company/26_run_sam31_stage1_tv21.sh` | `scripts/experiments/26_run_sam31_stage1_tv21.sh` |
| `scripts/company/27_queue_sam31_4gpu_cosine.sh` | `scripts/experiments/27_queue_sam31_4gpu_cosine.sh` |
| `scripts/company/28_queue_sam31_4gpu_interface.sh` | `scripts/experiments/28_queue_sam31_4gpu_interface.sh` |
| `scripts/company/29_queue_sam31_4gpu_relations.sh` | `scripts/experiments/29_queue_sam31_4gpu_relations.sh` |
| `scripts/company/30_stage_complete_sav_in_group.sh` | `scripts/experiments/30_stage_complete_sav_in_group.sh` |
| `scripts/company/31_audit_mounted_sav_release.sh` | `scripts/experiments/31_audit_mounted_sav_release.sh` |
| `scripts/company/32_audit_stage1_run_progress.sh` | `scripts/experiments/32_audit_stage1_run_progress.sh` |
| `scripts/company/33_prepare_mounted_sav_stage1_manifest.sh` | `scripts/experiments/33_prepare_mounted_sav_stage1_manifest.sh` |
| `scripts/company/34_run_stage1_recovery_lane.sh` | `scripts/experiments/34_run_stage1_recovery_lane.sh` |
| `scripts/company/35_report_stage1_experiment_metrics.sh` | `scripts/experiments/35_report_stage1_experiment_metrics.sh` |
| `scripts/company/36_measure_sam2_hybrid_sizes.sh` | `scripts/experiments/36_measure_sam2_hybrid_sizes.sh` |
| `scripts/company/37_download_repvit_pretrained.sh` | `scripts/experiments/37_download_repvit_pretrained.sh` |
| `scripts/company/38_run_repvit_sam21l_stage1.sh` | `scripts/experiments/38_run_repvit_sam21l_stage1.sh` |
| `scripts/company/39_run_sam2_task_finetune_3stage.sh` | `scripts/lib/task_finetune_stages.sh` |
| `scripts/company/40_sync_sav_task_annotations_from_datalake.sh` | `scripts/core/data_sync_sav_annotations.sh` |
| `scripts/company/41_sync_sav_runtime_from_datalake.sh` | `scripts/core/data_sync_sav_runtime.sh` |
| `scripts/company/42_run_sam2_task_finetune_v2.sh` | `scripts/experiments/42_run_sam2_task_finetune_v2.sh` |
| `scripts/company/43_run_sam2_mask_finetune_ablation.sh` | `scripts/experiments/43_run_sam2_mask_finetune_ablation.sh` |
| `scripts/company/44_run_sam2_mask_finetune_ablation_v2.sh` | `scripts/experiments/44_run_sam2_mask_finetune_ablation_v2.sh` |
| `scripts/company/45_report_all_experiments.sh` | `scripts/core/report_experiments.sh` |
| `scripts/company/46_run_remaining_experiment_lane.sh` | `scripts/experiments/46_run_remaining_experiment_lane.sh` |
| `scripts/company/47_run_priority_mask_finetune_lane.sh` | `scripts/experiments/47_run_priority_mask_finetune_lane.sh` |
| `scripts/company/48_run_selected_continuation_lane.sh` | `scripts/experiments/48_run_selected_continuation_lane.sh` |
| `scripts/company/49_run_edgetam_memory_ablation.sh` | `scripts/lib/train_eval_engine.sh` |
| `scripts/company/50_run_edgetam_memory_lane.sh` | `scripts/experiments/50_run_edgetam_memory_lane.sh` |
| `scripts/company/51_run_edgetam_memory_recovery_lane.sh` | `scripts/experiments/51_run_edgetam_memory_recovery_lane.sh` |
| `scripts/company/52_run_tinyvit_max_jf.sh` | `scripts/core/stage2_finetune_tinyvit.sh` |
| `scripts/company/53_run_edgetam_official_fidelity.sh` | `scripts/edgetam/verify_official_identity.sh` |
| `scripts/company/54_prepare_eval_edgetam_e1.sh` | `scripts/experiments/54_prepare_eval_edgetam_e1.sh` |
| `scripts/company/55_run_edgetam_behavior_lane.sh` | `scripts/experiments/55_run_edgetam_behavior_lane.sh` |
| `scripts/company/56_run_backbone_task_expansion_lane.sh` | `scripts/experiments/stage2_capacity_selection.sh` |
| `scripts/company/57_run_weekend_72h_lane.sh` | `scripts/experiments/stage2_finetune_capacity_sweep.sh` |
| `scripts/company/58_run_tinyvit5_pseudolabel_lane.sh` | `scripts/experiments/58_run_tinyvit5_pseudolabel_lane.sh` |
| `scripts/company/59_run_sam2_multiobject_scaling.sh` | `scripts/deploy/benchmark_multiobject_buckets.sh` |
| `scripts/company/60_run_sam2_multiobject_training.sh` | `scripts/experiments/60_run_sam2_multiobject_training.sh` |
| `scripts/company/61_run_sam2_object_slots.sh` | `scripts/experiments/61_run_sam2_object_slots.sh` |
| `scripts/company/62_run_sam2_object_slots_v2.sh` | `scripts/experiments/62_run_sam2_object_slots_v2.sh` |
| `scripts/company/63_run_sam2_object_slots_v3.sh` | `scripts/experiments/63_run_sam2_object_slots_v3.sh` |
| `scripts/company/64_run_sam2_multiplex_overnight_v4.sh` | `scripts/experiments/64_run_sam2_multiplex_overnight_v4.sh` |
| `scripts/company/65_prepare_full_sav_memory_data.sh` | `scripts/core/data_prepare_sav.sh` |
| `scripts/company/66_prepare_full_sav_frames_4node.sh` | `scripts/core/data_prepare_sav_frames_4node.sh` |
| `scripts/company/67_run_sam2_full_data_50.sh` | `scripts/experiments/67_run_sam2_full_data_50.sh` |
| `scripts/company/68_run_sam2_priority_18.sh` | `scripts/experiments/68_run_sam2_priority_18.sh` |
| `scripts/company/69_run_tinyvit21_edgetam_memory_v1.sh` | `scripts/experiments/69_run_tinyvit21_edgetam_memory_v1.sh` |
| `scripts/company/70_run_sam2_tv_multiplex_v1.sh` | `scripts/experiments/70_run_sam2_tv_multiplex_v1.sh` |
| `scripts/company/71_probe_edgetam_tv21_batch.sh` | `scripts/edgetam/probe_batch_capacity.sh` |
| `scripts/company/72_run_edgetam_tv21_sam21l_v1.sh` | `scripts/edgetam/run_tv21_compressed_memory.sh` |
| `scripts/company/73_run_eventsam2_coast_screen.sh` | `scripts/experiments/73_run_eventsam2_coast_screen.sh` |
| `scripts/company/74_profile_sam2_tracking_components.sh` | `scripts/deploy/profile_tracking_components.sh` |
