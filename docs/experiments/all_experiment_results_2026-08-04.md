# SAM2 distillation experiment ledger (2026-08-04)

This is the normalized, one-entry-per-experiment record derived from [`experiment_result_all.txt`](experiment_result_all.txt). The source is a read-only company-side snapshot generated at `2026-08-04T18:44:57Z`; statuses are historical and must not be interpreted as current process state.

## Scope and conventions

- Universal experiment rows: **181**.
- Evaluated rows reported by the source: **114**; started but not fully evaluated: **33**.
- Multi-object measurements: **140** rows across **35** aggregate files.
- Metric cells are `mIoU / AP / J&F`; latency cells are `image seconds / VOS seconds per video`.
- `-` means the source did not report the value. `progress=-` is unknown, not zero.
- The universal reporter collapsed seven `sam2_full_data_50_v1` parent directories to `main`. They remain distinct as `anonymous-main-NN` and are disambiguated by W&B ID; no variant name is guessed.
- For MX/MO/EM suites, names are restored by the stable row order corroborated by their latency/comparison paths.
- The captured `MO3` N=1 latency row lost p90/E2E/memory fields. Its N1 FPS is marked `*` and derived as `1000 / median_ms`; no other missing value is inferred.

## Snapshot summary

| Status | Count |
|---|---:|
| complete | 112 |
| final_checkpoint_incomplete | 23 |
| finalization_incomplete | 1 |
| not_started | 25 |
| superseded | 9 |
| training_incomplete | 1 |
| training_state_unknown | 7 |
| val_incomplete | 3 |

Key research signals in this snapshot:

- The selected TinyViT-21M reference remains **72.4 val / 74.7 test J&F**. The best recorded test J&F is **74.8**, reached by `tv21_S1_e2e_t4_continue_1ep` and `tv21_W3_selected_t8_2ep`, a +0.1-point change rather than a decisive gain.
- EdgeTAM official identity (`Q0`) preserves **68.3/69.5 J&F**, while the paper-scaled TinyViT recipe (`Q2`) reaches **55.6/58.0**. This is a real recovery over earlier TinyViT memory failures, but still only 81.4%/83.5% of Q0 val/test J&F.
- Persistent bucket execution is the strongest quality-preserving systems result: N=4/N=8 latency falls 29.6%/32.2%, with minimum per-mask IoU 0.9690. It failed the stricter binary-equivalence gate, so it is an engineering candidate, not a promoted result.
- Learned shared-K/V paths reach roughly 49 FPS at N=8, but full VOS J&F drops to about 60–64. The speed result is real; the representation/temporal-quality problem remains unresolved.
- None of MX13–MX28 passes the multiplex promotion gate in this snapshot. Completed screens retain only 41.7–75.8% of the MX5 quality cohort despite strong N=8 throughput.

## One-entry-per-experiment ledger

| ID | Suite | Experiment | Stage | Status | Progress % | Val mIoU/AP/J&F | Val image/VOS s | Test mIoU/AP/J&F | Test image/VOS s | W&B |
|---|---|---|---|---|---:|---|---|---|---|---|
| E001 | edgetam_fidelity_v3 | E0_official_upstream | - | training_state_unknown | - | 0.8224/0.6862/68.0000 | 0.0511/30.3483 | -/-/- | -/- | sd4v4ptj |
| E002 | edgetam_fidelity_v3 | E0_official_upstream_seed2 | - | training_state_unknown | - | -/-/- | -/- | -/-/- | -/- | - |
| E003 | edgetam_memory_ablation_v1 | M0_sam2_mem4 | main | complete | 100.00 | 0.8405/0.7167/71.5000 | 0.1178/38.1633 | 0.8391/0.7191/74.3000 | 0.1020/38.8936 | 8pxad6tr |
| E004 | edgetam_memory_ablation_v1 | M1_sam2_mem2 | main | complete | 100.00 | 0.8406/0.7167/53.3000 | 0.1113/37.2029 | 0.8391/0.7197/56.1000 | 0.0873/38.8279 | muha1isz |
| E005 | edgetam_memory_ablation_v1 | M2a_edgetam_hybrid2_official | main | complete | 100.00 | 0.8405/0.7166/15.6000 | 0.0867/36.8407 | 0.8391/0.7190/12.8000 | 0.0817/38.4183 | ow69rx39 |
| E006 | edgetam_memory_ablation_v1 | M2b_edgetam_hybrid2_current | main | complete | 100.00 | 0.8406/0.7167/13.2000 | 0.0846/38.8273 | 0.8391/0.7191/10.6000 | 0.0646/40.4279 | 2nfr05bt |
| E007 | edgetam_memory_ablation_v1 | R0_edgetam_e2e_t4_task | main | complete | 100.00 | 0.8369/0.7096/23.0000 | 0.0581/33.3275 | 0.8364/0.7137/21.5000 | 0.0581/40.1494 | c4xm8e29 |
| E008 | edgetam_memory_ablation_v1 | R1_edgetam_e2e_t4_imgkd | main | complete | 100.00 | 0.8379/0.7121/23.6000 | 0.0698/34.2270 | 0.8373/0.7165/21.7000 | 0.0562/35.0037 | urql35yg |
| E009 | edgetam_memory_ablation_v1 | R2_edgetam_e2e_t4_imgmemkd | main | complete | 100.00 | 0.8377/0.7117/25.3000 | 0.0623/30.0210 | 0.8374/0.7167/23.2000 | 0.0610/31.8370 | k2vhiyua |
| E010 | edgetam_memory_ablation_v1 | R3_edgetam_e2e_t8_imgmemkd | main | complete | 100.00 | 0.8374/0.7114/21.9000 | 0.0483/28.9706 | 0.8367/0.7157/19.1000 | 0.0457/30.2030 | 92phkgfc |
| E011 | edgetam_memory_recovery_v2 | C0_coherent_m0mem_align | main | final_checkpoint_incomplete | 100.00 | -/-/- | -/- | -/-/- | -/- | 1fqkzaub |
| E012 | edgetam_memory_recovery_v2 | C1_partial_m0mem_align | main | final_checkpoint_incomplete | 100.00 | -/-/- | -/- | -/-/- | -/- | 7ymgmv0a |
| E013 | edgetam_memory_recovery_v2 | C2_coherent_m0mem_joint2ep | main | final_checkpoint_incomplete | 100.00 | -/-/- | -/- | -/-/- | -/- | 808j2hls |
| E014 | edgetam_memory_recovery_v2 | C3_coherent_m0mem_staged | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E015 | edgetam_recipe_diagnostics_v5 | Q0_official_identity_t8_1ep | main | complete | 100.00 | 0.8224/0.6864/68.3000 | 0.0458/29.2401 | 0.8272/0.6990/69.5000 | 0.0473/30.6959 | z9aaydbl |
| E016 | edgetam_recipe_diagnostics_v5 | Q1_tinyvit_overfit16_t8_500ep | main | complete | 100.00 | 0.8404/0.7166/15.5000 | 0.0344/27.8558 | 0.8389/0.7189/15.5000 | 0.0353/30.1679 | za9k5aqu |
| E017 | edgetam_recipe_diagnostics_v5 | Q2_tinyvit_paper_scaled_sav_t8_5ep | main | complete | 100.00 | 0.8355/0.7065/55.6000 | 0.0554/29.8233 | 0.8343/0.7103/58.0000 | 0.0536/31.3072 | tkmm3hdk |
| E018 | edgetam_tinyvit21_behavior_v4 | D1_staged_image_align_1ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E019 | edgetam_tinyvit21_behavior_v4 | D2_staged_temporal_2ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E020 | edgetam_tinyvit21_behavior_v4 | D3_staged_t8_refine_1ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E021 | edgetam_tinyvit21_behavior_v4 | E1_a02_official_nonimage | main | complete | - | 0.0200/0.0001/2.1000 | 0.0517/30.7483 | 0.0151/0.0000/2.4000 | 0.0540/32.3150 | syggusw3 |
| E022 | edgetam_tinyvit21_behavior_v4 | J1_joint_behavior_2ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E023 | edgetam_tinyvit21_behavior_v4 | J2_joint_temporal_refine_1ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E024 | edgetam_tinyvit21_behavior_v4 | J3_joint_t8_refine_1ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E025 | edgetam_tinyvit21_behavior_v4 | S0_scratch_temporal_task_2ep | main | complete | 100.00 | 0.8402/0.7165/15.6000 | 0.0452/28.2092 | 0.8389/0.7190/13.8000 | 0.0457/29.6738 | rxrixqvf |
| E026 | edgetam_tinyvit21_behavior_v4 | S1_scratch_behavior_2ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E027 | edgetam_tinyvit21_behavior_v4 | S2_scratch_t8_refine_1ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E028 | repvit_stage1_v1 | repvit_m09_proj_sam21l_msehr_cos025_l1010 | - | finalization_incomplete | 100.00 | 0.5664/0.2791/37.1000 | 0.0387/36.7656 | 0.5417/0.2570/37.5000 | 0.0415/38.1890 | cg919yjs |
| E029 | repvit_stage1_v1 | repvit_m23_proj_sam21l_msehr_cos025_l1010 | - | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E030 | repvit_task_finetune_v2 | repvit_P1_encoder_recovery_3ep | repvit_P1_encoder_recovery_3ep | complete | 100.00 | 0.7481/0.5527/58.6000 | 0.0692/34.6661 | 0.7446/0.5534/57.4000 | 0.0693/36.3225 | 4bgxyj50 |
| E031 | repvit_task_finetune_v2 | repvit_P2_joint_frozenbn_2ep | repvit_P2_joint_frozenbn_2ep | complete | 100.00 | 0.7557/0.5675/59.7000 | 0.0728/34.8395 | 0.7541/0.5687/59.5000 | 0.0716/38.1917 | ugc4lcp5 |
| E032 | repvit_task_finetune_v2 | repvit_P2b_joint_trainbn_1ep | repvit_P2b_joint_trainbn_1ep | complete | 100.00 | 0.7162/0.4993/58.1000 | 0.0340/30.3690 | 0.7069/0.4944/57.0000 | 0.0348/31.7160 | s9htasm1 |
| E033 | repvit_task_finetune_v2 | repvit_P3_decmem_t8_refine_1ep | repvit_P3_decmem_t8_refine_1ep | complete | 100.00 | 0.7565/0.5683/60.3000 | 0.0337/30.4289 | 0.7549/0.5699/60.1000 | 0.0332/31.6438 | l29p2u8t |
| E034 | sam2_full_data_50_v1 | anonymous-main-01 | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | 2mjh5ut3 |
| E035 | sam2_full_data_50_v1 | anonymous-main-02 | - | training_state_unknown | - | -/-/- | -/- | -/-/- | -/- | ijid6zpj |
| E036 | sam2_full_data_50_v1 | anonymous-main-03 | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | 14up4h0x |
| E037 | sam2_full_data_50_v1 | anonymous-main-04 | - | training_state_unknown | - | -/-/- | -/- | -/-/- | -/- | nbuv9k6e |
| E038 | sam2_full_data_50_v1 | anonymous-main-05 | - | training_state_unknown | - | -/-/- | -/- | -/-/- | -/- | q18ycsg1 |
| E039 | sam2_full_data_50_v1 | anonymous-main-06 | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | m790int7 |
| E040 | sam2_full_data_50_v1 | anonymous-main-07 | - | training_state_unknown | - | -/-/- | -/- | -/-/- | -/- | j2r110uv |
| E041 | sam2_mask_finetune_ablation_v1 | decoder_lr2e6 | mask_decoder_lr2e6 | complete | 100.00 | 0.8374/0.7127/70.3000 | 0.0378/35.9077 | 0.8352/0.7161/70.6000 | 0.0401/37.5431 | fhqjlq9r |
| E042 | sam2_mask_finetune_ablation_v1 | decoder_lr2e7 | mask_decoder_lr2e7 | complete | 100.00 | 0.8375/0.7128/70.0000 | 0.0353/36.8992 | 0.8356/0.7161/71.5000 | 0.0370/38.4295 | h0guhxr9 |
| E043 | sam2_mask_finetune_ablation_v1 | decoder_lr5e7 | mask_decoder_lr5e7 | complete | 100.00 | 0.8376/0.7128/70.2000 | 0.0353/36.3961 | 0.8354/0.7162/71.4000 | 0.0364/36.6498 | ztx9qphr |
| E044 | sam2_mask_finetune_ablation_v1 | decoder_lr5e7_boxonly | mask_decoder_lr5e7_boxonly | complete | 100.00 | 0.8376/0.7128/70.3000 | 0.0347/36.5329 | 0.8357/0.7161/71.7000 | 0.0361/38.0166 | 8yu33386 |
| E045 | sam2_mask_finetune_ablation_v1 | encdec_low_frozenbn | mask_encdec_low_frozenbn | complete | 100.00 | 0.8374/0.7127/69.9000 | 0.0342/32.0464 | 0.8355/0.7163/71.2000 | 0.0363/33.6039 | h4nrpu0h |
| E046 | sam2_mask_finetune_ablation_v1 | encdec_low_trainbn | mask_encdec_low_trainbn | complete | 100.00 | 0.8343/0.7039/69.2000 | 0.0354/37.0937 | 0.8347/0.7143/71.2000 | 0.0374/38.3569 | vc4rfs0o |
| E047 | sam2_mask_finetune_ablation_v2 | A00_e2e_t4_box1_control | main | complete | 100.00 | 0.8376/0.7131/71.6000 | 0.0347/36.3856 | 0.8356/0.7164/73.7000 | 0.0386/39.2201 | mk5kspey |
| E048 | sam2_mask_finetune_ablation_v2 | A01_e2e_t4_box0 | main | complete | 100.00 | 0.8377/0.7132/71.3000 | 0.0352/35.8640 | 0.8355/0.7161/73.7000 | 0.0352/36.9851 | odjhtyua |
| E049 | sam2_mask_finetune_ablation_v2 | A02_e2e_t4_official_prompt | main | complete | 100.00 | 0.8374/0.7129/72.0000 | 0.0341/40.6596 | 0.8347/0.7151/74.1000 | 0.0366/36.8439 | 14kt70we |
| E050 | sam2_mask_finetune_ablation_v2 | A03_decmem_t4 | main | complete | 100.00 | 0.8377/0.7134/71.8000 | 0.0353/36.5912 | 0.8357/0.7167/73.4000 | 0.0369/38.0761 | vd7tfa7g |
| E051 | sam2_mask_finetune_ablation_v2 | A04_memory_t4 | main | complete | 100.00 | 0.8373/0.7128/71.7000 | 0.0345/36.5218 | 0.8353/0.7160/73.8000 | 0.0366/38.1244 | u9i4f7ai |
| E052 | sam2_mask_finetune_ablation_v2 | A05_e2e_t8 | main | complete | 100.00 | 0.8375/0.7132/71.9000 | 0.0342/35.8575 | 0.8355/0.7164/74.3000 | 0.0364/37.5059 | 4jsxjyz2 |
| E053 | sam2_mask_finetune_ablation_v2 | A06_e2e_t8_s4_t16_hard | refine_t16 | complete | 100.00 | 0.8377/0.7138/71.4000 | 0.0349/36.1557 | 0.8357/0.7169/73.9000 | 0.0367/37.9798 | oru10x1k |
| E054 | sam2_mask_finetune_ablation_v2 | A07_e2e_t4_warmup5 | main | complete | 100.00 | 0.8374/0.7131/71.3000 | 0.0347/36.6925 | 0.8356/0.7163/73.9000 | 0.0362/38.0487 | wldkp7po |
| E055 | sam2_mask_finetune_ablation_v2 | A08_e2e_t4_gb8 | main | complete | 100.00 | 0.8374/0.7133/71.9000 | 0.0351/36.4135 | 0.8355/0.7165/73.4000 | 0.0365/40.6399 | t3ncp5jf |
| E056 | sam2_mask_finetune_ablation_v2 | A09_e2e_t4_hard50x2 | main | complete | 100.00 | 0.8379/0.7137/70.8000 | 0.0351/35.5106 | 0.8357/0.7166/72.9000 | 0.0358/36.1398 | 5vtzjbbh |
| E057 | sam2_mask_finetune_ablation_v2 | A10_e2e_t4_box0_imgkd | main | complete | 100.00 | 0.8377/0.7134/71.3000 | 0.0350/35.1285 | 0.8361/0.7167/72.8000 | 0.0360/36.9194 | q132za5c |
| E058 | sam2_mask_finetune_ablation_v2 | A11_e2e_t4_box0_imgmemkd | main | complete | 100.00 | 0.8378/0.7143/71.3000 | 0.0337/34.7503 | 0.8358/0.7168/73.3000 | 0.0353/36.2101 | ismv3j75 |
| E059 | sam2_mask_finetune_ablation_v2 | smoke_A01_e2e_t4_box0_run327473-sam4-4 | - | val_incomplete | - | -/-/- | -/- | -/-/- | -/- | 89oyj04t |
| E060 | sam2_mask_finetune_ablation_v2 | smoke_A01_e2e_t4_box0_run327495-sam4-7 | - | val_incomplete | - | -/-/- | -/- | -/-/- | -/- | gfg3gl8a |
| E061 | sam2_multiobject_training_v1 | MO0_mem4_task_dense8_5ep | - | complete | - | 0.8410/0.7164/69.4000 | 0.0660/32.4679 | 0.8401/0.7227/72.4000 | 0.0543/32.7959 | 3xxqpml8 |
| E062 | sam2_multiobject_training_v1 | MO1_mem2_task_dense8_5ep | - | complete | - | 0.8406/0.7165/56.9000 | 0.0761/31.2964 | 0.8397/0.7213/58.5000 | 0.0696/32.7110 | uph3xhlj |
| E063 | sam2_multiobject_training_v1 | MO2_mem2_logits_dense8_5ep | - | complete | - | 0.8405/0.7166/57.1000 | 0.0564/29.2623 | 0.8397/0.7222/58.5000 | 0.0393/29.5809 | qzvamm3w |
| E064 | sam2_multiobject_training_v1 | MO3_mem2_memlogits_dense8_5ep | - | complete | - | 0.8406/0.7167/56.9000 | 0.0448/29.0419 | 0.8398/0.7213/58.4000 | 0.0436/30.5399 | 9d5el428 |
| E065 | sam2_multiplex_overnight_v4 | MX13_slot8_r2_mean_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | ecckuq0i |
| E066 | sam2_multiplex_overnight_v4 | MX14_slot8_r4_mean_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | 7yv0ol70 |
| E067 | sam2_multiplex_overnight_v4 | MX15_slot8_r8_mean_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | f9afgfko |
| E068 | sam2_multiplex_overnight_v4 | MX16_slot8_r16_mean_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | 2u94oa54 |
| E069 | sam2_multiplex_overnight_v4 | MX17_slot8_r8_mean_ptr4_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | wleft286 |
| E070 | sam2_multiplex_overnight_v4 | MX18_slot8_r8_mean_ptr8_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | nfhyoshy |
| E071 | sam2_multiplex_overnight_v4 | MX19_slot8_r8_latest_ptr8_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | 9qeb61dc |
| E072 | sam2_multiplex_overnight_v4 | MX20_slot8_r8_recency050_ptr8_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | npbuevkf |
| E073 | sam2_multiplex_overnight_v4 | MX21_slot8_r8_recency025_ptr8_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | gijv64tt |
| E074 | sam2_multiplex_overnight_v4 | MX22_slot8_r8_recency075_ptr8_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | x2mw6d65 |
| E075 | sam2_multiplex_overnight_v4 | MX23_slot4_r8_mean_ptr8_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | wr9i9lbn |
| E076 | sam2_multiplex_overnight_v4 | MX24_slot6_r8_mean_ptr8_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | 5dzhg6do |
| E077 | sam2_multiplex_overnight_v4 | MX25_slot8_r8_mean_ptr8_objkd025_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | xxi2gdan |
| E078 | sam2_multiplex_overnight_v4 | MX26_slot8_r8_mean_ptr8_objkd100_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | qamhsf1o |
| E079 | sam2_multiplex_overnight_v4 | MX27_slot8_min2_r8_mean_ptr8_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | n76h4ols |
| E080 | sam2_multiplex_overnight_v4 | MX28_slot8_min3_r8_mean_ptr8_screen3ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | m14lqqt6 |
| E081 | sam2_object_slots_v1 | MX1_slot4_decoder_kd_3ep | - | complete | - | 0.8403/0.7166/72.3000 | 0.0381/28.9992 | 0.8391/0.7191/74.6000 | 0.0335/30.5793 | 5twsaf73 |
| E082 | sam2_object_slots_v1 | MX2_slot8_decoder_kd_3ep | - | complete | - | 0.8403/0.7166/72.3000 | 0.0406/29.1831 | 0.8391/0.7191/74.6000 | 0.0386/30.7089 | d5yuknhq |
| E083 | sam2_object_slots_v1 | MX3_slot4_sharedkv_kd_3ep | - | complete | - | 0.8403/0.7166/63.8000 | 0.0414/29.7623 | 0.8391/0.7191/60.9000 | 0.0474/31.3841 | 6yar328i |
| E084 | sam2_object_slots_v1 | MX4_slot8_sharedkv_kd_3ep | - | complete | - | 0.8403/0.7166/63.6000 | 0.0416/29.2640 | 0.8391/0.7191/60.2000 | 0.0417/30.7515 | v4vog3bo |
| E085 | sam2_object_slots_v2 | MX5_slot8_decoder_t8_logits2_5ep | - | complete | - | 0.8403/0.7166/72.3000 | 0.0350/29.0093 | 0.8391/0.7191/74.6000 | 0.0353/30.5550 | 5mettlsu |
| E086 | sam2_object_slots_v2 | MX6_slot8_sharedkv_t8_mem1_5ep | - | complete | - | 0.8403/0.7166/63.6000 | 0.0387/29.1210 | 0.8391/0.7191/60.3000 | 0.0392/30.6492 | fgzts060 |
| E087 | sam2_object_slots_v2 | MX7_slot8_sharedkv_t8_mem4_5ep | - | complete | - | 0.8403/0.7166/63.7000 | 0.0437/29.4598 | 0.8391/0.7191/60.5000 | 0.0419/30.8623 | sbyfujsf |
| E088 | sam2_object_slots_v2 | MX8_slot8_sharedkv_t8_mem1_logits4_5ep | - | complete | - | 0.8403/0.7166/63.6000 | 0.0364/28.9142 | 0.8391/0.7191/60.3000 | 0.0376/30.3712 | jpeb2srz |
| E089 | sam2_task_finetune_tv21_v1 | stage1_encoder_task_2ep | stage1_encoder_task_2ep | complete | 100.00 | 0.8376/0.7129/70.3000 | 0.0372/29.7049 | 0.8356/0.7160/71.8000 | 0.0358/31.1089 | 331hstts |
| E090 | sam2_task_finetune_tv21_v1 | stage2_encoder_decoder_task_2ep | stage2_encoder_decoder_task_2ep | complete | 100.00 | 0.8379/0.7129/69.9000 | 0.0404/29.6395 | 0.8360/0.7167/71.3000 | 0.0360/31.1449 | fae9uebz |
| E091 | sam2_task_finetune_tv21_v1 | stage3_encoder_decoder_memory_task_1ep | stage3_encoder_decoder_memory_task_1ep | complete | 100.00 | 0.8381/0.7133/71.5000 | 0.0422/29.9859 | 0.8364/0.7171/74.3000 | 0.0402/31.3107 | 5ub0a9y2 |
| E092 | sam2_task_finetune_tv21_v2 | stage1_encoder_task_2ep_v2 | stage1_encoder_task_2ep_v2 | complete | 100.00 | 0.8373/0.7128/70.2000 | 0.0373/35.4660 | 0.8353/0.7160/71.3000 | 0.0387/35.3161 | rjjya960 |
| E093 | sam2_task_finetune_tv21_v2 | stage2_decoder_only_task_1ep_v2 | stage2_decoder_only_task_1ep_v2 | complete | 100.00 | 0.8374/0.7129/70.0000 | 0.0349/32.5604 | 0.8356/0.7161/71.3000 | 0.0371/34.0559 | rso2ko7v |
| E094 | sam2_task_finetune_tv21_v2 | stage3_encoder_decoder_memory_task_1ep_v2 | stage3_encoder_decoder_memory_task_1ep_v2 | complete | 100.00 | 0.8380/0.7131/71.7000 | 0.0356/32.7550 | 0.8356/0.7163/73.9000 | 0.0376/34.9745 | n1zib7mq |
| E095 | sam31_stage1 | tv21m_adapter_mse_cos025_5ep_v1 | - | superseded | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E096 | sam31_stage1 | tv21m_adapter_mse_cos025_5ep_v1_smoke | - | superseded | 100.00 | -/-/- | -/- | -/-/- | -/- | - |
| E097 | sam31_stage1_ablation_v1 | n1_cos000_adapter_ft_w2k | - | complete | 100.00 | 0.7011/0.4453/25.5000 | 0.3034/33.7622 | 0.6846/0.4310/25.3000 | 0.3812/34.1967 | k0othn97 |
| E098 | sam31_stage1_ablation_v1 | n1_cos025_adapter_ft_w2k | - | complete | 100.00 | 0.7015/0.4467/25.5000 | 0.3349/32.9352 | 0.6855/0.4326/25.3000 | 0.3785/32.3296 | p3iow86e |
| E099 | sam31_stage1_ablation_v1 | n1_cos100_adapter_ft_w2k | - | complete | 100.00 | 0.7034/0.4499/25.5000 | 0.3437/33.9116 | 0.6867/0.4360/25.3000 | 0.3552/34.0578 | rttzqjta |
| E100 | sam31_stage1_ablation_v1 | n2_adapter_cos025_frozen | - | complete | 100.00 | 0.4601/0.1572/25.5000 | 0.3199/31.3783 | 0.4550/0.1597/25.4000 | 0.3594/29.6099 | dheevqnm |
| E101 | sam31_stage1_ablation_v1 | n2_adapter_cos025_ft_w0 | - | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E102 | sam31_stage1_ablation_v1 | n2_projection_cos025_ft_w2k | - | complete | 100.00 | 0.6999/0.4441/25.5000 | 0.2952/33.0533 | 0.6823/0.4291/25.3000 | 0.3598/33.9058 | 7ll2x8qv |
| E103 | sam31_stage1_ablation_v1 | n3_cos025_relation010_adapter_ft_w2k | - | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E104 | sam31_stage1_ablation_v1 | n3_cos150_adapter_ft_w2k | - | training_incomplete | 20.00 | -/-/- | -/- | -/-/- | -/- | yhwfo56j |
| E105 | sam31_stage1_ablation_v1 | n3_relation010_adapter_ft_w2k | - | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E106 | sav_stage1_ablation_v2 | tv11_adapter_sam21l_msehr | - | complete | 100.00 | 0.8029/0.6514/65.6000 | 0.0166/26.9673 | 0.8054/0.6635/69.0000 | 0.0169/31.7065 | n0npj42x |
| E107 | sav_stage1_ablation_v2 | tv11_proj_sam21l_msehr | - | complete | 100.00 | 0.8095/0.6617/66.8000 | 0.0997/36.4113 | 0.8128/0.6723/69.4000 | 0.0929/37.2279 | 7woradoj |
| E108 | sav_stage1_ablation_v2 | tv11_proj_sam21l_msehr_cos025 | - | complete | 100.00 | 0.8095/0.6619/67.1000 | 0.0446/31.4598 | 0.8126/0.6725/69.5000 | 0.0464/32.9804 | ci6juty3 |
| E109 | sav_stage1_ablation_v2 | tv21_adapter_sam21l_msehr | - | complete | 100.00 | 0.8301/0.7020/71.1000 | 0.1062/34.4817 | 0.8306/0.7063/72.9000 | 0.1148/36.2079 | xp8aaheh |
| E110 | sav_stage1_ablation_v2 | tv21_adapter_sam21l_msehr_cos025 | - | complete | 100.00 | 0.8347/0.7078/70.5000 | 0.0343/34.7888 | 0.8359/0.7171/74.2000 | 0.0359/36.0345 | hlg1hh8a |
| E111 | sav_stage1_ablation_v2 | tv21_proj_sam21bplus_msehr | - | val_incomplete | 100.00 | -/-/- | -/- | -/-/- | -/- | 9xjiaa9u |
| E112 | sav_stage1_ablation_v2 | tv21_proj_sam21l_hr025 | - | complete | 100.00 | 0.8352/0.7088/70.4000 | 0.0402/29.9281 | 0.8363/0.7175/73.5000 | 0.0396/31.3156 | w3kixzum |
| E113 | sav_stage1_ablation_v2 | tv21_proj_sam21l_image_only | - | complete | 100.00 | 0.7885/0.6297/68.5000 | 0.0640/32.6563 | 0.7836/0.6247/70.2000 | 0.0605/34.4005 | 4w1ensds |
| E114 | sav_stage1_ablation_v2 | tv21_proj_sam21l_msehr | - | complete | 100.00 | 0.8307/0.7010/70.6000 | 0.0976/34.3292 | 0.8319/0.7089/73.9000 | 0.1099/36.4368 | fovv6o9x |
| E115 | sav_stage1_ablation_v2 | tv21_proj_sam21l_msehr_cos025 | - | complete | 100.00 | 0.8314/0.7016/70.5000 | 0.1077/34.5654 | 0.8325/0.7096/73.7000 | 0.1126/36.1136 | 8blojdod |
| E116 | sav_stage1_ablation_v2 | tv21_proj_sam21l_msehr_cos1 | - | complete | 100.00 | 0.8338/0.7057/70.4000 | 0.0488/38.6757 | 0.8346/0.7121/73.8000 | 0.0488/40.3627 | f4btay3j |
| E117 | sav_stage1_ablation_v2 | tv21_proj_sam21l_msehr_l1_025 | - | complete | 100.00 | 0.8354/0.7084/70.9000 | 0.0353/29.6253 | 0.8359/0.7171/73.8000 | 0.0332/33.7039 | jqwvvf6p |
| E118 | sav_stage1_ablation_v2 | tv5_adapter_sam21l_msehr | - | complete | 100.00 | 0.7951/0.6371/64.2000 | 0.0472/33.1468 | 0.7976/0.6465/67.1000 | 0.0472/34.5249 | tzy3sdv4 |
| E119 | sav_stage1_ablation_v2 | tv5_proj_sam21l_msehr | - | complete | 100.00 | 0.7928/0.6321/64.2000 | 0.0469/31.4687 | 0.7971/0.6458/66.5000 | 0.0464/33.0099 | dbnlshx9 |
| E120 | sav_stage1_ablation_v2 | tv5_proj_sam21l_msehr_cos025 | - | complete | 100.00 | 0.7923/0.6318/63.7000 | 0.1096/35.6212 | 0.7958/0.6444/66.2000 | 0.1001/36.7279 | t76p32bz |
| E121 | stage1_online_teacher_sav000_018_vbal32_tv11m_4gpu_b16_mse_cos_5ep_v1 | stage1_online_teacher_sav000_018_vbal32_tv11m_4gpu_b16_mse_cos_5ep_v1 | - | superseded | 100.00 | -/-/- | -/- | -/-/- | -/- | - |
| E122 | stage1_online_teacher_sav000_018_vbal32_tv11m_8gpu_b8_mse_only_5ep_v1 | stage1_online_teacher_sav000_018_vbal32_tv11m_8gpu_b8_mse_only_5ep_v1 | - | superseded | 100.00 | -/-/- | -/- | -/-/- | -/- | - |
| E123 | stage1_online_teacher_sav000_018_vbal32_tv21m_4gpu_b4_highres_only_5ep_v1 | stage1_online_teacher_sav000_018_vbal32_tv21m_4gpu_b4_highres_only_5ep_v1 | - | superseded | 40.02 | -/-/- | -/- | -/-/- | -/- | - |
| E124 | stage1_online_teacher_sav000_018_vbal32_tv21m_4gpu_b4_mse_cos_5ep_v1 | stage1_online_teacher_sav000_018_vbal32_tv21m_4gpu_b4_mse_cos_5ep_v1 | - | superseded | 30.00 | -/-/- | -/- | -/-/- | -/- | - |
| E125 | stage1_online_teacher_sav000_018_vbal32_tv21m_8gpu_b4_mse_only_5ep_v1 | stage1_online_teacher_sav000_018_vbal32_tv21m_8gpu_b4_mse_only_5ep_v1 | - | superseded | 50.00 | -/-/- | -/- | -/-/- | -/- | - |
| E126 | stage1_online_teacher_sav000_018_vbal32_tv5m_4gpu_b32_mse_cos_5ep_v1 | stage1_online_teacher_sav000_018_vbal32_tv5m_4gpu_b32_mse_cos_5ep_v1 | - | superseded | 100.00 | -/-/- | -/- | -/-/- | -/- | - |
| E127 | stage1_online_teacher_sav000_018_vbal32_tv5m_4gpu_b32_mse_only_5ep_v1 | stage1_online_teacher_sav000_018_vbal32_tv5m_4gpu_b32_mse_only_5ep_v1 | - | superseded | 100.00 | -/-/- | -/- | -/-/- | -/- | - |
| E128 | tinyvit21_edgetam_memory_v1 | EM1_t4_official_temporal_2ep | - | final_checkpoint_incomplete | - | -/-/- | -/- | -/-/- | -/- | xgw90n7f |
| E129 | tinyvit21_edgetam_memory_v1 | EM2_t8_joint_edgetam_5ep | - | training_state_unknown | - | -/-/- | -/- | -/-/- | -/- | ftg434pq |
| E130 | tinyvit5_pseudolabel_v1 | tv5_PL0_gt_t4_3ep | tv5_PL0_gt_t4_3ep | complete | 100.00 | 0.8004/0.6446/66.0000 | 0.0376/29.9622 | 0.8031/0.6570/67.3000 | 0.0346/31.1594 | hfxebhnn |
| E131 | tinyvit5_pseudolabel_v1 | tv5_PL1_sam21l_soft025_t4_3ep | tv5_PL1_sam21l_soft025_t4_3ep | complete | 100.00 | 0.7998/0.6441/65.9000 | 0.0326/29.6724 | 0.8031/0.6571/68.1000 | 0.0333/31.1025 | 125i31ph |
| E132 | tinyvit5_pseudolabel_v1 | tv5_PL2_sam21l_soft050_t4_3ep | tv5_PL2_sam21l_soft050_t4_3ep | complete | 100.00 | 0.7999/0.6436/65.6000 | 0.0445/30.0294 | 0.8033/0.6580/68.0000 | 0.0378/31.1568 | 6j2yslqy |
| E133 | tinyvit5_pseudolabel_v1 | tv5_PL3_selected_t8_2ep | tv5_PL3_selected_t8_2ep | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E134 | tinyvit_capacity_freeze_v2 | tv11_F1_decmem_frozen_2ep | tv11_F1_decmem_frozen_2ep | complete | 100.00 | 0.8125/0.6678/66.7000 | 0.0504/31.3331 | 0.8156/0.6784/69.6000 | 0.0450/32.1507 | rhm80bk5 |
| E135 | tinyvit_capacity_freeze_v2 | tv11_F2_joint_low_1ep | tv11_F2_joint_low_1ep | complete | 100.00 | 0.8131/0.6691/67.8000 | 0.0503/31.6263 | 0.8154/0.6792/69.9000 | 0.0464/32.0823 | s1jihjjx |
| E136 | tinyvit_capacity_freeze_v2 | tv21_F1_decmem_frozen_2ep | tv21_F1_decmem_frozen_2ep | complete | 100.00 | 0.8374/0.7129/71.2000 | 0.0461/30.7076 | 0.8355/0.7167/74.2000 | 0.0460/32.1522 | lm3xsscz |
| E137 | tinyvit_capacity_freeze_v2 | tv21_F2_joint_low_1ep | tv21_F2_joint_low_1ep | complete | 100.00 | 0.8378/0.7138/71.7000 | 0.0476/30.7225 | 0.8356/0.7165/74.4000 | 0.0462/32.3508 | y8xjlec2 |
| E138 | tinyvit_capacity_freeze_v2 | tv5_F1_decmem_frozen_2ep | tv5_F1_decmem_frozen_2ep | complete | 100.00 | 0.7985/0.6426/65.1000 | 0.1131/36.6279 | 0.8016/0.6509/67.7000 | 0.1603/38.0709 | ywur6yme |
| E139 | tinyvit_capacity_freeze_v2 | tv5_F2_joint_low_1ep | tv5_F2_joint_low_1ep | complete | 100.00 | 0.7997/0.6426/65.8000 | 0.0463/30.8790 | 0.8022/0.6523/67.6000 | 0.0463/33.4098 | k8a3c0wb |
| E140 | tinyvit_max_jf_v1 | tv11 | selected_best | complete | - | 0.8138/0.6714/68.5000 | 0.0634/33.1316 | 0.8157/0.6794/70.3000 | 0.0666/36.1215 | bb6vcv3d |
| E141 | tinyvit_max_jf_v1 | tv11_S1_encoder_t2_2ep | - | complete | - | 0.8145/0.6712/67.2000 | 0.0692/33.4400 | 0.8161/0.6796/70.5000 | 0.0679/34.6438 | uyo3cg53 |
| E142 | tinyvit_max_jf_v1 | tv11_S2_e2e_t4_low_1ep | - | complete | - | 0.8137/0.6703/67.8000 | 0.0715/33.5622 | 0.8153/0.6790/70.5000 | 0.0698/35.0101 | ozudq8bi |
| E143 | tinyvit_max_jf_v1 | tv11_S3_decmem_t4_refine_1ep | - | complete | - | 0.8138/0.6714/68.5000 | 0.0634/33.1316 | 0.8157/0.6794/70.3000 | 0.0666/36.1215 | bb6vcv3d |
| E144 | tinyvit_max_jf_v1 | tv21 | selected_best | complete | - | 0.8371/0.7127/72.4000 | 0.0319/30.4431 | 0.8353/0.7162/74.7000 | 0.0409/31.5906 | rvjh6zfe |
| E145 | tinyvit_max_jf_v1 | tv21_S1_e2e_t4_continue_1ep | - | complete | - | 0.8370/0.7128/72.2000 | 0.0582/40.4597 | 0.8350/0.7162/74.8000 | 0.0599/41.4881 | w4z6lg3p |
| E146 | tinyvit_max_jf_v1 | tv21_S2_e2e_t4_low_1ep | - | complete | - | 0.8372/0.7128/71.8000 | 0.0410/30.1991 | 0.8353/0.7162/74.7000 | 0.0406/31.1600 | h0nyosb6 |
| E147 | tinyvit_max_jf_v1 | tv21_S3_decmem_t4_refine_1ep | - | complete | - | 0.8371/0.7127/72.4000 | 0.0319/30.4431 | 0.8353/0.7162/74.7000 | 0.0409/31.5906 | rvjh6zfe |
| E148 | tinyvit_max_jf_v1 | tv5 | selected_best | complete | - | 0.8012/0.6461/65.3000 | 0.1135/36.2813 | 0.8037/0.6568/67.9000 | 0.1379/38.3224 | 8f0ba0pf |
| E149 | tinyvit_max_jf_v1 | tv5_S1_encoder_t2_2ep | - | complete | - | 0.8016/0.6463/65.2000 | 0.0378/30.3016 | 0.8038/0.6581/67.8000 | 0.0391/31.5691 | s60leoc9 |
| E150 | tinyvit_max_jf_v1 | tv5_S2_e2e_t4_low_1ep | - | complete | - | 0.8014/0.6460/65.1000 | 0.0536/33.1267 | 0.8034/0.6576/67.9000 | 0.0540/34.7423 | 7agzwkel |
| E151 | tinyvit_max_jf_v1 | tv5_S3_decmem_t4_refine_1ep | - | complete | - | 0.8012/0.6461/65.3000 | 0.1135/36.2813 | 0.8037/0.6568/67.9000 | 0.1379/38.3224 | 8f0ba0pf |
| E152 | weekend_72h_v1 | K1a_m0_task_5ep | main | complete | 100.00 | 0.8404/0.7165/38.7000 | 0.0445/28.6094 | 0.8389/0.7190/39.0000 | 0.0433/30.1974 | bca11wka |
| E153 | weekend_72h_v1 | K1b_m0_logits_5ep | main | complete | 100.00 | 0.8404/0.7165/38.7000 | 0.0503/29.0252 | 0.8389/0.7190/38.6000 | 0.0501/30.7018 | cmeu731b |
| E154 | weekend_72h_v1 | K1c_m0_memlogits_5ep | main | complete | 100.00 | 0.8404/0.7165/38.5000 | 0.0434/27.8061 | 0.8389/0.7189/38.8000 | 0.0399/29.2678 | ptvgqi56 |
| E155 | weekend_72h_v1 | K1d_m0_full_5ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E156 | weekend_72h_v1 | K2a_m0_task_t8_2ep | main | complete | 100.00 | 0.8404/0.7160/42.1000 | 0.0519/33.6382 | 0.8389/0.7190/40.0000 | 0.0529/30.7377 | 7fr4d79g |
| E157 | weekend_72h_v1 | K2b_m0_logits_t8_2ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E158 | weekend_72h_v1 | K2c_m0_memlogits_t8_2ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E159 | weekend_72h_v1 | K2d_m0_full_t8_2ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E160 | weekend_72h_v1 | W1_official_image_align_2ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E161 | weekend_72h_v1 | W2a_official_logits_5ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E162 | weekend_72h_v1 | W2b_official_memlogits_5ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E163 | weekend_72h_v1 | W2c_official_full_5ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E164 | weekend_72h_v1 | W3a_official_logits_t8_3ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E165 | weekend_72h_v1 | W3b_official_memlogits_t8_3ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E166 | weekend_72h_v1 | W3c_official_full_t8_3ep | main | not_started | 0.00 | -/-/- | -/- | -/-/- | -/- | - |
| E167 | weekend_72h_v1 | repvit_W1_encoder_t2_5ep | repvit_W1_encoder_t2_5ep | complete | 100.00 | 0.7610/0.5777/60.9000 | 0.0576/34.9615 | 0.7590/0.5774/60.2000 | 0.0658/37.6474 | kc8evzut |
| E168 | weekend_72h_v1 | repvit_W2_decoder_t2_5ep | repvit_W2_decoder_t2_5ep | complete | 100.00 | 0.7596/0.5738/60.4000 | 0.0327/30.4041 | 0.7578/0.5755/59.2000 | 0.0333/31.7295 | 1nc3oft5 |
| E169 | weekend_72h_v1 | repvit_W3_decmem_t4_5ep | repvit_W3_decmem_t4_5ep | complete | 100.00 | 0.7596/0.5733/61.4000 | 0.0350/33.7664 | 0.7581/0.5762/60.0000 | 0.0339/31.7160 | a8uybcgl |
| E170 | weekend_72h_v1 | repvit_W4_joint_t4_5ep | repvit_W4_joint_t4_5ep | complete | 100.00 | 0.7613/0.5780/60.9000 | 0.0337/30.4422 | 0.7605/0.5789/60.4000 | 0.0349/31.7166 | 507sfftk |
| E171 | weekend_72h_v1 | repvit_W5_selected_t8_3ep | repvit_W5_selected_t8_3ep | complete | 100.00 | 0.7598/0.5740/60.4000 | 0.0471/30.9195 | 0.7580/0.5766/60.9000 | 0.0441/32.8488 | 3jk3mmm1 |
| E172 | weekend_72h_v1 | repvit_W6_joint_low_t4_3ep | repvit_W6_joint_low_t4_3ep | complete | 100.00 | 0.7616/0.5781/60.9000 | 0.0305/31.9703 | 0.7597/0.5775/61.1000 | 0.0320/33.2961 | w49p3bol |
| E173 | weekend_72h_v1 | tv11_W1_decmem_t4_3ep | tv11_W1_decmem_t4_3ep | complete | 100.00 | 0.8141/0.6717/68.6000 | 0.0469/30.7140 | 0.8161/0.6791/70.5000 | 0.0457/32.1338 | hyn8zq0d |
| E174 | weekend_72h_v1 | tv11_W2_joint_t4_3ep | tv11_W2_joint_t4_3ep | complete | 100.00 | 0.8142/0.6717/68.0000 | 0.0461/31.8576 | 0.8163/0.6802/70.6000 | 0.0520/32.2837 | alcdaikn |
| E175 | weekend_72h_v1 | tv11_W3_selected_t8_2ep | tv11_W3_selected_t8_2ep | complete | 100.00 | 0.8140/0.6718/68.2000 | 0.0464/30.9381 | 0.8160/0.6803/69.7000 | 0.0459/32.1517 | dz8ttkkq |
| E176 | weekend_72h_v1 | tv21_W1_decmem_t4_3ep | tv21_W1_decmem_t4_3ep | complete | 100.00 | 0.8372/0.7130/71.7000 | 0.0479/31.9554 | 0.8357/0.7166/74.5000 | 0.0464/32.0183 | ip2oxq8j |
| E177 | weekend_72h_v1 | tv21_W2_joint_t4_3ep | tv21_W2_joint_t4_3ep | complete | 100.00 | 0.8373/0.7131/71.5000 | 0.0444/31.6897 | 0.8354/0.7165/74.4000 | 0.0445/33.6671 | jtiz8js7 |
| E178 | weekend_72h_v1 | tv21_W3_selected_t8_2ep | tv21_W3_selected_t8_2ep | complete | 100.00 | 0.8374/0.7132/72.0000 | 0.0418/32.4380 | 0.8357/0.7166/74.8000 | 0.0423/33.4526 | 38j3u0jq |
| E179 | weekend_72h_v1 | tv5_W1_decmem_t4_3ep | tv5_W1_decmem_t4_3ep | complete | 100.00 | 0.7997/0.6425/66.0000 | 0.0476/30.8584 | 0.8025/0.6552/67.6000 | 0.0467/32.3247 | f6h5wrv0 |
| E180 | weekend_72h_v1 | tv5_W2_joint_t4_3ep | tv5_W2_joint_t4_3ep | complete | 100.00 | 0.8000/0.6436/65.8000 | 0.0464/30.8608 | 0.8028/0.6559/67.8000 | 0.0467/32.5260 | lrxcziw0 |
| E181 | weekend_72h_v1 | tv5_W3_selected_t8_2ep | tv5_W3_selected_t8_2ep | complete | 100.00 | 0.7996/0.6429/65.4000 | 0.0457/34.0834 | 0.8027/0.6539/67.9000 | 0.0480/33.3821 | 8yyl1sml |

## Multi-object latency ledger

Each row is one `aggregate.csv`. FPS columns are propagation FPS. The final columns retain N=8 median/p90 latency, peak memory, and the source gate flags for N=2/4/8.

| ID | Suite/variant | Samples×videos | N1 FPS | N2 FPS | N4 FPS | N8 FPS | N8 median/p90 ms | N8 peak MB | Gate N2/N4/N8 |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| L001 | sam2_full_data_50_v1/FD01_tv21_t4_decmem_5ep | 3×1 | 57.24 | 47.16 | 37.08 | 23.96 | 41.74/41.75 | 12521 | 0/0/0 |
| L002 | sam2_full_data_50_v1/FD46_mem4_t8_decmem_8ep | 3×1 | 56.84 | 47.66 | 36.38 | 24.09 | 41.52/41.57 | 12531 | 0/0/0 |
| L003 | sam2_multiobject_bucket_mx1_v1/tv21_best/point_n1-2-4-8 | 2×1 | 72.86 | 45.89 | 27.24 | 14.15 | 70.69/71.72 | 12207 | 0/0/0 |
| L004 | sam2_multiobject_bucket_mx1_v1/tv21_best/point_n1-2-4-8_bucket4 | 2×1 | 35.82 | 35.49 | 28.54 | 15.71 | 63.64/64.12 | 17862 | 1/0/0 |
| L005 | sam2_multiobject_bucket_mx1p_v1/tv21_best/point_n1-2-4-8 | 2×1 | 72.47 | 47.68 | 26.83 | 14.96 | 66.83/67.10 | 12207 | 0/0/0 |
| L006 | sam2_multiobject_bucket_mx1p_v1/tv21_best/point_n1-2-4-8_bucket4_persistent_m4 | 2×1 | 71.36 | 47.15 | 38.09 | 22.07 | 45.30/45.31 | 12791 | 0/0/0 |
| L007 | sam2_multiobject_scaling_v1/tv21_best/point_n1-2-4-8 | 2×1 | 72.57 | 48.17 | 27.64 | 15.02 | 66.68/68.55 | 12207 | 0/0/0 |
| L008 | sam2_multiobject_training_v1/MO0_mem4_task_dense8_5ep | 2×1 | 57.05 | 47.91 | 36.68 | 24.16 | 41.39/41.57 | 12551 | 0/0/0 |
| L009 | sam2_multiobject_training_v1/MO1_mem2_task_dense8_5ep | 2×1 | 65.34 | 58.16 | 48.83 | 34.32 | 29.14/29.16 | 12515 | 0/0/0 |
| L010 | sam2_multiobject_training_v1/MO2_mem2_logits_dense8_5ep | 2×1 | 68.87 | 61.24 | 49.58 | 34.98 | 28.59/28.59 | 12482 | 0/0/0 |
| L011 | sam2_multiobject_training_v1/MO3_mem2_memlogits_dense8_5ep | 2×1 | 65.57* | 61.14 | 49.51 | 35.19 | 28.42/28.43 | 12515 | 1/0/0 |
| L012 | sam2_multiplex_overnight_v4/MX13_slot8_r2_mean_screen3ep | 1×1 | 60.27 | 49.58 | 53.46 | 47.04 | 21.26/21.26 | 11705 | 0/1/1 |
| L013 | sam2_multiplex_overnight_v4/MX14_slot8_r4_mean_screen3ep | 1×1 | 58.97 | 48.55 | 53.10 | 46.97 | 21.29/21.29 | 11706 | 0/1/1 |
| L014 | sam2_multiplex_overnight_v4/MX15_slot8_r8_mean_screen3ep | 1×1 | 55.59 | 48.13 | 51.03 | 46.76 | 21.39/21.39 | 11712 | 0/1/1 |
| L015 | sam2_multiplex_overnight_v4/MX16_slot8_r16_mean_screen3ep | 1×1 | 57.22 | 46.77 | 50.92 | 46.94 | 21.30/21.30 | 11701 | 0/1/1 |
| L016 | sam2_multiplex_overnight_v4/MX17_slot8_r8_mean_ptr4_screen3ep | 1×1 | 56.42 | 47.00 | 51.22 | 46.34 | 21.58/21.58 | 11712 | 0/1/1 |
| L017 | sam2_multiplex_overnight_v4/MX18_slot8_r8_mean_ptr8_screen3ep | 1×1 | 56.75 | 45.75 | 50.42 | 45.90 | 21.79/21.79 | 11717 | 0/1/1 |
| L018 | sam2_multiplex_overnight_v4/MX19_slot8_r8_latest_ptr8_screen3ep | 1×1 | 60.97 | 49.16 | 54.17 | 48.43 | 20.65/20.65 | 11670 | 0/1/1 |
| L019 | sam2_multiplex_overnight_v4/MX20_slot8_r8_recency050_ptr8_screen3ep | 1×1 | 59.76 | 48.40 | 52.13 | 46.94 | 21.30/21.30 | 11698 | 0/1/1 |
| L020 | sam2_multiplex_overnight_v4/MX21_slot8_r8_recency025_ptr8_screen3ep | 1×1 | 60.93 | 50.40 | 53.56 | 47.25 | 21.16/21.16 | 11712 | 0/1/1 |
| L021 | sam2_multiplex_overnight_v4/MX22_slot8_r8_recency075_ptr8_screen3ep | 1×1 | 59.02 | 49.20 | 50.56 | 46.76 | 21.39/21.39 | 11698 | 0/1/1 |
| L022 | sam2_multiplex_overnight_v4/MX23_slot4_r8_mean_ptr8_screen3ep | 1×1 | 58.81 | 47.63 | 53.69 | 32.35 | 30.91/30.91 | 11593 | 0/1/0 |
| L023 | sam2_multiplex_overnight_v4/MX24_slot6_r8_mean_ptr8_screen3ep | 1×1 | 59.96 | 50.67 | 54.50 | 30.51 | 32.78/32.78 | 12002 | 0/1/0 |
| L024 | sam2_multiplex_overnight_v4/MX25_slot8_r8_mean_ptr8_objkd025_screen3ep | 1×1 | 56.27 | 46.60 | 46.85 | 46.99 | 21.28/21.28 | 11718 | 0/1/1 |
| L025 | sam2_multiplex_overnight_v4/MX26_slot8_r8_mean_ptr8_objkd100_screen3ep | 1×1 | 59.82 | 50.35 | 53.21 | 47.40 | 21.10/21.10 | 11703 | 0/1/1 |
| L026 | sam2_multiplex_overnight_v4/MX27_slot8_min2_r8_mean_ptr8_screen3ep | 1×1 | 60.52 | 57.06 | 53.64 | 46.81 | 21.36/21.36 | 11712 | 1/1/1 |
| L027 | sam2_multiplex_overnight_v4/MX28_slot8_min3_r8_mean_ptr8_screen3ep | 1×1 | 59.51 | 48.81 | 50.37 | 47.41 | 21.09/21.09 | 11695 | 0/1/1 |
| L028 | sam2_object_slots_v1/MX1_slot4_decoder_kd_3ep | 3×1 | 59.68 | 49.66 | 37.59 | 21.68 | 46.13/46.15 | 11977 | 0/0/0 |
| L029 | sam2_object_slots_v1/MX2_slot8_decoder_kd_3ep | 3×1 | 61.65 | 49.85 | 37.77 | 24.46 | 40.88/40.89 | 12513 | 0/0/0 |
| L030 | sam2_object_slots_v1/MX3_slot4_sharedkv_kd_3ep | 3×1 | 60.66 | 49.70 | 55.40 | 34.52 | 28.97/28.99 | 11581 | 0/1/0 |
| L031 | sam2_object_slots_v1/MX4_slot8_sharedkv_kd_3ep | 3×1 | 59.87 | 49.54 | 55.47 | 49.03 | 20.39/20.46 | 11662 | 0/1/1 |
| L032 | sam2_object_slots_v2/MX5_slot8_decoder_t8_logits2_5ep | 3×1 | 60.11 | 49.68 | 37.61 | 24.40 | 40.99/41.00 | 12497 | 0/0/0 |
| L033 | sam2_object_slots_v2/MX6_slot8_sharedkv_t8_mem1_5ep | 3×1 | 60.28 | 49.99 | 55.52 | 49.06 | 20.38/20.40 | 11698 | 0/1/1 |
| L034 | sam2_object_slots_v2/MX7_slot8_sharedkv_t8_mem4_5ep | 3×1 | 60.88 | 49.43 | 55.50 | 48.72 | 20.52/20.53 | 11680 | 0/1/1 |
| L035 | sam2_object_slots_v2/MX8_slot8_sharedkv_t8_mem1_logits4_5ep | 3×1 | 59.26 | 49.93 | 54.89 | 48.79 | 20.50/20.55 | 11698 | 0/1/1 |

## Bucket and learned-slot decisions

### Object bucket implementations

| Implementation | Decision | Agreement | Min mask IoU | N1 FPS gain | N2 gain | N4 gain | N8 gain | N8 latency reduction | Peak-memory consequence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| bucket4 (non-persistent) | REJECT | 0.9999888 | - | -50.8% | -22.7% | +4.8% | +11.0% | +10.0% | +5,655 MB at N8 |
| bucket4 persistent m4 | REJECT by strict gate | 0.9999849 | 0.9690 | -1.5% | -1.1% | +42.0% | +47.5% | +32.2% | +584 MB at N8 |

The persistent version fixes the severe N=1/N=2 regression and materially accelerates N=4/N=8. Rejection is caused by non-bit-exact masks and the promotion-count gate, not by a failure to accelerate.

### Learned object-slot v1/v2

| Variant | Status | Val/Test J&F | Min quality retention | Learned mask IoU | N1 FPS/retention | N8 FPS/gain | Promote |
|---|---|---|---:|---:|---|---|---:|
| MX1_slot4_decoder_kd_3ep | complete | 72.3/74.6 | 0.999 | 0.000 | 59.68/0.836 | 21.68/-1.8% | 0 |
| MX2_slot8_decoder_kd_3ep | complete | 72.3/74.6 | 0.999 | 1.000 | 61.65/0.864 | 24.46/+10.8% | 0 |
| MX3_slot4_sharedkv_kd_3ep | complete | 63.8/60.9 | 0.815 | 0.413 | 60.66/0.850 | 34.52/+56.4% | 0 |
| MX4_slot8_sharedkv_kd_3ep | complete | 63.6/60.2 | 0.806 | 1.000 | 59.87/0.839 | 49.03/+122.1% | 0 |
| MX5_slot8_decoder_t8_logits2_5ep | pending in comparison snapshot | - | - | - | - | - | 0 |
| MX6_slot8_sharedkv_t8_mem1_5ep | complete | 63.6/60.3 | 0.807 | 1.000 | 60.28/0.845 | 49.06/+122.3% | 0 |
| MX7_slot8_sharedkv_t8_mem4_5ep | complete | 63.7/60.5 | 0.810 | 1.000 | 60.88/0.853 | 48.72/+120.7% | 0 |
| MX8_slot8_sharedkv_t8_mem1_logits4_5ep | complete | 63.6/60.3 | 0.807 | 1.000 | 59.26/0.830 | 48.79/+121.0% | 0 |

### MX13–MX28 screen

Quality uses a fixed 32-video SA-V val cohort against MX5. Pending means the comparison artifact lacked a completed quality screen even when a latency aggregate existed.

| Variant | Screen | Mini J&F | Retention | Mask IoU | N1 FPS | N4 FPS | N8 FPS/gain | Promote |
|---|---|---:|---:|---:|---:|---:|---|---:|
| MX13_slot8_r2_mean_screen3ep | pending | - | - | - | - | - | - | 0 |
| MX14_slot8_r4_mean_screen3ep | pending | - | - | - | - | - | - | 0 |
| MX15_slot8_r8_mean_screen3ep | fail | 53.60 | 0.757 | 1.000 | 55.59 | 51.03 | 46.76/+111.8% | 0 |
| MX16_slot8_r16_mean_screen3ep | pending | - | - | - | - | - | - | 0 |
| MX17_slot8_r8_mean_ptr4_screen3ep | fail | 53.70 | 0.758 | 1.000 | 56.42 | 51.22 | 46.34/+110.0% | 0 |
| MX18_slot8_r8_mean_ptr8_screen3ep | pending | - | - | - | - | - | - | 0 |
| MX19_slot8_r8_latest_ptr8_screen3ep | fail | 53.00 | 0.749 | 1.000 | 60.97 | 54.17 | 48.43/+119.4% | 0 |
| MX20_slot8_r8_recency050_ptr8_screen3ep | pending | - | - | - | - | - | - | 0 |
| MX21_slot8_r8_recency025_ptr8_screen3ep | fail | 52.90 | 0.747 | 1.000 | 60.93 | 53.56 | 47.25/+114.1% | 0 |
| MX22_slot8_r8_recency075_ptr8_screen3ep | pending | - | - | - | - | - | - | 0 |
| MX23_slot4_r8_mean_ptr8_screen3ep | fail | 53.70 | 0.758 | 0.000 | 58.81 | 53.69 | 32.35/+46.6% | 0 |
| MX24_slot6_r8_mean_ptr8_screen3ep | pending | - | - | - | - | - | - | 0 |
| MX25_slot8_r8_mean_ptr8_objkd025_screen3ep | pending | - | - | - | - | - | - | 0 |
| MX26_slot8_r8_mean_ptr8_objkd100_screen3ep | pending | - | - | - | - | - | - | 0 |
| MX27_slot8_min2_r8_mean_ptr8_screen3ep | fail | 29.50 | 0.417 | 1.000 | 60.52 | 53.64 | 46.81/+112.1% | 0 |
| MX28_slot8_min3_r8_mean_ptr8_screen3ep | pending | - | - | - | - | - | - | 0 |

## Interpretation guardrails

- Image mIoU/AP staying near 0.84/0.72 does not imply temporal quality is preserved; the shared-K/V runs demonstrate this directly.
- VOS seconds per video were collected in multi-worker evaluation and are not interchangeable with isolated propagation FPS.
- `complete` describes the pipeline artifact state in the universal report. It does not mean the model passed a research or deployment gate.
- `final_checkpoint_incomplete` frequently means training reached its target checkpoint but final validation/export was absent.
- Comparisons use different cohorts (full SA-V val/test versus a fixed 32-video screen); compare values only within the stated protocol.
