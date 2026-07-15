# RepViT to SAM2.1-L Stage 1 Distillation

This experiment asks whether a mobile CNN can preserve SAM2.1-L promptable
segmentation and memory-tracking quality more efficiently than the existing
TinyViT students.

| student | ImageNet initialization | parameters | role |
| --- | --- | ---: | --- |
| RepViT-M0.9 | `repvit_m0_9.dist_450e_in1k` | 5.5M | small-model challenge |
| RepViT-M2.3 | `repvit_m2_3.dist_450e_in1k` | 23.7M | mobile-architecture control |

M2.3 is not smaller than TinyViT-21M by parameter count. Its value is testing
the RepViT architecture and deployment profile at a similar scale.

Prepare the pinned Hugging Face checkpoints on the company cluster:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only origin edgetam-tinyvit-pipeline

HF_XET_HIGH_PERFORMANCE=1 \
scripts/company/37_download_repvit_pretrained.sh
```

The command writes checkpoints, Hugging Face configs, SHA256 files, and model
contract summaries under:

```text
/group-volume/danny-dataset/sam2_distill/checkpoints/repvit
```

The planned training contract matches the TinyViT Stage 1 comparison: frozen
SAM2.1-L teacher, projection-only RepViT student interface, MSE + cosine + L1,
five epochs on the mounted SA-V manifest, full feature-loss validation for
`best.pt`, then full `sav_val` and `sav_test` box-prompt image and memory-VOS
evaluation. Image metrics are mIoU, AP50:95, and latency; VOS metrics are J&F
and latency. RepViT runs use a separate W&B project.
