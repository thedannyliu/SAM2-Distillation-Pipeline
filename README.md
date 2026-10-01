# SAM2 distillation and task fine-tuning

Distill SAM2.1 image features into TinyViT encoders, then fine-tune the assembled
model for prompted segmentation and video object tracking. This repository
owns training and evaluation; the companion
[SAM2 TensorRT runtime](https://github.com/thedannyliu/Efficient-SAM2-TensorRT)
owns export and on-device execution.

## Pipeline

1. Prepare image/video manifests and audit the dataset.
2. Distill the image encoder against a frozen SAM2.1 teacher.
3. Fine-tune the student with the SAM2 decoder and temporal memory.
4. Select on SA-V validation, evaluate the chosen checkpoint, and export a bundle.

The student returns SAM2-compatible image embeddings and high-resolution
features. Encoder size alone does not determine tracking quality: memory and
checkpoint reconstruction must match the trained model.

## Recorded results

The supplied project handoff reports the following SA-V results. These are
historical training results, not measurements reproduced by the CPU tests.

| Student | Validation J&F | Test J&F | Test image mIoU |
| --- | ---: | ---: | ---: |
| TinyViT-21M | 72.4 | 75.5 | 0.837 |
| TinyViT-11M | 68.6 | 71.5 | 0.816 |
| TinyViT-5M | 66.0 | 69.1 | 0.803 |

See [results and limitations](docs/results.md) for provenance and scope.

## Start here

Use the [training guide](docs/training.md) for prerequisites, data gates,
commands and checkpoint reconstruction. The company container uses Python 3.10
and PyTorch 2.4; setup must preserve its preinstalled torch and run the SAM2
compatibility smoke. Model weights and datasets are external.

```bash
# From the repository root, in the intended training container:
SAM2_UPSTREAM=/path/to/sam2 bash scripts/core/setup_env.sh
# Inspect the prepared data before scheduling training:
SAV_ROOT=/path/to/SA-V SAM2D_ROOT=/path/to/sam2_distill \
  bash scripts/core/data_prepare_sav.sh audit
```

## Code map

| Path | Responsibility |
| --- | --- |
| [`sam2_distill/models/`](sam2_distill/models) | Student adapters, model assembly and checkpoint contracts |
| [`sam2_distill/training/`](sam2_distill/training) | Distillation objectives |
| [`tools/`](tools) | Reusable data, training, evaluation and benchmark CLIs |
| [`scripts/core/`](scripts/core) | Main data and training workflows |
| [`scripts/lib/`](scripts/lib) | Shared training/evaluation drivers |
| [`scripts/edgetam/`](scripts/edgetam) | Experimental compressed-memory workflow |
| [`scripts/deploy/`](scripts/deploy) | Tracking profiles and multi-object benchmarks |
| [`scripts/experiments/`](scripts/experiments) | Research recipes and historical ablations |
| [`tests/`](tests) | Dataset, loss, checkpoint and multi-object regression checks |

[Documentation index](docs/README.md) · [script migration](scripts/README.md)

Upstream SAM2, EdgeTAM, timm and pretrained weights retain their own terms.
