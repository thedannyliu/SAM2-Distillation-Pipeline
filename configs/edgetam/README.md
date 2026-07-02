# EdgeTAM Configs

This directory stores small, repo-owned config templates. The full SAM2 and
EdgeTAM model configs are generated or loaded from pinned upstream checkouts.

Generate the TinyViT student model YAML:

```bash
python tools/edgetam/write_tinyvit_edgetam_config.py \
  --out runs/edgetam_smoke/configs/edgetam_tinyvit21m.yaml
```

The generator probes `timm.feature_info` to avoid hardcoded TinyViT channels.

`tinyvit_video_distill_smoke.yaml` is the repo-owned full-trainer smoke
template. It uses the EdgeTAM TinyViT model, official SAM2 VOS task loss, and
EdgeTAM image/memory distillation wrapper. Validate it without launching a
trainer with:

```bash
scripts/pace/06_run_edgetam_tinyvit_smoke.sh validate-train-config
```
