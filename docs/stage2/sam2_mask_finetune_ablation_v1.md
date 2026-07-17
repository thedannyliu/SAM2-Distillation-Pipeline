# SAM2 Mask Fine-Tuning Ablation V1

## Question

Which optimization choice controls mask-decoder adaptation and the observed
step-loss variance after Stage-1 TinyViT task tuning?

All runs start from the same completed V2 Stage-1 task checkpoint, use one
training epoch and two frames per clip, then run full `sav_val` and `sav_test`
image/VOS evaluation. Use validation metrics for decisions; test metrics are
recorded only because the production pipeline requires complete evaluation.

## Matrix

| Variant | Trainable modules | Encoder LR | Decoder LR | BN | Correction points | Hypothesis |
|---|---|---:|---:|---|---:|---|
| `decoder_lr2e7` | decoder | frozen | 2e-7 to 5e-8 | frozen | 1 | Conservative decoder adaptation |
| `decoder_lr5e7` | decoder | frozen | 5e-7 to 1e-7 | frozen | 1 | V2 decoder-only baseline |
| `decoder_lr2e6` | decoder | frozen | 2e-6 to 2e-7 | frozen | 1 | Test decoder LR sensitivity |
| `encdec_low_frozenbn` | encoder + decoder | 1e-7 to 2e-8 | 5e-7 to 1e-7 | frozen | 1 | Test low-LR co-adaptation |
| `encdec_low_trainbn` | encoder + decoder | 1e-7 to 2e-8 | 5e-7 to 1e-7 | train | 1 | Isolate small-batch BN instability |
| `decoder_lr5e7_boxonly` | decoder | frozen | 5e-7 to 1e-7 | frozen | 0 | Isolate iterative correction noise |

The prior V1 encoder+decoder run at higher LR and two epochs is the external
control; it is not repeated. Compare raw loss and `*_ema` in W&B, then rank
runs by val mIoU/AP without materially reducing val J&F.

## Results

| Variant | Status | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F |
|---|---|---:|---:|---:|---:|---:|---:|
| `decoder_lr2e7` | pending | | | | | | |
| `decoder_lr5e7` | pending | | | | | | |
| `decoder_lr2e6` | pending | | | | | | |
| `encdec_low_frozenbn` | pending | | | | | | |
| `encdec_low_trainbn` | pending | | | | | | |
| `decoder_lr5e7_boxonly` | pending | | | | | | |
