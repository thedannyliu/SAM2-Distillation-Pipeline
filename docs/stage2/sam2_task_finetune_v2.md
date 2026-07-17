# SAM2 Task Fine-Tuning V2

## Question

Can decoder-only adaptation preserve the Stage 1 representation and avoid the
VOS regression observed when the encoder and mask decoder were jointly tuned,
while retaining the gain from the final memory-unfreeze stage?

## Prior Signal

| Stage | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F |
|---|---:|---:|---:|---:|---:|---:|
| Encoder task tuning | 0.83763 | 0.71286 | 70.3 | 0.83558 | 0.71602 | 71.8 |
| Encoder + decoder | 0.83788 | 0.71293 | 69.9 | 0.83599 | 0.71669 | 71.3 |
| Encoder + decoder + memory | 0.83806 | 0.71334 | 71.5 | 0.83642 | 0.71714 | 74.3 |

The decoder co-adaptation stage produced negligible image gains and reduced
J&F. The memory stage produced the only material task gain. These are reused
V1 artifacts, so V2 writes to a new run root and W&B project.

## V2 Design

| Stage | Trainable modules | Frames | Epochs | Encoder LR | Other LR |
|---|---|---:|---:|---:|---:|
| 1 | TinyViT + projection | 2 | 2 | 1e-6 to 1e-7 | n/a |
| 2 | Mask decoder only | 2 | 1 | frozen | 5e-7 to 1e-7 |
| 3 | Encoder + decoder + memory/tracking | 4 | 1 | 3e-7 to 3e-8 | 1e-6 to 1e-7 |

The prompt encoder and all BatchNorm modules remain frozen. Every stage runs
training followed by full `sav_val` and `sav_test` image/VOS evaluation. W&B
also records an EMA companion for every instantaneous task-loss metric and
writes the completed split metrics back to that stage's W&B summary.

## Results

| Stage | Train status | val mIoU | val AP | val J&F | test mIoU | test AP | test J&F |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | pending | | | | | | |
| 2 | pending | | | | | | |
| 3 | pending | | | | | | |
