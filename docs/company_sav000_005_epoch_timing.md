# Company SA-V sav_000-sav_005 Epoch Timing

Use this run to measure one full training epoch on SA-V shards `sav_000`
through `sav_005` with aggressive H100 settings.

Default training setting:

```text
trainable: full image_encoder only
frozen: prompt encoder, mask decoder, memory encoder, memory attention, other SAM2 components
per-GPU batch size: 4
global batch size: 4 on 1xH100, 16 on 4xH100
frames per sample: 8
resolution: 1024
image_encoder_forward_batch_size: 16
image_encoder_activation_checkpoint: disabled
epochs: 1
```

Prepare the combined symlink layout:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline

scripts/company/09_run_sav000_005_epoch_timing.sh prepare
```

If a shard has mp4 files but no extracted JPEG frames, prepare will fail. Extract
frames once with:

```bash
EXTRACT_MISSING_FRAMES=1 scripts/company/09_run_sav000_005_epoch_timing.sh prepare
```

Run one epoch on one H100:

```bash
scripts/company/09_run_sav000_005_epoch_timing.sh 1gpu
```

Run one epoch on four H100s:

```bash
scripts/company/09_run_sav000_005_epoch_timing.sh 4gpu
```

If the aggressive setting OOMs, first retry with:

```bash
BATCH_SIZE=2 IMAGE_ENCODER_BATCH=8 IMAGE_ENCODER_CKPT=0 \
scripts/company/09_run_sav000_005_epoch_timing.sh 1gpu
```

and for 4 GPUs:

```bash
BATCH_SIZE=2 IMAGE_ENCODER_BATCH=8 IMAGE_ENCODER_CKPT=0 \
scripts/company/09_run_sav000_005_epoch_timing.sh 4gpu
```

Outputs are under:

```text
/group-volume/danny-dataset/sam2_distill/runs/sav000_005_epoch_timing/
```

Each run writes:

```text
preflight.json              estimated steps/global batch before training
train.log                   full trainer log
gpu_usage.csv               nvidia-smi samples
runtime_epoch.json          wall-clock runtime
summary.json                trainer summary from rank 0
epoch_timing_summary.json   final throughput and GPU utilization summary
```

Read the final timing summary:

```bash
cat /group-volume/danny-dataset/sam2_distill/runs/sav000_005_epoch_timing/1gpu_b4_ieb16_ckpt0/epoch_timing_summary.json
cat /group-volume/danny-dataset/sam2_distill/runs/sav000_005_epoch_timing/4gpu_b4_ieb16_ckpt0/epoch_timing_summary.json
```
