# SAM2.1-L tracking component profile v1

## Question

During a single SAM2.1 Hiera-Large tracking stream, what fraction of the
end-to-end model inference step is spent in the image encoder?

This experiment intentionally loads only the official SAM2.1-L checkpoint. It
does not load TinyViT, EdgeTAM, or a task-finetuned student.

## Measurement contract

- Dataset: one sufficiently long `sav_val` video, selected deterministically
  from `sav_val.txt`.
- Prompt: every object with a frame-0 ground-truth mask, in one shared
  multi-object session.
- Warm-up: 16 non-prompt tracking frames.
- Measurement: 128 frames repeated three times on one isolated GPU.
- Full latency: synchronized wall time for one `propagate_in_video` step.
- Image encoder: CUDA-event time around `image_encoder.forward` in the same
  tracking step.
- Image-feature path: CUDA-event time around `_get_image_feature`, reported as
  a secondary number.

Model loading, video initialization/preloading, prompt insertion, JPEG output,
and PNG serialization are excluded. The result is therefore end-to-end
**model tracking-step latency**, not camera/decode/render application latency.

## Company command

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only
mkdir -p /user-volume/log/sam21l_tracking_component_profile_v1

GPU=0 scripts/company/74_profile_sam2_tracking_components.sh 2>&1 | \
  tee /user-volume/log/sam21l_tracking_component_profile_v1/run.log
echo "SAM2.1-L profile status: ${PIPESTATUS[0]}"
```

## Results

| GPU | Objects | Tracking ms | Encoder ms | Encoder share | Image-feature-path share |
|---|---:|---:|---:|---:|---:|
| pending | pending | pending | pending | pending | pending |

Machine-readable output:

`/group-volume/danny-dataset/sam2_distill/runs/sam2_tracking_component_profile_v1/sam21l/summary.json`
