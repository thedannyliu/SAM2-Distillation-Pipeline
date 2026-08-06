# SAM2.1-L tracking component profile v2

## Question

During a single SAM2.1 Hiera-Large tracking stream, what fraction of the
end-to-end model inference step is spent in each top-level semantic component?

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
- Additive top-level components: image encoder, prompt encoder, memory
  attention, mask decoder, memory encoder, object-pointer projection, and
  object-pointer temporal projection.
- Residual: synchronized full step minus the additive top-level CUDA-event
  times. It includes Python/framework overhead, postprocessing, uninstrumented
  operators, launch/synchronization overhead, and CPU time.
- Nested drill-down: image trunk/neck, mask-decoder transformer, and memory
  encoder mask downsampler/pixel projection/fuser/output projection. These
  overlap their parent and are not added into the top-level 100%.
- Image-feature path: CUDA-event time around `_get_image_feature`, reported as
  a secondary non-additive number.

Model loading, video initialization/preloading, prompt insertion, JPEG output,
and PNG serialization are excluded. The result is therefore end-to-end
**model tracking-step latency**, not camera/decode/render application latency.

## Company command

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only
mkdir -p /user-volume/log/sam21l_tracking_component_profile_v2

GPU=0 scripts/company/74_profile_sam2_tracking_components.sh 2>&1 | \
  tee /user-volume/log/sam21l_tracking_component_profile_v2/run.log
echo "SAM2.1-L profile status: ${PIPESTATUS[0]}"
```

## Results

| Component | Mean ms | Tracking share | Additive? |
|---|---:|---:|---|
| image encoder | pending | pending | yes |
| prompt encoder | pending | pending | yes |
| memory attention | pending | pending | yes |
| mask decoder | pending | pending | yes |
| memory encoder | pending | pending | yes |
| object-pointer projections | pending | pending | yes |
| framework/uninstrumented residual | pending | pending | yes |

Machine-readable output:

`/group-volume/danny-dataset/sam2_distill/runs/sam2_tracking_component_profile_v2/sam21l/summary.json`

Terminal-friendly additive table:

`/group-volume/danny-dataset/sam2_distill/runs/sam2_tracking_component_profile_v2/sam21l/component_summary.csv`
