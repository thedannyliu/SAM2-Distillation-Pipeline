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
| image encoder | 25.941 | 50.57% | yes |
| prompt encoder | 0.105 | 0.20% | yes |
| memory attention | 20.487 | 39.94% | yes |
| mask decoder | 1.867 | 3.64% | yes |
| memory encoder | 0.772 | 1.50% | yes |
| object-pointer projections | 0.030 | 0.06% | yes |
| framework/uninstrumented residual | 2.091 | 4.08% | yes |

The official SAM2.1-L measurement completed on one H100 using video
`sav_000262`, one object, 16 warm-up frames, and 384 measured frames. Mean
tracking latency is 51.293 ms, median is 51.223 ms, and P90 is 51.793 ms.

The image encoder and memory attention together account for 90.51% of the
single-object model tracking step. Within the image encoder, the trunk is
25.523 ms (98.39% of encoder time) and the neck is 0.385 ms. Within the mask
decoder, the transformer is 1.535 ms (82.22% of decoder time). Within the
memory encoder, the mask downsampler and fuser are 0.326 and 0.378 ms.

This establishes two primary bottlenecks for event-triggered work: expensive
current-frame observation and temporal retrieval. Optimizing only the mask
decoder or memory write targets 5.14% of latency, whereas an action that skips
both the image encoder and memory attention targets 90.51%. These percentages
are for one object and model-only propagation; multi-object scaling and Thor
end-to-end latency require separate measurements.

Machine-readable output:

`/group-volume/danny-dataset/sam2_distill/runs/sam2_tracking_component_profile_v2/sam21l/summary.json`

Terminal-friendly additive table:

`/group-volume/danny-dataset/sam2_distill/runs/sam2_tracking_component_profile_v2/sam21l/component_summary.csv`
