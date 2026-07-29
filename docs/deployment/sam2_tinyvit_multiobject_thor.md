# TinyViT SAM2 models and multi-object buckets for Thor

This is the deployment handoff for the selected TinyViT SAM2 models and the
current runtime-only multi-object acceleration. The bucket path is an opt-in
deployment candidate: it has a useful measured latency gain, but it did not
pass the repository's original strict equivalence and latency promotion gates.

## 1. Selected model artifacts

Selection uses full SA-V validation J&F across completed formal experiments.
Test metrics are reported for context and were not used to select a model.

| Model | Selected run | val J&F | test J&F | Company task checkpoint |
| --- | --- | ---: | ---: | --- |
| TinyViT-5M | `weekend_72h_v1/tv5_W1_decmem_t4_3ep` | **66.0** | 67.6 | `/group-volume/danny-dataset/sam2_distill/runs/weekend_72h_v1/tinyvit/tv5_W1_decmem_t4_3ep/checkpoints/best.pt` |
| TinyViT-11M | `weekend_72h_v1/tv11_W1_decmem_t4_3ep` | **68.6** | 70.5 | `/group-volume/danny-dataset/sam2_distill/runs/weekend_72h_v1/tinyvit/tv11_W1_decmem_t4_3ep/checkpoints/best.pt` |
| TinyViT-21M | `tinyvit_max_jf_v1/tv21` | **72.4** | 74.7 | `/group-volume/danny-dataset/sam2_distill/runs/tinyvit_max_jf_v1/tv21/main/checkpoints/best.pt` |

The selection record and the complete metrics are in
[`backbone_task_expansion_v2.md`](../experiments/backbone_task_expansion_v2.md).

### Required runtime bundle

The current Python loader reconstructs a SAM2.1-L predictor, replaces its
image path with the selected TinyViT student, and loads the task-tuned
decoder/memory tensors. Therefore `best.pt` is not a standalone TensorRT
engine or a standalone SAM2 checkpoint.

Copy these shared files once:

```text
/group-volume/danny-dataset/sam2_distill/checkpoints/sam2.1/sam2.1_hiera_large.pt
/user-volume/repo/facebookresearch-sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
```

Copy the matching TinyViT initializer with each task checkpoint:

```text
TV5M:
  /group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors
  /group-volume/danny-dataset/sam2_distill/runs/weekend_72h_v1/tinyvit/tv5_W1_decmem_t4_3ep/resolved_config.yaml

TV11M:
  /group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors
  /group-volume/danny-dataset/sam2_distill/runs/weekend_72h_v1/tinyvit/tv11_W1_decmem_t4_3ep/resolved_config.yaml

TV21M:
  /group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors
  /group-volume/danny-dataset/sam2_distill/runs/tinyvit_max_jf_v1/tv21/main/resolved_config.yaml
```

The resolved training configs are provenance artifacts. The current inference
loader uses the official SAM2.1-L config plus metadata inferred from the
checkpoint, while the resolved config remains useful for reproducing the
selected task model and auditing its training settings.

Before transfer, produce a manifest on the company node:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
mkdir -p /user-volume/thor_sam2_export
sha256sum \
  /group-volume/danny-dataset/sam2_distill/runs/weekend_72h_v1/tinyvit/tv5_W1_decmem_t4_3ep/checkpoints/best.pt \
  /group-volume/danny-dataset/sam2_distill/runs/weekend_72h_v1/tinyvit/tv11_W1_decmem_t4_3ep/checkpoints/best.pt \
  /group-volume/danny-dataset/sam2_distill/runs/tinyvit_max_jf_v1/tv21/main/checkpoints/best.pt \
  /group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors \
  /group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_11m_224.dist_in22k_ft_in1k.safetensors \
  /group-volume/danny-dataset/sam2_distill/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors \
  /group-volume/danny-dataset/sam2_distill/checkpoints/sam2.1/sam2.1_hiera_large.pt | \
tee /user-volume/thor_sam2_export/SHA256SUMS
echo "Manifest status: ${PIPESTATUS[0]}"
```

Use `rsync -avP` or the company's artifact transfer mechanism to copy the
files, the corresponding config, `SHA256SUMS`, and this repository commit to
Thor. Verify on Thor with `sha256sum -c SHA256SUMS` before engine export.

## 2. What the bucket acceleration changes

The implementation is
[`sam2_object_buckets.py`](../../sam2_distill/models/sam2_object_buckets.py).
It wraps an existing `SAM2VideoPredictor`; model weights and the public prompt
and propagation API do not change.

Standard SAM2 keeps independent object histories and propagates a dynamic
object batch whose batch size is the current object count. The adapter:

1. preserves the stable object-ID order already stored in the SAM2 inference
   state;
2. partitions the objects once into capacity-four buckets;
3. concatenates synchronized per-object conditional and non-conditional
   histories along the existing tensor batch dimension;
4. calls SAM2's `_run_single_frame_inference` once per capacity-bounded bucket
   per frame instead of one dynamically growing object batch;
5. appends the result to a persistent bucket history instead of repacking the
   full history every frame;
6. slices the bucket output back into per-object histories and emits masks in
   the original object-ID order.

For `N` objects and bucket capacity `B`, bucket mode executes
`ceil(N / B)` bounded calls instead of one call with dynamic batch size `N`.
The total object batch across those calls is still `N`, so object-dependent
GPU work and mask output do not become constant. Its opportunity is stable
bounded shapes, persistent bucket histories, and better kernel/memory
behavior at larger object counts. It may add launch overhead, which is why
small sessions stay on the legacy path. This is not a learned multiplex
decoder and it does not yet share memory-attention K/V between objects.

Small sessions use the original predictor exactly. Sessions with
unsynchronized per-object frame histories also fall back to the original
predictor rather than changing tracker semantics.

## 3. Reference integration

The wrapper delegates all methods other than propagation to the original
predictor, so the application can keep its existing SAM2 session code:

```python
from sam2_distill.models.sam2_object_buckets import SAM2ObjectBucketAdapter

base_predictor = build_existing_sam2_predictor(...)
predictor = SAM2ObjectBucketAdapter(
    base_predictor,
    bucket_size=4,
    min_bucket_objects=4,
)

state = predictor.init_state(video_path=video_path)
for object_id, prompt in prompts:
    predictor.add_new_points_or_box(
        state,
        frame_idx=prompt.frame_idx,
        obj_id=object_id,
        points=prompt.points,
        labels=prompt.labels,
    )

for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
    consume(frame_idx, object_ids, masks)
```

Required predictor settings:

```text
non_overlap_masks_for_mem_enc = false
clear_non_cond_mem_around_input = false
bucket_size = 4
min_bucket_objects = 4
```

Operational requirements:

- register the initial prompts before propagation;
- keep object registration order stable for the life of the session;
- use synchronized prompt/history frames when bucket execution is desired;
- keep each mutable `inference_state` owned by one session;
- do not concurrently mutate one predictor or inference state from multiple
  application threads;
- inspect `predictor.execution_stats` to distinguish bucket execution from
  small-session and unsynchronized-history fallback.

The adapter raises at construction if either unsupported predictor setting is
enabled. It raises for empty sessions or malformed/missing outputs instead of
silently returning reordered masks.

## 4. Thor engine boundary

The Python adapter is the executable reference, not the final TensorRT
orchestrator. For a native Thor engine:

- export the selected TinyViT + task-tuned SAM2 per-frame graph after loading
  the complete runtime bundle;
- provide fixed or optimized batch profiles for object batch sizes 1–4;
- keep stable object-to-bucket mapping and persistent history in the engine
  session controller;
- run one batch-four profile for four objects and two batch-four profiles for
  eight objects; use the exact low-object path for one to three objects;
- implement packing and demultiplexing with device-resident views or copies
  where possible;
- preserve per-object memory tensors. Do not average or share K/V in this
  runtime path;
- return object IDs in the original registration order.

The current reference boundary is the call to
`_run_single_frame_inference` in
[`SAM2ObjectBucketAdapter.propagate_in_video`](../../sam2_distill/models/sam2_object_buckets.py).
Use that call's actual tensor contract during export rather than inventing a
new checkpoint schema.

Checkpoint reconstruction is implemented in
[`run_sam2_vos_prompt_dataset.py`](../../tools/eval/run_sam2_vos_prompt_dataset.py):

- `build_predictor` creates the SAM2.1-L video predictor;
- `patch_stage1_forward_image` builds and loads the selected TinyViT;
- `load_task_non_image_state` in
  [`stage1_checkpoint.py`](../../sam2_distill/models/stage1_checkpoint.py)
  strictly loads the task-tuned decoder and memory tensors.

Export and numerical validation should happen only after this PyTorch loader
reports no missing or unexpected student keys and a successful strict
non-image task load.

## 5. Measured result and current acceptance status

The capacity-four persistent implementation was measured on one H100 using
the selected TV21M checkpoint, one 580-frame SA-V video, two repetitions, and
one to eight point-prompted objects.

| Objects | Legacy FPS | Bucket FPS | FPS change | Legacy ms/frame | Bucket ms/frame | Peak-memory change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 72.47 | 71.36 | -1.5% | 13.80 | 14.01 | -1 MB |
| 2 | 47.68 | 47.15 | -1.1% | 20.97 | 21.21 | +1 MB |
| 4 | 26.83 | 38.09 | **+42.0%** | 37.29 | 26.26 | +501 MB |
| 8 | 14.96 | 22.07 | **+47.5%** | 66.83 | 45.30 | +584 MB |

Numerical comparison:

```text
global binary-mask agreement = 0.9999849392
minimum per-mask IoU          = 0.9690141082
bit-exact match               = false
```

The repository comparison returned `REJECT` because the original gate
required stricter mask equivalence and count-relative latency targets. The
minimum per-mask IoU is above the later 95% quality tolerance, but this small
single-video equivalence run is not a substitute for a full deployment-set
quality evaluation. Treat bucket mode as opt-in until the Thor engine passes
the gates below.

Reference artifacts:

```text
/group-volume/danny-dataset/sam2_distill/runs/sam2_multiobject_bucket_mx1p_v1/tv21_best/point_n1-2-4-8
/group-volume/danny-dataset/sam2_distill/runs/sam2_multiobject_bucket_mx1p_v1/tv21_best/point_n1-2-4-8_bucket4_persistent_m4
/group-volume/danny-dataset/sam2_distill/runs/sam2_multiobject_bucket_mx1p_v1/comparisons/tv21_bucket4_persistent_vs_legacy
```

## 6. Validation and rollout gates

The benchmark entry point is
[`benchmark_sam2_multiobject_scaling.py`](../../tools/benchmark/benchmark_sam2_multiobject_scaling.py).
It can run legacy or bucket execution and includes a frame-level equivalence
check. The company wrapper is
[`59_run_sam2_multiobject_scaling.sh`](../../scripts/company/59_run_sam2_multiobject_scaling.sh).
Matched-run comparison is implemented in
[`summarize_sam2_object_buckets.py`](../../tools/benchmark/summarize_sam2_object_buckets.py).
Unit coverage for packing, ordering, persistent history, the low-object fast
path, and safe fallback is in
[`test_multiobject_scaling.py`](../../tests/test_multiobject_scaling.py).

Before enabling bucket mode by default on Thor, require:

1. identical model, video, object IDs, prompts, frame range, precision, and
   warmup for legacy and bucket runs;
2. at least 95% per-mask IoU for every checked object/frame, plus full
   application-quality metrics on the deployment validation set;
3. no object-ID swaps, missing masks, or shape/order changes;
4. no regression greater than 3% for one- and two-object P50/P90 latency;
5. positive P50 and P90 latency gain at four and eight objects;
6. peak memory within the Thor service budget;
7. legacy fallback retained behind a runtime flag and exercised in staging.

Start with `bucket_size=4` and `min_bucket_objects=4`. Do not extrapolate the
TV21M H100 numbers to TV5M, TV11M, or Thor; benchmark each exported engine
independently.

## 7. What is not in this deployment

The learned fixed-slot decoder and shared-memory K/V experiments are separate
research code:

- [`sam2_object_slots.py`](../../sam2_distill/models/sam2_object_slots.py)
- [`sam2_object_slot_predictor.py`](../../sam2_distill/models/sam2_object_slot_predictor.py)
- [`sam2_learned_object_slots_v1.md`](../experiments/sam2_learned_object_slots_v1.md)

Those paths change model behavior and require trained slot/shared-KV weights.
They are not part of the three selected checkpoints above and should not be
enabled in the Thor deployment until their train/val/test and latency suites
complete successfully.
