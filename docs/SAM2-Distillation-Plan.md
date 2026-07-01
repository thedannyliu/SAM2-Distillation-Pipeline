# SAM2 Distillation Plan for Robotics: Efficient Box-Prompt Segmentation and Tracking

## 1. Goal

Build an efficient SAM2-like model for a robotics agentic system where the upstream VLM or open-vocabulary detector provides a bounding box, and our module performs:

```text
VLM / open-vocabulary detector
        -> bbox + text label + frame id
Efficient SAM2-like segmentation
        -> mask + object id + confidence
SAM2-like memory tracker
        -> mask propagation over time
Robot policy / manipulation / navigation
```

The target model should preserve the useful SAM2 behavior:

- box-prompt segmentation;
- point-prompt and mask-prompt compatibility if needed later;
- video tracking from one or more prompted frames;
- memory-based mask propagation;
- occlusion and reappearance handling;
- multi-object tracking with bounded memory.

The company-specific contribution should be the training/distillation pipeline, robotics data, box-noise robustness, and efficient memory-bank design.

---

## 2. Starting Point: Repositories and What to Reuse

### Primary training scaffold: official SAM2

Use the official `facebookresearch/sam2` repository as the base training code. Its training folder contains dataset loaders, the `SAM2Train` training model, loss functions, optimizer utilities, the trainer loop, and `training/train.py`. It supports image training, video training, and mixed image/video training.

Use this as the actual engineering base:

```text
fork facebookresearch/sam2
branch: sam2-distillation-robotics
```

### Architecture references: EdgeTAM and EfficientTAM

Use EdgeTAM as the main architecture reference because it is explicitly designed as an efficient SAM2-style promptable video segmentation and tracking model. EdgeTAM introduces a lightweight image encoder, fewer memory-attention blocks, and a 2D Spatial Perceiver for memory compression. It reports 22x faster speed than SAM2 and 16 FPS on iPhone 15 Pro Max without quantization.

However, EdgeTAM should not be used as the training-code base because the official repository does not appear to release the full training/distillation pipeline. There is an open GitHub issue requesting training-code release.

Use EfficientTAM as a secondary architecture reference for lightweight image encoding and efficient memory cross-attention. Treat it as a reference implementation, not as the main training scaffold, because the public repository also does not appear to provide a full training pipeline.

### Image-distillation references: MobileSAM and EdgeSAM

Use MobileSAM as a conceptual reference for encoder-only feature distillation. Its idea is useful for Stage 1: train a lightweight encoder to mimic the teacher image encoder output.

Use EdgeSAM as a stronger image-stage reference because it shows that task-agnostic encoder-only distillation is not enough for promptable segmentation. EdgeSAM adds prompt-in-the-loop distillation using box and point prompts, which is directly relevant to our Stage 2. Note that EdgeSAM is SAM1/image-only and has a different license, so use it as a method reference unless legal review allows code reuse.

Recommended dependency policy:

```text
Actual codebase:     official SAM2 training code
Architecture ideas:  EdgeTAM + EfficientTAM
Image KD ideas:      MobileSAM + EdgeSAM
Company additions:   robotics data, bbox-noise training, video memory KD, efficient memory bank
```

---

## 3. High-Level System Design

### Teacher

Use a strong SAM2.1 teacher:

```text
Primary teacher: SAM2.1-Large or SAM2.1-B+
Alternative:     SAM2-HieraB+ if matching EdgeTAM more closely
```

The teacher is used to generate:

- image embeddings;
- high-resolution feature maps;
- box-prompt mask logits;
- predicted IoU / stability scores;
- video masklets;
- memory-attention outputs;
- object pointer features;
- occlusion / visibility signals if available.

### Student

Build a SAM2-compatible efficient student:

```text
Student = lightweight image encoder
        + SAM2-compatible feature neck / projection adapters
        + SAM2 prompt encoder
        + SAM2 mask decoder
        + memory encoder
        + reduced memory attention
        + efficient memory bank
```

Initial student choices:

```text
Image encoder: RepViT-M1, TinyViT, EfficientViT, or lightweight Hiera variant
Memory attention blocks: start with 2 blocks
Memory compression: 2D Spatial Perceiver or simpler token-pruning baseline
Memory cap: fixed max memory frames/tokens per object
```

The student must expose the same inference contract as SAM2:

```python
set_image(image)
predict(box=box_prompt) -> mask, score
init_video_state(video_or_stream)
add_new_points_or_box(frame_idx, obj_id, box=box_prompt)
propagate_in_video(state) -> masks_over_time
```

---

## 4. Implementation Layout

Proposed repo changes:

```text
sam2/
  training/
    distill/
      teacher_sam2.py              # teacher wrapper and output caching
      hooks.py                     # feature / memory-attention hooks
      losses.py                    # KD losses
      cache_schema.md              # cached teacher-output format
    model/
      sam2_student.py              # student model wrapper
      lightweight_encoders/
        repvit.py
        tinyvit.py
        efficientvit.py
      memory/
        spatial_perceiver.py
        sparse_memory_retrieval.py
        memory_policy.py
    dataset/
      robot_image_dataset.py
      robot_video_dataset.py
      bbox_augmentation.py
    configs/
      distill_stage1_encoder.yaml
      distill_stage2_box_prompt.yaml
      distill_stage3_short_video.yaml
      distill_stage4_long_video.yaml
      distill_stage5_memory_efficiency.yaml
  tools/
    cache_teacher_image_outputs.py
    cache_teacher_video_outputs.py
    eval_robot_box_prompt.py
    eval_robot_tracking.py
```

Keep the first version simple:

```text
1. Make the student forward pass shape-compatible with SAM2.
2. Run the student using the frozen SAM2 prompt encoder and mask decoder.
3. Verify single-image box-prompt inference.
4. Verify video inference state and memory propagation.
5. Only then add distillation losses and memory optimizations.
```

---

## 5. Data Plan

### Public data

Use public data to retain broad SAM2-like generalization:

```text
Image: SA-1B or sampled SA-1B subset
Video: SA-V, DAVIS, MOSE, YouTube-VOS, or company-approved alternatives
```

### Company robotics data

Use internal data to bias the model toward the actual deployment setting:

```text
robot manipulation videos
bin-picking videos
tabletop manipulation videos
gripper-object interaction clips
moving-camera clips
occlusion/reappearance clips
transparent or reflective objects
deformable objects
small tools and parts
same-category distractors
```

### Pseudo-labeling

Generate pseudo-labels with the teacher:

```text
VLM/detector bbox -> SAM2.1 teacher -> teacher mask/masklet
```

Use human annotation selectively for failure cases:

```text
bad teacher masks
transparent/specular objects
gripper occlusion
thin structures
object boundaries important for grasping
same-category distractors
long occlusion/reappearance
```

---

## 6. Stage 0: Baseline Integration

Before training, build the baseline robotics inference chain:

```text
VLM bbox -> SAM2.1 teacher mask -> SAM2.1 tracking
```

Then replace teacher with the untrained or pretrained student and verify API compatibility.

Deliverables:

```text
- VLM bbox to SAM2 mask demo
- SAM2 video propagation demo
- robot video evaluation set
- latency profiler
- memory profiler
- teacher output cache format
```

This stage avoids training blind. It defines the actual task contract.

---

## 7. Stage 1: Encoder-Only Distillation

### Purpose

Warm-start the lightweight image encoder so it produces SAM2-compatible image features.

This follows the MobileSAM-style idea:

```text
teacher image encoder -> teacher features
student image encoder -> student features
feature-level distillation
```

This stage is useful but not sufficient for video tracking. It only aligns the image feature distribution.

### Data

```text
SA-1B images
internal robot images
optional: frames sampled from robot videos
```

### Frozen modules

```text
freeze teacher: all modules
freeze student prompt encoder: yes
freeze student mask decoder: yes
freeze student memory modules: yes
train: student image encoder + neck/projection adapters
```

### Feature targets

Distill all features needed by downstream SAM2 modules:

```text
final image embedding
high-resolution features used by mask decoder
multi-scale features if exposed by the SAM2 image encoder
neck/projection outputs
```

If the student and teacher feature dimensions differ, add lightweight projection adapters:

```text
student feature -> 1x1 conv / linear projection -> teacher feature dimension
```

### Losses

```text
L_stage1 = lambda_mse  * MSE(z_student, z_teacher)
         + lambda_l1   * SmoothL1(z_student, z_teacher)
         + lambda_cos  * (1 - cosine_similarity(z_student, z_teacher))
         + lambda_hr   * MSE(high_res_student, high_res_teacher)
```

Initial weights:

```text
lambda_mse = 1.0
lambda_l1  = 0.5
lambda_cos = 0.1
lambda_hr  = 1.0
```

### Success criteria

```text
- student feature maps match teacher output shapes
- stable training loss
- frozen SAM2 decoder can consume student features
- box-prompt mask quality is not yet final, but not collapsed
```

Expected limitation:

```text
Single-image masks may become reasonable.
Video memory/tracking will still be weak unless trained in later stages.
```

---

## 8. Stage 2: Box-Prompt Image Distillation

### Purpose

Train the actual deployment behavior:

```text
image + VLM-style bbox -> accurate object mask
```

This stage addresses the weakness of pure encoder-only MSE: feature similarity does not guarantee correct prompt-to-mask behavior.

### Data

```text
SA-1B masks
robot image masks
teacher pseudo masks from SAM2.1
human-corrected masks for hard cases
```

### Prompt policy

Bias heavily toward box prompts because the upstream VLM provides bounding boxes.

Use prompt distribution:

```text
box prompt: 70-85%
point prompt: 10-20%
mask prompt: 0-10%
negative points / correction prompts: optional
```

### Bbox noise augmentation

Train against realistic detector noise:

```text
random shift
random scale up/down
aspect-ratio distortion
partial boxes
oversized boxes
boxes that include distractors
boxes with missing object parts
jittered boxes from VLM outputs
```

Recommended jitter range:

```text
translation: 0-20% of box size
scale:       0.8-1.4x
hard cases:  up to 30% shift or distractor inclusion
```

### Trainable modules

Start conservative:

```text
train: student image encoder, neck/projection adapters, lightweight decoder adapters
freeze initially: prompt encoder, most of mask decoder
```

Then optionally unfreeze:

```text
mask decoder IoU head
small LoRA/adapters in mask decoder
```

Avoid fully retraining the whole mask decoder at first, or the student may lose SAM2 prompt compatibility.

### Losses

```text
L_stage2 = lambda_mask  * L_mask(student_mask, target_mask)
         + lambda_logit * KD(student_logits, teacher_logits)
         + lambda_iou   * MSE(student_iou, teacher_iou_or_gt_iou)
         + lambda_bound * boundary_loss(student_mask, target_mask)
         + lambda_feat  * feature_distillation_loss
```

Where:

```text
L_mask = focal/BCE + dice
KD     = MSE on logits or KL/BCE on softened logits
```

Initial weights:

```text
lambda_mask  = 1.0
lambda_logit = 1.0
lambda_iou   = 0.5
lambda_bound = 0.1
lambda_feat  = 0.2
```

### Success criteria

Evaluate on clean and noisy boxes:

```text
box-prompt mIoU
boundary F-score
mask stability score
IoU-head calibration
performance under bbox jitter
latency per image
```

The student should be robust to VLM-style boxes before moving to video tracking.

---

## 9. Stage 3: Short-Clip Video Distillation

### Purpose

Train SAM2-like memory behavior:

```text
first-frame bbox prompt -> mask propagation over 4-8 frames
```

This is the key stage that encoder-only distillation cannot replace.

### Data

```text
4-8 frame clips
SA-V / DAVIS / MOSE / YTVOS
internal robot videos
teacher-generated masklets from SAM2.1
```

### Input protocol

For each clip:

```text
1. sample object
2. sample first-frame bbox prompt
3. teacher produces first-frame mask and propagated masklet
4. student receives the same first-frame bbox prompt
5. student propagates masks through the clip
```

Add correction prompts in a subset of clips:

```text
first-frame bbox only:           70%
first-frame bbox + later click:  20%
first-frame bbox + later box:    10%
```

### Distillation targets

Cache or hook the teacher to obtain:

```text
per-frame mask logits
per-frame selected mask
predicted IoU / confidence
memory-attention output features
object pointer features
occlusion / no-object logits if available
```

### Losses

```text
L_stage3 = lambda_mask  * sum_t L_mask(M_s[t], M_t[t])
         + lambda_logit * sum_t KD(logits_s[t], logits_t[t])
         + lambda_mem   * sum_t MSE(mem_attn_s[t], mem_attn_t[t])
         + lambda_ptr   * sum_t cosine_or_MSE(ptr_s[t], ptr_t[t])
         + lambda_occ   * BCE(occ_s[t], occ_t[t])
         + lambda_temp  * temporal_consistency_loss
```

Initial weights:

```text
lambda_mask  = 1.0
lambda_logit = 0.5
lambda_mem   = 1.0
lambda_ptr   = 0.2
lambda_occ   = 0.2
lambda_temp  = 0.1
```

### Memory-attention distillation

This is mandatory for SAM2-like tracking. Aligning only the image encoder can leave the memory module with the wrong feature geometry.

Implementation:

```python
teacher_mem = hook(teacher.memory_attention.output)
student_mem = hook(student.memory_attention.output)
L_mem = mse(project(student_mem), teacher_mem)
```

If student memory-attention dimensions differ, use a projection adapter:

```text
student memory output -> linear/1x1 projection -> teacher dimension
```

### Training tricks

```text
teacher forcing for early iterations
scheduled sampling for student-generated memories
random memory dropout
random frame dropping
motion blur / camera motion augmentation
occlusion augmentation
same-category distractor sampling
```

### Success criteria

```text
J&F over short clips
drift rate
ID switch rate
failure after mild occlusion
latency per frame
memory frames/tokens per object
```

---

## 10. Stage 4: Long-Clip Fine-Tuning

### Purpose

Improve long-horizon tracking, occlusion recovery, and robustness to robotics failure modes.

### Data

```text
16-32 frame clips
robot videos with gripper occlusion
objects leaving and re-entering view
camera ego-motion
same-category distractors
small or thin objects
deformable objects
transparent/specular objects
```

### Training policy

Start with short clips and progressively increase length:

```text
4-8 frames   -> stable short tracking
16 frames    -> medium horizon
32 frames    -> long horizon
```

Recommended module freezing:

```text
freeze image encoder after Stage 3 if stable
train memory encoder
train memory attention
train memory policy
train small decoder adapters if needed
```

### Losses

Use the Stage 3 losses, but increase emphasis on tracking stability:

```text
increase lambda_temp
increase hard-negative/distractor sampling
increase occlusion/reappearance cases
use lower teacher confidence weight for uncertain pseudo-labels
```

For teacher pseudo-labels, weight losses by teacher confidence:

```text
L_weighted = confidence_teacher * L_mask
```

### Success criteria

```text
low drift over 16-32 frames
low ID switch with distractors
recovery after occlusion
stable masks during gripper-object interaction
bounded memory consumption
```

---

## 11. Stage 5: Efficient Memory Bank

### Purpose

Make memory efficient without destroying tracking.

The memory bank should be optimized from the beginning, but Stage 5 makes it explicit and measurable.

### 5.1 Quality-based memory insertion

Do not insert every frame. Insert only useful memories.

Memory quality score:

```text
Q = w_iou      * predicted_iou
  + w_stab     * mask_stability
  + w_vis      * visibility_score
  + w_motion   * motion_novelty
  + w_boundary * boundary_confidence
  - w_redun    * redundancy_with_existing_memory
```

Keep:

```text
first prompted frame
most recent high-quality frame
large-pose-change frame
reappearance frame
high-confidence clean mask
```

Drop:

```text
low-confidence frame
heavy occlusion frame
redundant similar frame
frame with unstable mask
frame dominated by background
```

### 5.2 Memory token pruning

Prune tokens inside memory frames:

```text
keep foreground tokens
keep boundary tokens
keep high-attention tokens
keep tokens near uncertain regions
prune easy background tokens
```

Start with deterministic pruning:

```text
foreground mask dilation -> token keep mask
boundary band -> token keep mask
uniform background subsampling
```

Then add learned pruning:

```text
small token scorer -> top-k memory tokens
```

### 5.3 Sparse memory retrieval

Instead of attending to all memory tokens, retrieve a small relevant subset:

```text
current frame query tokens
        -> similarity to memory tokens
        -> top-k tokens per object / per memory frame
        -> memory attention over selected tokens only
```

Use constraints:

```text
max_memory_frames_per_object = N
max_tokens_per_memory_frame = K
max_total_tokens_per_object = B
```

Example starting values:

```text
N = 4-8 memory frames per object
K = 64-256 tokens per memory frame
B = 512-1024 total memory tokens per object
```

### 5.4 Memory dropout during training

Train the model to survive missing or degraded memory:

```text
randomly drop memory frames
randomly drop memory tokens
randomly corrupt low-confidence memories
randomly skip memory insertion
```

This prevents the student from overfitting to a perfect teacher memory bank.

### 5.5 Memory-specific evaluation

Report quality-efficiency curves:

```text
J&F vs memory frames per object
J&F vs memory tokens per object
latency vs memory tokens
VRAM/RAM vs tracked objects
occlusion recovery vs memory budget
```

---

## 12. Evaluation Protocol

### Image segmentation

```text
box-prompt mIoU
boundary F-score
mask AP if applicable
IoU-head calibration
stability score
robustness under bbox jitter
```

Run jitter sweeps:

```text
0% box noise
5% box noise
10% box noise
20% box noise
30% box noise
```

### Video tracking

```text
J&F
Jaccard IoU over time
drift rate
ID switch rate
occlusion recovery rate
reappearance success rate
mask flicker / temporal instability
```

### Robotics-specific tests

```text
gripper occlusion
object leaving/re-entering view
camera ego-motion
multiple same-category objects
transparent/specular objects
small/thin objects
bin-picking clutter
deformable objects
```

### Efficiency

```text
FPS / latency per frame
first-frame prompt latency
propagation latency
peak GPU memory
CPU memory
memory tokens per object
max tracked objects at target FPS
power if on edge device
```

Target metrics should be defined per deployment hardware:

```text
server GPU: A10/A100/L4/4090/etc.
edge GPU: Jetson Orin / Orin Nano
mobile: iPhone / Android NPU if needed
robot compute: actual onboard target
```

---

## 13. Milestones

### Milestone 0: Baseline pipeline

```text
VLM bbox -> SAM2.1 teacher mask -> SAM2.1 tracking
robot evaluation set ready
latency/memory profiler ready
```

### Milestone 1: Student shape compatibility

```text
lightweight encoder outputs SAM2-compatible features
frozen SAM2 decoder accepts student features
single-image box prompt runs end-to-end
```

### Milestone 2: Encoder-only distillation

```text
student feature loss converges
student+frozen decoder produces non-collapsed masks
feature cache and training loop stable
```

### Milestone 3: Box-prompt image distillation

```text
student reaches acceptable box-prompt mIoU
robust to VLM-style bbox jitter
latency meets single-image target
```

### Milestone 4: Short-video distillation

```text
4-8 frame tracking works
memory-attention KD implemented
short-clip J&F close to teacher/student target
```

### Milestone 5: Long-video robotics fine-tuning

```text
16-32 frame tracking stable
occlusion/reappearance recovery improves
ID switches reduced on distractor-heavy robot videos
```

### Milestone 6: Efficient memory

```text
bounded memory bank
quality-based insertion
token pruning
sparse retrieval
quality-efficiency Pareto curve reported
```

### Milestone 7: Deployment

```text
export ONNX/TensorRT/CoreML if needed
profile target hardware
optional INT8/QAT after quality is stable
```

---

## 14. Recommended Training Order

Do not start with memory optimization alone. Use this order:

```text
1. Reproduce SAM2 teacher pipeline on robot videos.
2. Implement student shape compatibility.
3. Train encoder-only distillation.
4. Train box-prompt image distillation.
5. Train short-clip video memory distillation.
6. Fine-tune on long robot clips.
7. Add and train efficient memory policies.
8. Export and optimize for deployment.
```

The most important checkpoint is after Stage 3:

```text
first-frame VLM bbox -> student mask -> student tracking over short clips
```

If this does not work, Stage 5 memory pruning will only make tracking worse.

---

## 15. Main Risks and Mitigations

### Risk 1: Encoder-only distillation looks good but tracking fails

Mitigation:

```text
add Stage 2 prompt-in-the-loop image distillation
add Stage 3 memory-attention distillation
evaluate tracking early, not only image masks
```

### Risk 2: Student features are shape-compatible but distribution-incompatible

Mitigation:

```text
feature distill multiple levels
use projection adapters
freeze decoder early
unfreeze decoder adapters only after stable masks
```

### Risk 3: Teacher pseudo-labels are wrong on robotics data

Mitigation:

```text
confidence-weight losses
human-correct hard cases
filter low-stability teacher masks
use multi-prompt teacher refinement
```

### Risk 4: Memory pruning causes irreversible drift

Mitigation:

```text
keep first prompted frame permanently
keep recent high-quality frame
train with memory dropout
compare against no-pruning baseline
use quality-efficiency curves
```

### Risk 5: VLM boxes are noisy

Mitigation:

```text
heavy bbox jitter training
partial-box augmentation
distractor-in-box augmentation
evaluate box-noise sweep
```

---

## 16. Minimal Viable Version

The minimal version should not attempt to be novel. It should prove the pipeline:

```text
Teacher: SAM2.1-B+ or SAM2.1-Large
Student: EdgeTAM-like lightweight SAM2
Training scaffold: official SAM2 training code
Stage 1: encoder-only distillation
Stage 2: box-prompt image distillation
Stage 3: 4-8 frame video memory distillation
Evaluation: internal robot benchmark + latency/memory profiling
```

Only after this works should we add:

```text
long-clip fine-tuning
quality-based memory insertion
memory token pruning
sparse memory retrieval
quantization / TensorRT / CoreML export
```

---

## 17. References

- Official SAM2 training code: https://github.com/facebookresearch/sam2/blob/main/training/README.md
- SAM2 paper: https://arxiv.org/abs/2408.00714
- EdgeTAM repository: https://github.com/facebookresearch/EdgeTAM
- EdgeTAM paper: https://arxiv.org/abs/2501.07256
- EdgeTAM training-code issue: https://github.com/facebookresearch/EdgeTAM/issues/8
- EfficientTAM repository: https://github.com/yformer/EfficientTAM
- EfficientTAM training-code issue: https://github.com/yformer/EfficientTAM/issues/1
- EdgeSAM repository: https://github.com/chongzhou96/EdgeSAM
- EdgeSAM project page: https://mmlab-ntu.github.io/project/edgesam/
- MobileSAM repository: https://github.com/ChaoningZhang/MobileSAM
