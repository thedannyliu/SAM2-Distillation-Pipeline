# SAM2.1-L adjacent image-embedding drift (20-video diagnostic)

## Question

How redundant are consecutive SA-V frames after each frame is independently
encoded by the official SAM2.1-L image encoder?

This diagnostic does not reuse features and does not invoke memory attention or
the mask decoder. Every frame receives a fresh image-encoder call. The measured
tensor is the post-neck SAM2 image embedding exposed as
`SAM2ImagePredictor._features["image_embed"]`, with shape `[256, 64, 64]` per
frame. This is the deep visual observation consumed by the temporal model.

## Sampling and metrics

- Split: full `sav_val` 24 FPS JPEG frames.
- Videos: 20 selected deterministically with seed `250107256`, five from each
  frame-count quartile. `selection.json` records the exact set.
- Comparisons: every current frame against its immediately preceding frame.
- Cosine similarity: global cosine after flattening the complete embedding.
- MSE: mean squared error over all embedding elements.
- Symmetric NMSE: MSE divided by the mean energy of the two embeddings,
  `0.5 * (mean(F_t^2) + mean(F_{t-1}^2))`. This makes MSE more comparable across
  videos without privileging either frame.
- Encoder inference: bf16 autocast by default; extracted features are converted
  to float32 before all metric calculations.

The analysis is descriptive evidence for visual redundancy. High adjacent
similarity alone does not prove that segmentation can safely reuse an old
embedding; that claim still requires the R1--R6 J&F and latency evaluation.

## Visual outputs

| Output | Research question |
|---|---|
| `cosine_small_multiples.png` | Where does each video's adjacent cosine similarity drop? |
| `mse_small_multiples.png` | Where does raw embedding MSE spike? |
| `nmse_small_multiples.png` | Where does scale-normalized feature change spike? |
| `embedding_drift_heatmap.png` | Which videos and frame transitions are least redundant? |
| `cosine_distance_vs_nmse_hexbin.png` | Are cosine distance and NMSE redundant diagnostics? |
| `adjacent_embedding_metrics.csv` | Exact per-transition values for follow-up analysis. |
| `per_video_summary.csv` | Comparable per-video aggregate statistics. |

Small multiples preserve every transition without overlaying 20 unreadable
curves. The heatmap pads shorter videos with blank cells; it does not resample
or normalize time.

## Company command

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only origin research/sam21l-two-clock-reuse-v1
mkdir -p /user-volume/log/sam21l_two_clock_reuse_v1

GPU=0 scripts/company/76_analyze_sam21l_embedding_drift.sh 2>&1 | \
  tee /user-volume/log/sam21l_two_clock_reuse_v1/embedding_drift_20_v1.log
echo "embedding drift status: ${PIPESTATUS[0]}"
```

Outputs are written to:

`/group-volume/danny-dataset/sam2_distill/runs/sam21l_two_clock_reuse_v1/embedding_drift_20_v1`

## Results

Pending company H100 execution.
