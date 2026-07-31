# Full SA-V data for memory training

## Purpose

The existing `sav_stage1_vbal16_6fps_group_runtime.parquet` cache keeps 16
annotation-aligned frames per train video. It is a controlled dataset for the
current multiplex comparisons, but it limits the temporal windows available to
SAM2 memory training.

The full-memory preparation keeps every frame on the 6 FPS annotation cadence.
It does not expand the videos to redundant 24 FPS JPEGs. The result supports
T4/T8/T16 task, decoder, memory, shared-K/V, and object-slot experiments.

## Outputs

| Artifact | Company path |
|---|---|
| Raw train release | `/group-volume/danny-dataset/SA-V/sav_train` |
| Full 6 FPS frame cache | `/group-volume/danny-dataset/SA-V/sav_train_6fps_full/JPEGImages` |
| Training manifest | `/group-volume/danny-dataset/sam2_distill/manifests/sav_train_6fps_full.parquet` |
| Dense-2 cohort | `/group-volume/danny-dataset/sam2_distill/cohorts/sav_train_6fps_full/dense2_unique.txt` |
| Dense-4 cohort | `/group-volume/danny-dataset/sam2_distill/cohorts/sav_train_6fps_full/dense4_unique.txt` |
| Dense-8 cohort | `/group-volume/danny-dataset/sam2_distill/cohorts/sav_train_6fps_full/dense8_unique.txt` |

The raw release gate expects 50,453 MP4 files, 50,452 manual JSON files, and
48,306 auto JSON files. The one video without a manual annotation remains in
the manifest but is excluded by the SAM2 task dataset.

## Preparation

The workflow is resumable. S3 downloads use temporary files and size checks,
and frame extraction skips existing JPEGs.

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull --ff-only
mkdir -p /user-volume/sav_full_memory_logs

# Set this only when the company Data Lake requires a custom endpoint.
# export S3_ENDPOINT_URL="https://company-s3-endpoint"

scripts/company/65_prepare_full_sav_memory_data.sh all 2>&1 | \
tee "/user-volume/sav_full_memory_logs/all_$(date +%Y%m%d_%H%M%S).log"
echo "Full preparation status: ${PIPESTATUS[0]}"
```

The same workflow can be resumed one stage at a time:

```bash
scripts/company/65_prepare_full_sav_memory_data.sh sync
scripts/company/65_prepare_full_sav_memory_data.sh prepare
scripts/company/65_prepare_full_sav_memory_data.sh audit
scripts/company/65_prepare_full_sav_memory_data.sh cohorts
```

`audit` checks manifest cardinality, frame cadence, duplicate IDs, cache
cardinality, sampled image paths, sampled manual annotations, and the actual
SAM2 task-data adapter.

## Training selection

Set the full manifest explicitly so controlled experiments do not silently
switch datasets:

```bash
MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/sav_train_6fps_full.parquet
```

For general memory training, omit `TASK_VIDEO_IDS_FILE` to use every video with
a readable manual annotation. For multi-object training, use a unique dense
cohort:

```bash
TASK_VIDEO_IDS_FILE=/group-volume/danny-dataset/sam2_distill/cohorts/sav_train_6fps_full/dense4_unique.txt
```

Dense-8 remains the strict multiplex cohort. Dense-4 is the recommended
diversity expansion because it adds unique videos instead of merely repeating
the small dense-8 set.

Do not mix the full manifest into the current v4 architecture screen. Finish
that fixed-data comparison first, then compare the selected architecture on the
16-frame and full-6-FPS manifests with matched optimizer updates.
