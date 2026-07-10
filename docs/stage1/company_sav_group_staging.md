# Stage Complete SA-V in Group Volume Before Data-Lake Transfer

This gate does not transfer or delete data. It first stages and verifies one
complete source copy under:

```text
/group-volume/danny-dataset/SA-V/sav_train
/group-volume/danny-dataset/SA-V/sav_val
/group-volume/danny-dataset/SA-V/sav_test
/group-volume/danny-dataset/sam2_distill/data/sav_v2/frame_cache/stage1_vbal16_6fps
```

The source is the currently mounted released dataset:

```text
/mnt/data/danny-dataset/SA-V
```

## 1. Restore and inspect the source release

The source mount must contain these sentinels:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline

scripts/company/30_stage_complete_sav_in_group.sh preflight
```

Preflight prints source sizes and available group-volume capacity. Do not start
the copy if the source sentinels fail or available capacity is insufficient.

## 2. Synchronize raw official splits

```bash
scripts/company/30_stage_complete_sav_in_group.sh sync-raw
```

This uses resumable `rsync` and does not delete target extras or source files.
Rerunning the command continues or verifies already copied files.

## 3. Materialize stable validation frames

```bash
scripts/company/30_stage_complete_sav_in_group.sh materialize-val
```

This reuses all existing train rows and copies only the selected official
validation frames into `JPEGImages_val_sav`, preventing train/val video-ID
collisions and removing `/mnt` dependencies from Stage 1 validation.

## 4. Full audit

```bash
scripts/company/30_stage_complete_sav_in_group.sh audit
```

The audit checks:

- every source relative file exists in group-volume with the same size;
- train MP4/manual/auto annotation inventory;
- every listed val/test video has image and annotation directories;
- val/test image and mask counts are nonzero;
- processed manifest has 807,248 train and 1,240 val rows;
- processed manifest has 50,453 train and 155 val videos with no overlap;
- all 808,488 processed image paths exist under group-volume;
- 200 deterministic processed-image samples decode successfully.

Success creates:

```text
/group-volume/danny-dataset/sam2_distill/migration_reports/
  group_sav_completeness.json
  dataset_complete.ready
```

Do not begin the data-lake transfer unless `dataset_complete.ready` exists.

To run all three mutating/checking steps after preflight:

```bash
scripts/company/30_stage_complete_sav_in_group.sh all
```
