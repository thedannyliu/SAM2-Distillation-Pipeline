# SA-V Mount Failure Recovery

## Symptom

Training continues normally on cached train frames, then fails at validation or
post-training VOS evaluation with missing files below:

```text
/mnt/data/danny-dataset/SA-V/sav_val/...
/mnt/data/danny-dataset/SA-V/sav_test/...
```

This means the active node no longer sees the data-lake dataset version used to
build the manifest. It is not a model or CUDA failure.

## Verify the mount

```bash
findmnt -T /mnt/data/danny-dataset
ls -lh /mnt/data/danny-dataset/SA-V/sav_val/JPEGImages_24fps/sav_000262/00060.jpg
ls -lh /mnt/data/danny-dataset/SA-V/sav_test/Annotations_6fps/sav_013624/000/00000.png
```

If either file is unavailable, attach or mount the same released data-lake
dataset version that contains `SA-V/sav_val` and `SA-V/sav_test` before continuing.

## Materialize stable validation images

After the source mount is restored, update the code and rebuild only validation.
Train rows are reused and are not re-extracted.

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git pull origin edgetam-tinyvit-pipeline

MANIFEST=/group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps.parquet
cp -p "$MANIFEST" "${MANIFEST}.before_val_materialization"

DATA_ROOT=/group-volume/danny-dataset \
SAV_ROOT=/mnt/data/danny-dataset/SA-V \
CACHE_NAME=stage1_vbal16_6fps \
MANIFEST="$MANIFEST" \
REUSE_TRAIN_MANIFEST="$MANIFEST" \
TRAIN_FRAMES_PER_VIDEO=16 \
VAL_FRAMES_PER_VIDEO=8 \
TEST_FRAMES_PER_VIDEO=0 \
NUM_WORKERS=64 \
scripts/company/18_prepare_sav_stage1_frame_cache.sh

touch /group-volume/danny-dataset/sam2_distill/manifests/sav_stage1_vbal16_6fps.done
```

The corrected `val_sav` rows point to stable files below:

```text
/group-volume/danny-dataset/sam2_distill/data/sav_v2/frame_cache/
  stage1_vbal16_6fps/JPEGImages_val_sav/<video>/<frame>.jpg
```

Validate before restarting jobs:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

manifest = Path(
    "/group-volume/danny-dataset/sam2_distill/manifests/"
    "sav_stage1_vbal16_6fps.parquet"
)
df = pd.read_parquet(manifest)
val = df[df["split"] == "val_sav"]
missing = [path for path in val["image_path"] if not Path(path).is_file()]
print("val rows:", len(val))
print("missing:", len(missing))
print("path root:", val.iloc[0]["image_path"])
assert len(val) == 1240
assert not missing
assert val["image_path"].str.contains("JPEGImages_val_sav").all()
print("stable val cache: PASS")
PY
```

Restart the original queue command. Runs with `last.pt` resume the same W&B run.
The complete `sav_test` image and annotation trees are still required for image
segmentation and VOS benchmarks; missing test masks must not be skipped.
