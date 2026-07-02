# Company Cluster Login: Hugging Face And W&B

Use this on each new company node/session if Hugging Face or W&B cannot find existing credentials.

## 1. Activate Environment

```bash
export SAM2D_ENV=/user-volume/env/sam2_stage1_torch24
source $SAM2D_ENV/bin/activate

python -m pip install -r /user-volume/repo/SAM2-Distillation-Pipeline/requirements-stage1.txt
```

`requirements-stage1.txt` includes both:

```text
huggingface_hub
wandb
```

## 2. Recommended Persistent Cache Locations

Set these before login so credentials/cache are stored under persistent user storage, not node-local scratch:

```bash
export HF_HOME=/user-volume/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export WANDB_CONFIG_DIR=/user-volume/.config/wandb
export WANDB_CACHE_DIR=/user-volume/.cache/wandb
export WANDB_DATA_DIR=/user-volume/.cache/wandb-data

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
```

If company nodes still do not share these paths reliably, put the exports in your job script and login again on the new node.

## 3. Hugging Face Login

Create a read token:

```text
https://huggingface.co/settings/tokens
```

Interactive login:

```bash
huggingface-cli login
huggingface-cli whoami
```

Non-interactive login for jobs:

```bash
export HF_TOKEN='hf_xxx'
huggingface-cli login --token "$HF_TOKEN"
huggingface-cli whoami
```

Minimal download test:

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="timm/tiny_vit_21m_512.dist_in22k_ft_in1k",
    filename="model.safetensors",
)
p = Path(path)
print("path:", p)
print("exists:", p.exists())
print("size_mb:", round(p.stat().st_size / 1024**2, 2))
PY
```

Copy TinyViT into the pipeline path:

```bash
export SAM2D_ROOT=/group-volume/danny-dataset/sam2_distill
mkdir -p $SAM2D_ROOT/checkpoints/tinyvit

cp "$(python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    repo_id="timm/tiny_vit_21m_512.dist_in22k_ft_in1k",
    filename="model.safetensors",
))
PY
)" "$SAM2D_ROOT/checkpoints/tinyvit/tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors"
```

## 4. W&B Login

Get API key:

```text
https://wandb.ai/authorize
```

Interactive login:

```bash
wandb login
wandb status
```

Non-interactive login for jobs:

```bash
export WANDB_API_KEY='...'
wandb login "$WANDB_API_KEY"
wandb status
```

Minimal online logging test:

```bash
python - <<'PY'
import wandb

run = wandb.init(project="sam2-distill-smoke", name="company-login-smoke")
wandb.log({"ok": 1})
run.finish()
PY
```

If online logging fails during a job, use offline mode and sync later:

```bash
export WANDB_MODE=offline
# run training
wandb sync wandb/offline-run-*
```

## 5. Put This In Job Scripts

For repeatable jobs, include:

```bash
source /user-volume/env/sam2_stage1_torch24/bin/activate

export HF_HOME=/user-volume/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export WANDB_CONFIG_DIR=/user-volume/.config/wandb
export WANDB_CACHE_DIR=/user-volume/.cache/wandb
export WANDB_DATA_DIR=/user-volume/.cache/wandb-data
export WANDB_PROJECT=sam2-distill-stage1
```

If the node cannot see saved credentials:

```bash
export HF_TOKEN='hf_xxx'
export WANDB_API_KEY='...'
huggingface-cli login --token "$HF_TOKEN"
wandb login "$WANDB_API_KEY"
```
