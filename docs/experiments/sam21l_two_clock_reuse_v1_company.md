# SAM2.1-L two-clock Wave-1 company runbook

This runbook launches the approved O2/A/B/C/D/E screen. It does not access
SA-V test, DAVIS, E+, or the learned gate. Commands run in the foreground and
the runner mirrors live output to `/user-volume/log/sam21l_two_clock_reuse_v1`.

## Immutable runtime contract

- Code: `/user-volume/repo/SAM2-Distillation-Pipeline`
- Official SAM2: `/user-volume/repo/facebookresearch-sam2`
- Container: `ngc24.06/ub22/py3.10/cu12.5/cudnn9.1/pytorch2.4`
- Branch: `research/sam21l-two-clock-reuse-v1`
- Implemented commit at handoff: `2dacf06` or a later commit on that branch
- GPUs per process group: `0,1,2,3`
- W&B project: `sam2-two-clock-reuse-v1`
- Run root:
  `/group-volume/danny-dataset/sam2_distill/runs/sam21l_two_clock_reuse_v1`
- Dataset manifest:
  `/group-volume/danny-dataset/sam2_distill/manifests/sav_train_6fps_full.parquet`

Do not create a venv or upgrade container PyTorch. The audit imports the
mounted official SAM2 checkout, and the E smoke is the actual Torch 2.4
forward/backward compatibility gate. If either fails, stop and preserve the
log; pin a compatible official SAM2 commit or request a Torch 2.5.1 company
image instead of modifying the runtime silently.

Before entering the container, make this branch available in the company
checkout through the team's normal Git transport. The local development
handoff does not itself publish or merge the branch.

## One-node hard gate

Run these first on one 4xH100 node:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
git switch research/sam21l-two-clock-reuse-v1
git rev-parse --short HEAD
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh audit
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh smoke E
```

The audit must report 50,337 usable videos, successful sampled MP4 T8 decode,
the official Hiera-L checkpoint tensor prefixes, and the actual imported
Torch/SAM2 paths. The smoke must complete four-rank DDP forward/backward and
write all of the following:

- `smoke/E/checkpoints/checkpoint.pt` and `last.pt`;
- `smoke/E/resolved_config.yaml`;
- `smoke/E/capacity_rank*.json`;
- `smoke/E/gradient_diagnostics.json`;
- `smoke/E/training_status.json` with `status: complete`.

Do not start formal jobs if the smoke fails, any rank reports non-finite
gradients, the resolved per-GPU batch differs from one, or the input audit
does not match the registered data contract.

## Six-node Wave 1

After the hard gate passes, paste one command into each of six company
terminals. Each command stays attached to its terminal and is resumable by
running the identical command again.

Node 1:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh run O2
```

Node 2:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh run A
```

Node 3:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh run B
```

Node 4:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh run C
```

Node 5:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh run D
```

Node 6:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh run E
```

For each target, `run` performs:

1. the shared, locked full-data audit;
2. five full SA-V epochs from the official SAM2.1-L initializer;
3. full SA-V val R4 evaluation for `checkpoint_1.pt` through
   `checkpoint_5.pt`;
4. selection of `best.pt` by maximum R4 J&F only;
5. full SA-V val evaluation of the selected checkpoint at R1--R6;
6. official aggregate J/F plus GT-only realized-age metrics and actual
   logical encoder-call counts.

The selected R4 pass is intentionally rerun under the selected-output folder
so every R1--R6 artifact has the same checkpoint path and reporting layout.
No test data is touched.

## Evaluation-only controls

Run controls on the first node that becomes free:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh controls
```

This evaluates official O0 at R1 and the frozen official O1 at R2--R6. O1 R1
is exactly O0 and is not recomputed. The causal schedule-exposure comparison
is A versus O2 at matched R2--R6, not A versus O1.

## Monitoring and targeted resume

Status is read-only:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
scripts/company/75_run_sam21l_two_clock_reuse_v1.sh status
```

To resume only one phase, use `train`, `select`, or `curves` with the same
target. Examples:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh train C
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh select C
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh curves C
```

Training reuses `checkpoints/checkpoint.pt`, the saved W&B run ID, the same
TensorBoard directory, and the same checkpoint directory. Evaluation skips
only an output whose `sav_eval.json` explicitly records `status: pass`.

## Promotion boundary

After all six runs and controls finish, update the result ledger in
`docs/experiments/sam21l_two_clock_reuse_v1.md`. Run C-r only if C exceeds B
at the registered R4 metric. Do not implement or launch E+ unless E passes its
registered comparison with D. Do not touch SA-V test until the architecture,
checkpoint, prompt protocol, and policy are frozen from validation.
