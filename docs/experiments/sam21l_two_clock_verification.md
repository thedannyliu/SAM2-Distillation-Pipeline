# SAM2.1-L two-clock verification and promotion system

This system separates inexpensive logic verification from real SAM2/H100
integration verification. A process reporting completion is not promotion
evidence. Only the machine-readable stamps and post-run audit defined here may
promote a new run.

## Scope and trust boundary

PACE owns implementation, review, and CPU tests. The company platform owns
the real SAM2.1-L forward/backward, four-rank DDP smoke, data-path audit, and
formal training. The company platform reruns the short CPU test gate so the
local and remote stamps describe the same clean checkout.

The hashes protect against accidental code, config, dataset, SAM2 checkout,
or initializer drift. They are not a cryptographic signature against a person
who intentionally edits both code and evidence.

The binding machine-readable requirements are in
`configs/verification/sam21l_two_clock_reuse_v1.json`. Every requirement has a
design statement, code location, automated test, and runtime evidence field.

## State machine

New runs follow this sequence:

```text
LOCAL_LOGIC_VERIFIED
        -> SMOKE_ELIGIBLE
        -> REMOTE_INTEGRATION_VERIFIED
        -> FORMAL_ELIGIBLE
        -> FORMAL_RUNNING
        -> RESULT_ACCEPTED or RESULT_QUARANTINED
```

The launch manifest binds the following identity fields:

- experiment-repository commit, branch, and clean state;
- official SAM2 commit and clean state;
- design, verification-contract, and training-config hashes;
- dataset-manifest, official SAM2 config, and initializer checkpoint hashes;
- target, seed, GPU assignment, W&B project, and W&B mode.
- formal epochs, data limit, T8/batch/object contract, encoder freeze, and all
  registered learning rates.

Changing any bound input invalidates the prior remote stamp. There is no
`--force` bypass.

## Current Wave-1 migration

The O2/A/B/C/D/E jobs that began at `d8a5e7c` predate this gate. Do not pull,
switch branches, or edit files in their active company checkout while the
loaded runner may still start training or evaluation subprocesses. If one of
those jobs needs an immediate resume, resume it from the original `d8a5e7c`
checkout first.

After a target reaches a safe boundary, audit it without claiming prospective
verification:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh audit-existing E
```

Repeat for each completed or interrupted target. The command writes:

```text
<run>/<target>/verification/retroactive_audit.json
<run>/<target>/verification/retroactive_audit.md
```

The audit distinguishes evidence observed during the run from the current
checkout identity. Missing launch-time SHA or preflight stamps remain
`unavailable_evidence`; they are never inferred from the current checkout.
Legacy results remain `provisional` or become `quarantined` if a recorded
failure is found.

A new verified E smoke may be run after the active jobs stop using the shared
checkout. It provides real-stack evidence for the new commit, but does not
retroactively convert the old jobs into prospectively verified runs.

## PACE development gate

During development, run the focused tests directly:

```bash
python -m pytest \
  tests/test_two_clock_core.py \
  tests/test_two_clock_data.py \
  tests/test_two_clock_training_loss.py \
  tests/test_two_clock_company_runner.py \
  tests/test_two_clock_verification.py
```

After committing, the same clean-commit gate can issue a local stamp when the
company run root is mounted. On the company platform this is normally invoked
automatically by `smoke E`:

```bash
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh verify-local
```

## Company remote gate

Run the registered smoke only on E:

```bash
cd /user-volume/repo/SAM2-Distillation-Pipeline
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh smoke E
```

This command runs, in order:

1. the local CPU requirement tests;
2. the immutable full-data input audit;
3. a fresh four-GPU E smoke under
   `verification/smoke/<git-sha>/E`;
4. optimizer membership and teacher-isolation audit;
5. per-rank temporal-mechanism gradient diagnostics;
6. per-rank T8 flight-recorder validation;
7. remote-stamp issuance.

The commit-namespaced smoke directory prevents an old checkpoint from
satisfying a new commit's integration gate. Each trace records action, raw and
source frame, age, encoder call, spatial write, and pointer write for every
frame. The gate requires at least one real reuse frame and at least one
nonzero gradient for every temporal mechanism enabled by E.

The expected top-level evidence is:

```text
<run>/verification/local_stamp.json
<run>/verification/local_pytest.xml
<run>/verification/remote_stamp.json
<run>/verification/smoke/<git-sha>/E/verification/launch_manifest.json
```

## Future formal runs

After `smoke E` succeeds, `train` and `run` record and validate a per-target
formal launch manifest before `torchrun` starts:

```bash
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh run E
```

If code or immutable inputs changed after the smoke, the command refuses to
launch. Resuming the same target also requires its existing launch manifest to
match exactly, including seed and W&B mode.

After training, checkpoint selection, and R1--R6 validation finish, run:

```bash
GPUS=0,1,2,3 scripts/company/75_run_sam21l_two_clock_reuse_v1.sh postrun E
```

The post-run audit requires complete training, finite gradients, a passing
optimizer audit, all five epoch checkpoints, `best.pt`, the R4 selection
record, and passing full-validation results for R1--R6. It writes
`verification/postrun_audit.json` and returns nonzero when the result must be
quarantined.

## Failure handling

- Preserve failed logs and artifacts; do not overwrite them to obtain a pass.
- Fix company-only bugs in the canonical PACE checkout, commit a new SHA, and
  rerun the affected local and remote gates.
- A new SHA always receives a new commit-namespaced smoke directory.
- Do not manually edit stamps. Regenerate them through the verification tool.
- `RETROACTIVE_AUDIT_INCOMPLETE` is not equivalent to failure, but it is also
  not prospective proof.
