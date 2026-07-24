# EdgeTAM fidelity ladder v3

## Question

Can the official EdgeTAM temporal architecture first be reproduced under the
local SA-V evaluator, and then transferred to TinyViT-21M without conflating
evaluator, topology, initialization, and optimization failures?

The answer is not established by the current hybrid results. M2/C0 imported
subsets of official weights into an A02-derived model; neither experiment
first demonstrated that the unmodified official checkpoint works in the same
runtime and evaluator.

## Evidence from the completed memory suites

| Model | Temporal topology | val J&F | Gate J&F |
| --- | --- | ---: | ---: |
| M0 | four uncompressed SAM2 attention blocks | 71.5 | 71.6 |
| M1 | two truncated SAM2 attention blocks | 53.3 | not run |
| M2a | partial official EdgeTAM transfer | 15.6 | not run |
| C0 | coherent temporal transfer + M0 `F_M` MSE | not run | 31.5 |
| C1 | partial transfer + M0 `F_M` MSE | not run | 31.1 |

The prompted-image metrics of C0/C1 match M0 while tracking loses about 40
points. A final memory-conditioned feature MSE therefore does not constrain
the internal memory tokens, temporal positions, object pointer, and attention
dynamics tightly enough.

## Fidelity ladder

### E0: unmodified official upstream

Use the released `facebook/EdgeTAM` `edgetam.pt` and `edgetam.yaml` without a
TinyViT swap or A02 tensors. Run the exact local box-prompt evaluator on a
fixed 32-video SA-V val sample. Require J&F >= 55 before full val/test.

This threshold is a runtime-integrity gate, not a claim of paper
reproduction. A failure means the local configuration, checkpoint loading, or
evaluation contract must be fixed before interpreting any hybrid.

Implemented entry point:

`scripts/company/53_run_edgetam_official_fidelity.sh gate|all`

### E1: zero-training encoder swap

Only after E0 passes, instantiate the official topology and load every
non-image tensor from the official checkpoint. Replace only the image encoder
with the selected TinyViT-21M Stage 1 interface. Run the same 32-video gate
without training.

This differs materially from C0: C0 retained the A02 prompt encoder and mask
decoder. E1 isolates whether the TinyViT interface is compatible with a
coherent released EdgeTAM temporal stack.

### E2: same-topology image-interface alignment

If E1 loses image quality or first-frame quality, freeze the complete official
non-image model and align TinyViT image/high-resolution features to the
official EdgeTAM teacher. Do not train memory yet. Re-run the gate and require
both image drops <= 0.005 before moving on.

### E3: temporal behavior distillation

If E1/E2 preserve prompted images but lose propagation, train T4 first with:

- normal focal/Dice/IoU/object task losses;
- teacher mask-logit targets on propagated frames;
- object-pointer cosine/MSE alignment;
- compressed memory-token alignment before attention; and
- mapped layer-output alignment from student blocks 1/2 to teacher temporal
  checkpoints.

Final `F_M` MSE remains an auxiliary loss, not the only temporal target.
Start with two full SA-V passes and gate after each pass. Stop candidates
below 60 gate J&F.

### E4: longer temporal context

Only a candidate that reaches at least 60 gate J&F proceeds to T8 with up to
three objects. T16/T32 is downstream refinement with the encoder frozen.
R3 already shows that longer clips do not repair a broken T4 interface.

## Data-scale interpretation

The official video stage uses 130K updates at batch 256 and T8, plus a mixed
SA-V/SA-1B/DAVIS/MOSE/YTVOS corpus. One local SA-V epoch is only about
1/1,322 of that frame exposure. The local objective is therefore a faithful
transfer study under restricted data, not a from-scratch reproduction of the
official training budget.

## Current execution decision

Use the two additional four-H100 nodes for independent E0 checks while the
three TinyViT size fine-tunes continue:

- primary seed `edgetam-memory-gate-v2`: gate, then full val/test on pass;
- replication seed `edgetam-fidelity-v3-seed2`: independent gate only.

Both checks use separate run directories and W&B runs. Do not spend this
block on C2, C3, another Perceiver LR sweep, or T8. E1-E4 remain gated;
implement and launch each only when its predecessor identifies the next
failure boundary.

The checks subsequently passed at 65.2 and 72.1 gate J&F. The primary run
reached 68.0 full-val J&F. This validates the local runtime/evaluator and
unblocks the strict E1 plus behavior-transfer suite documented in
`docs/experiments/edgetam_tinyvit21_behavior_v4.md`.

W&B project: `edgetam-fidelity-v3`

Run root:

`/group-volume/danny-dataset/sam2_distill/runs/edgetam_fidelity_v3`
