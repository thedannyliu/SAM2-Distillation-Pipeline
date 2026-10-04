# Recorded results and scope

Source: the project owner's `SAM2-distill-finetune-danny` handoff. These numbers
summarize prior runs; raw checkpoints and evaluation outputs are not bundled.
Detailed experiment protocols remain under `stage1/` and `stage2/`.

| Student | Validation J&F | Test J&F | Test image mIoU |
| --- | ---: | ---: | ---: |
| TinyViT-21M | 72.4 | 75.5 | 0.837 |
| TinyViT-11M | 68.6 | 71.5 | 0.816 |
| TinyViT-5M | 66.0 | 69.1 | 0.803 |
| RepViT-M0.9 | 61.4 | 60.0 | 0.758 |

Use validation for selection. These SA-V results are not directly comparable
with published multi-dataset training results. The historical driver also
emits test metrics per stage; avoid using those metrics for model selection.

## Findings to preserve

- Decoder-only tuning did not improve the recorded validation result; unfreezing
  temporal memory did. Keep the trainable scope explicit in every run.
- High-resolution feature supervision mattered: removing it reduced image mIoU
  and video J&F. Preserve all three teacher feature targets.
- The compressed-memory lane did not meet its promotion gate. The handoff
  reports its best candidate at 55.6 validation J&F against a 68.8 gate.
  It remains experimental.
- Bucketed multi-object execution is a runtime optimization. On one H100 and
  one 580-frame video, the recorded gains were 42.0% at four objects and 47.5%
  at eight. Minimum mask agreement was 0.9690, not bit-exact. This is neither
  broad equivalence validation nor a Thor performance result.

Reproduce results using the exact checkpoint, resolved config, data manifest,
prompt protocol and evaluator revision. Report per-object identity failures
as well as aggregate overlap.
