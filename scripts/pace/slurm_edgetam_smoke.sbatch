#!/usr/bin/env bash
#SBATCH --job-name=edgetam-smoke
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/edgetam-smoke-%j.out

set -euo pipefail

ROOT="${ROOT:-/storage/project/r-agarg35-0/eliu354/projects/SAM2-Distillation-Pipeline}"
ENV_PREFIX="${ENV_PREFIX:-${ROOT}/.conda/edgetam}"
TASK="${TASK:-probe-tinyvit}"
cd "${ROOT}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Missing env ${ENV_PREFIX}; run scripts/setup/prepare_edgetam_env.sh first." >&2
  exit 2
fi

export PATH="${ENV_PREFIX}/bin:${PATH}"

case "${TASK}" in
  prepare-data|validate-data|probe-tinyvit|write-config|validate-train-config|stage1-feature-smoke|sav-eval-smoke|vos-style-data|vos-style-eval-smoke|davis-data|davis-eval-smoke|video-mask-train-smoke|edgetam-distill-loss-smoke|edgetam-full-trainer-smoke|edgetam-full-trainer-resume-smoke|edgetam-full-trainer-cache-smoke|edgetam-teacher-cache-smoke|edgetam-full-trainer-forward-cache-smoke|edgetam-image-trainer-smoke|edgetam-image-forward-cache-smoke|edgetam-export-checkpoint-smoke|edgetam-exported-image-smoke|edgetam-exported-vos-smoke|edgetam-progressive-full-trainer-smoke|progressive-video-smoke|edgetam-image-smoke|edgetam-image-benchmark|edgetam-vos-smoke|edgetam-vos-track-later-smoke|edgetam-sav-eval|all-cpu)
    scripts/pace/06_run_edgetam_tinyvit_smoke.sh "${TASK}"
    ;;
  *)
    echo "Unknown TASK=${TASK}" >&2
    echo "Supported: prepare-data validate-data probe-tinyvit write-config validate-train-config stage1-feature-smoke sav-eval-smoke vos-style-data vos-style-eval-smoke davis-data davis-eval-smoke video-mask-train-smoke edgetam-distill-loss-smoke edgetam-full-trainer-smoke edgetam-full-trainer-resume-smoke edgetam-full-trainer-cache-smoke edgetam-teacher-cache-smoke edgetam-full-trainer-forward-cache-smoke edgetam-image-trainer-smoke edgetam-image-forward-cache-smoke edgetam-export-checkpoint-smoke edgetam-exported-image-smoke edgetam-exported-vos-smoke edgetam-progressive-full-trainer-smoke progressive-video-smoke edgetam-image-smoke edgetam-image-benchmark edgetam-vos-smoke edgetam-vos-track-later-smoke edgetam-sav-eval all-cpu" >&2
    exit 2
    ;;
esac

if [[ "${RECORD_GENERIC_SLURM_ROW:-0}" == "1" ]]; then
  python tools/experiments/record_experiment.py \
    --task "Slurm ${TASK}" \
    --data "<=500 smoke subsets" \
    --command "sbatch --export=TASK=${TASK} scripts/pace/slurm_edgetam_smoke.sbatch" \
    --seed "deterministic path sort" \
    --gpu "${SLURM_JOB_GPUS:-1 GPU}" \
    --output "runs/edgetam_smoke" \
    --result "completed"
fi
