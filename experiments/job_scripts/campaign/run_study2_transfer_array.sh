#!/usr/bin/env bash

#$ -l tmem=24G
#$ -l h_rt=72:00:00
#$ -l gpu=true,gpu_type=!(gtx1080ti|titanx|rtx2080|rtx2080ti|rtx4070ti|p100|v100)
#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -N study2_transfer
#$ -t 1-800
#$ -o experiments/hpc_outputs/sge/$JOB_ID.$TASK_ID.out
#$ -e experiments/hpc_outputs/sge/$JOB_ID.$TASK_ID.err

set -euo pipefail

# shellcheck disable=SC1091
source experiments/job_scripts/common_job_env.sh
activate_cluster_python
resolve_python_bin
setup_campaign_defaults

PARAMS_FILE="${PARAMS_FILE:-experiments/job_scripts/campaign/params/study2_transfer.txt}"
require_file "${PARAMS_FILE}" "Params file not found: ${PARAMS_FILE}" || exit 2

LINE_NO=${SGE_TASK_ID:-1}
TOTAL=$(count_runnable_lines "${PARAMS_FILE}")
LINE=$(get_runnable_line "${PARAMS_FILE}" "${LINE_NO}")
if [[ -z "${LINE}" ]]; then
  echo "[ERROR] No runnable line ${LINE_NO} after filtering. Total=${TOTAL}" >&2
  exit 3
fi

read -r RUNNER BASE_CONFIG SEED TARGET SOURCE_MODEL TARGET_MODEL <<<"${LINE}"
require_file "${BASE_CONFIG}" "Config not found: ${BASE_CONFIG}" || exit 4
export SEED="${SEED:-42}"

TMP_CONFIG=$(mktemp "${TMPDIR:-/tmp}/study2_transfer_cfg_${JOB_ID:-nojid}_${SGE_TASK_ID:-notask}_XXXX.yaml")
cleanup() {
  [[ -f "${TMP_CONFIG}" ]] && rm -f "${TMP_CONFIG}"
}
trap cleanup EXIT

"${PYTHON_BIN}" -u experiments/job_scripts/prepare_campaign_config.py \
  "${BASE_CONFIG}" "${TMP_CONFIG}" \
  --seed "${SEED}" \
  --target "${TARGET:-}" \
  --source-model "${SOURCE_MODEL:-}" \
  --target-model "${TARGET_MODEL:-}"

RUN_STUB=$(basename "${TMP_CONFIG}")
RUN_STUB="${RUN_STUB%.yaml}"
set_results_run_parts \
  "$(results_part task "${LINE_NO}")" \
  "$(results_part seed "${SEED}")" \
  "$(results_part dataset "${TARGET:-all}")" \
  "$(results_part source_model "${SOURCE_MODEL:-all}")" \
  "$(results_part target_model "${TARGET_MODEL:-all}")" \
  "$(results_part run "${RUN_STUB}")"

export CSI_RESULTS_JOB_DIR="results/jobs/study2_transfer_${JOB_ID:-manual}"
export CSI_JOB_NAME="${CSI_JOB_NAME:-study2_transfer-${JOB_ID:-unknown}}"
export CSI_JOB_TS="${CSI_JOB_TS:-$(date +%Y%m%d_%H%M%S)}"

echo "[INFO] Host=$(hostname) | Start=$(date) | TaskID=${SGE_TASK_ID:-NA}/${TOTAL}"
echo "[INFO] PARAMS_FILE=${PARAMS_FILE}"
echo "[INFO] RUNNER=${RUNNER} | BASE_CONFIG=${BASE_CONFIG} | TARGET=${TARGET:-all} | SOURCE_MODEL=${SOURCE_MODEL:-all} | TARGET_MODEL=${TARGET_MODEL:-all} | SEED=${SEED}"
echo "[INFO] RESOLVED_CONFIG=${TMP_CONFIG}"
echo "[INFO] RESULTS_JOB_DIR=${CSI_RESULTS_JOB_DIR}"
echo "[INFO] RESULTS_RUN_PARTS=${CSI_RESULTS_RUN_PARTS:-}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "[WARN] nvidia-smi not available"

CMD=("${PYTHON_BIN}" -u -m "experiments.attacks_raw_csi.${RUNNER}" "${TMP_CONFIG}")
echo "[INFO] Executing: ${CMD[*]}"
"${CMD[@]}"

echo "[INFO] End=$(date)"
