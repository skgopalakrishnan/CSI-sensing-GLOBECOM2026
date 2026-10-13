#!/usr/bin/env bash

#$ -l tmem=32G
#$ -l h_rt=168:00:00
#$ -l gpu=true,gpu_type=!(gtx1080ti|titanx|rtx2080|rtx2080ti|rtx3090)
#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -N ntu_transfer_optuna
#$ -t 1-10
#$ -o experiments/hpc_outputs/sge/$JOB_ID.$TASK_ID.out
#$ -e experiments/hpc_outputs/sge/$JOB_ID.$TASK_ID.err

set -euo pipefail

if [[ -n "${CSI_PYTHON_SOURCE:-}" && -f "${CSI_PYTHON_SOURCE}" ]]; then
  # shellcheck disable=SC1091
  source "${CSI_PYTHON_SOURCE}"
fi

if [[ -f adv_csi/bin/activate ]]; then
  # shellcheck disable=SC1091
  source adv_csi/bin/activate
elif [[ -n "${CSI_VENV:-}" && -f "${CSI_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${CSI_VENV}/bin/activate"
fi

PYTHON_BIN=${PYTHON_BIN:-}
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python)
  else
    echo "[ERROR] No python interpreter found in PATH." >&2
    exit 5
  fi
fi

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info[0] >= 3 else 1)'; then
  echo "[ERROR] Python 3 required, but got: ${PYTHON_BIN}" >&2
  "${PYTHON_BIN}" --version || true
  exit 6
fi
echo "[INFO] Python interpreter: ${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"

require_file() {
  local path="$1"
  local message="${2:-Required file not found: ${path}}"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] ${message}" >&2
    return 1
  fi
}

list_runnable_lines() {
  local file="$1"
  grep -Ev '^[[:space:]]*(#|$)' "${file}" | sed '/^$/d'
}

count_runnable_lines() {
  local file="$1"
  list_runnable_lines "${file}" | wc -l | awk '{print $1}'
}

get_runnable_line() {
  local file="$1"
  local line_no="${2:-1}"
  list_runnable_lines "${file}" | sed -n "${line_no}p"
}

PARAMS_FILE=${PARAMS_FILE:-experiments/job_scripts/training/params/ntu_transfer_optuna.txt}
RESULTS_ROOT=${RESULTS_ROOT:-results}
OPTUNA_RESUME_JOB_DIR=${OPTUNA_RESUME_JOB_DIR:-}
if [[ -n "${OPTUNA_RESUME_JOB_DIR}" ]]; then
  OPTUNA_RESULTS_JOB_DIR=${OPTUNA_RESUME_JOB_DIR}
elif [[ -n "${OPTUNA_RESULTS_JOB_DIR:-}" ]]; then
  OPTUNA_RESULTS_JOB_DIR=${OPTUNA_RESULTS_JOB_DIR}
else
  OPTUNA_RESULTS_JOB_DIR=${RESULTS_ROOT}/jobs/ntu_transfer_optuna_${JOB_ID:-manual}
fi
if [[ -n "${OPTUNA_RESUME_JOB_DIR}" && ! -d "${OPTUNA_RESULTS_JOB_DIR}" ]]; then
  echo "[ERROR] OPTUNA_RESUME_JOB_DIR does not exist: ${OPTUNA_RESULTS_JOB_DIR}" >&2
  exit 7
fi

require_file "${PARAMS_FILE}" "Params file not found: ${PARAMS_FILE}" || exit 2

LINE_NO=${SGE_TASK_ID:-1}
TOTAL=$(count_runnable_lines "${PARAMS_FILE}")
LINE=$(get_runnable_line "${PARAMS_FILE}" "${LINE_NO}")
if [[ -z "${LINE}" ]]; then
  echo "[ERROR] No runnable line ${LINE_NO} after filtering. Total=${TOTAL}" >&2
  exit 3
fi

read -r BASE_CONFIG DIRECTION MODEL_NAME N_TRIALS TIMEOUT SEED DET_FLAG <<<"${LINE}"
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[ERROR] Base config not found: ${BASE_CONFIG}" >&2
  exit 4
fi

export SEED=${SEED:-42}
if [[ -n "${DET_FLAG:-}" ]]; then
  export DETERMINISTIC=${DET_FLAG}
fi
export AUTO_DATALOADER=${AUTO_DATALOADER:-1}
export CSI_RESULTS_JOB_DIR="${OPTUNA_RESULTS_JOB_DIR}"
export CSI_JOB_NAME="${CSI_JOB_NAME:-ntu_transfer_optuna-${JOB_ID:-unknown}}"
export CSI_JOB_TS="${CSI_JOB_TS:-$(date +%Y%m%d_%H%M%S)}"

TMP_CONFIG=$(mktemp "${TMPDIR:-/tmp}/ntu_transfer_optuna_cfg_${JOB_ID:-nojid}_${SGE_TASK_ID:-notask}_XXXX.yaml")
cleanup() {
  [[ -f "${TMP_CONFIG}" ]] && rm -f "${TMP_CONFIG}"
}
trap cleanup EXIT

"${PYTHON_BIN}" - "${BASE_CONFIG}" "${TMP_CONFIG}" "${DIRECTION}" "${MODEL_NAME}" "${N_TRIALS}" "${TIMEOUT:-null}" "${SEED}" <<'PY'
import sys
import os
from pathlib import Path

import yaml

base_cfg, out_cfg, direction, model_name, n_trials, timeout, seed = sys.argv[1:8]
cfg = yaml.safe_load(Path(base_cfg).read_text(encoding="utf-8")) or {}

run_cfg = cfg.setdefault("run", {})
run_cfg["targets"] = ["ntu"]

transfer = cfg.setdefault("transfer", {})
transfer["directions"] = [direction]

model = cfg.setdefault("model", {})
model["architectures"] = [model_name]

optuna_cfg = cfg.setdefault("optuna", {})
optuna_cfg["n_trials"] = int(n_trials)
if timeout.lower() in {"", "null", "none"}:
    optuna_cfg["timeout"] = None
else:
    optuna_cfg["timeout"] = int(timeout)
optuna_cfg.setdefault("direction", "maximize")
optuna_cfg.setdefault("objective_metric", "val_accuracy")

common = cfg.setdefault("common", {})
common.setdefault("experiment_name", "ntu-transfer-optuna")

data = cfg.setdefault("data", {})
data["seed"] = int(seed)
auto_loader = str(os.environ.get("AUTO_DATALOADER", "1")).strip().lower() not in {"0", "false", "no", "off"}
if auto_loader:
    for section_name in ("ntu", "csi_bench"):
        section = cfg.get(section_name)
        if isinstance(section, dict):
            for key in ("num_workers", "pin_memory", "persistent_workers", "prefetch_factor"):
                section.pop(key, None)

Path(out_cfg).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

echo "[INFO] Host=$(hostname) | Start=$(date) | TaskID=${SGE_TASK_ID:-NA}/${TOTAL}"
echo "[INFO] PARAMS_FILE=${PARAMS_FILE}"
echo "[INFO] BASE_CONFIG=${BASE_CONFIG}"
echo "[INFO] DIRECTION=${DIRECTION} | MODEL=${MODEL_NAME} | N_TRIALS=${N_TRIALS} | TIMEOUT=${TIMEOUT:-null} | SEED=${SEED}"
echo "[INFO] AUTO_DATALOADER=${AUTO_DATALOADER}"
echo "[INFO] RESOLVED_CONFIG=${TMP_CONFIG}"
echo "[INFO] RESULTS_JOB_DIR=${CSI_RESULTS_JOB_DIR}"
if [[ -n "${OPTUNA_RESUME_JOB_DIR}" ]]; then
  echo "[INFO] Resume mode enabled. Reusing existing Optuna job dir: ${OPTUNA_RESULTS_JOB_DIR}"
fi
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "[WARN] nvidia-smi not available"

CMD=("${PYTHON_BIN}" -u -m experiments.train_models.sl_transfer_optuna "${TMP_CONFIG}")
echo "[INFO] Executing: ${CMD[*]}"
"${CMD[@]}"

echo "[INFO] End=$(date)"
