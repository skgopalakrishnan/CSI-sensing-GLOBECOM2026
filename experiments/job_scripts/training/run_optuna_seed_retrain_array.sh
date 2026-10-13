#!/usr/bin/env bash

#$ -l tmem=32G
#$ -l h_rt=120:00:00
#$ -l gpu=true,gpu_type=!(gtx1080ti|titanx|rtx2080|rtx2080ti|rtx3090)
#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -N optuna_seed_retrain
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

PARAMS_FILE=${PARAMS_FILE:-experiments/job_scripts/training/params/optuna_seed_retrain_targets.txt}
RESULTS_ROOT=${RESULTS_ROOT:-results}
SEED_RETRAIN_RESULTS_JOB_DIR=${SEED_RETRAIN_RESULTS_JOB_DIR:-${RESULTS_ROOT}/jobs/optuna_seed_retrain_${JOB_ID:-manual}}

require_file "${PARAMS_FILE}" "Params file not found: ${PARAMS_FILE}" || exit 2

LINE_NO=${SGE_TASK_ID:-1}
TOTAL=$(count_runnable_lines "${PARAMS_FILE}")
LINE=$(get_runnable_line "${PARAMS_FILE}" "${LINE_NO}")
if [[ -z "${LINE}" ]]; then
  echo "[ERROR] No runnable line ${LINE_NO} after filtering. Total=${TOTAL}" >&2
  exit 3
fi

read -r BASE_CONFIG SUMMARY_JSON SEEDS_CSV DET_FLAG AUTO_LOADER_FLAG INCLUDE_MODEL INCLUDE_DATASET INCLUDE_TASK <<<"${LINE}"
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[ERROR] Base config not found: ${BASE_CONFIG}" >&2
  exit 4
fi
if [[ ! -f "${SUMMARY_JSON}" ]]; then
  echo "[ERROR] Summary JSON not found: ${SUMMARY_JSON}" >&2
  exit 4
fi

if [[ -n "${DET_FLAG:-}" ]]; then
  export DETERMINISTIC="${DET_FLAG}"
fi
export AUTO_DATALOADER="${AUTO_LOADER_FLAG:-1}"
export CSI_RESULTS_JOB_DIR="${SEED_RETRAIN_RESULTS_JOB_DIR}"
export CSI_JOB_NAME="${CSI_JOB_NAME:-optuna_seed_retrain-${JOB_ID:-unknown}}"
export CSI_JOB_TS="${CSI_JOB_TS:-$(date +%Y%m%d_%H%M%S)}"
SKIP_COMPLETED_OPTUNA_SEED_RETRAIN=${SKIP_COMPLETED_OPTUNA_SEED_RETRAIN:-1}
FORCE_OPTUNA_SEED_RETRAIN=${FORCE_OPTUNA_SEED_RETRAIN:-0}

TMP_CONFIG=$(mktemp "${TMPDIR:-/tmp}/optuna_seed_retrain_cfg_${JOB_ID:-nojid}_${SGE_TASK_ID:-notask}_XXXX.yaml")
cleanup() {
  [[ -f "${TMP_CONFIG}" ]] && rm -f "${TMP_CONFIG}"
}
trap cleanup EXIT

"${PYTHON_BIN}" - "${BASE_CONFIG}" "${TMP_CONFIG}" "${SUMMARY_JSON}" "${SEEDS_CSV}" "${DET_FLAG:-false}" "${INCLUDE_MODEL:-}" "${INCLUDE_DATASET:-}" "${INCLUDE_TASK:-}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

base_cfg, out_cfg, summary_json, seeds_csv, det_flag, include_model, include_dataset, include_task = sys.argv[1:9]
cfg = yaml.safe_load(Path(base_cfg).read_text(encoding="utf-8")) or {}

selection = cfg.setdefault("selection", {})
selection["summary_json"] = str(summary_json)

run_cfg = cfg.setdefault("run", {})
seeds = [int(token.strip()) for token in str(seeds_csv).split(",") if token.strip()]
if not seeds:
    raise SystemExit(f"[ERROR] Invalid seeds list: {seeds_csv}")
run_cfg["seeds"] = seeds
run_cfg["deterministic"] = str(det_flag).strip().lower() in {"1", "true", "yes", "y", "on"}
run_cfg["require_test_split"] = True
run_cfg.setdefault("continue_on_error", False)

def _norm_filter(token: str) -> str:
    value = str(token).strip()
    if value.lower() in {"", "none", "null", "-", "_"}:
        return ""
    return value

model_filter = _norm_filter(include_model)
dataset_filter = _norm_filter(include_dataset)
task_filter = _norm_filter(include_task)
if model_filter:
    run_cfg["include_models"] = [model_filter]
else:
    run_cfg.pop("include_models", None)
if dataset_filter:
    run_cfg["include_datasets"] = [dataset_filter]
else:
    run_cfg.pop("include_datasets", None)
if task_filter:
    run_cfg["include_tasks"] = [task_filter]
else:
    run_cfg.pop("include_tasks", None)

data = cfg.setdefault("data", {})
auto_loader = str(os.environ.get("AUTO_DATALOADER", "1")).strip().lower() not in {"0", "false", "no", "off"}
if auto_loader:
    for key in ("num_workers", "pin_memory", "persistent_workers", "prefetch_factor"):
        data.pop(key, None)

Path(out_cfg).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

echo "[INFO] Host=$(hostname) | Start=$(date) | TaskID=${SGE_TASK_ID:-NA}/${TOTAL}"
echo "[INFO] PARAMS_FILE=${PARAMS_FILE}"
echo "[INFO] BASE_CONFIG=${BASE_CONFIG}"
echo "[INFO] SUMMARY_JSON=${SUMMARY_JSON}"
echo "[INFO] SEEDS_CSV=${SEEDS_CSV} | DETERMINISTIC=${DETERMINISTIC:-0} | AUTO_DATALOADER=${AUTO_DATALOADER}"
echo "[INFO] INCLUDE_MODEL=${INCLUDE_MODEL:-none} | INCLUDE_DATASET=${INCLUDE_DATASET:-none} | INCLUDE_TASK=${INCLUDE_TASK:-none}"
echo "[INFO] RESOLVED_CONFIG=${TMP_CONFIG}"
echo "[INFO] RESULTS_JOB_DIR=${CSI_RESULTS_JOB_DIR}"

if [[ "${FORCE_OPTUNA_SEED_RETRAIN}" != "1" && "${SKIP_COMPLETED_OPTUNA_SEED_RETRAIN}" != "0" ]]; then
  PRECHECK_STATUS=0
  PRECHECK_OUTPUT=$("${PYTHON_BIN}" - "${TMP_CONFIG}" <<'PY'
import json
import os
import sys
from pathlib import Path

import yaml

DEFAULT_CLEAN_EXPORT_SEED = 42
DEFAULT_CLEAN_WEIGHTS_DIR = "model_params_optuna_tiebreak_seed42"


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [token.strip() for token in value.split(",") if token.strip()]
    if isinstance(value, (list, tuple)):
        return [str(token).strip() for token in value if str(token).strip()]
    return [str(value).strip()]


def _coerce_seed_list(value):
    if value is None:
        return [42, 11, 23, 37, 53]
    if isinstance(value, str):
        return [int(token.strip()) for token in value.split(",") if token.strip()]
    if isinstance(value, (list, tuple)):
        return [int(token) for token in value]
    return [int(value)]


def _sanitize_path_component(part):
    raw = str(part).strip()
    sanitised = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in raw)
    return sanitised or "unnamed"


def _resolve_project_path(path_str):
    return Path(path_str).expanduser().resolve()


def _load_selection(summary_path, domain):
    payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    studies = payload.get("studies") or []
    out = []
    for study in studies:
        if not isinstance(study, dict):
            continue
        selected = study.get("selected_trial")
        if not isinstance(selected, dict):
            continue
        params = selected.get("params")
        if not isinstance(params, dict):
            continue
        model = study.get("model")
        dataset = study.get("dataset")
        task_name = study.get("task_name")
        if not model:
            continue
        if domain == "ntu" and not dataset:
            continue
        if domain == "csi_bench" and not task_name:
            continue
        out.append(
            {
                "dataset": str(dataset) if dataset is not None else None,
                "task_name": str(task_name) if task_name is not None else None,
                "model": str(model),
            }
        )
    out.sort(key=lambda row: (str(row.get("dataset") or row.get("task_name") or ""), str(row.get("model") or "")))
    return out


def _filter_targets(targets, run_cfg):
    include_models = {token.lower() for token in _as_list(run_cfg.get("include_models"))}
    include_datasets = set(_as_list(run_cfg.get("include_datasets")))
    include_tasks = set(_as_list(run_cfg.get("include_tasks")))
    out = []
    for row in targets:
        model_name = str(row.get("model", ""))
        dataset = row.get("dataset")
        task_name = row.get("task_name")
        if include_models and model_name.lower() not in include_models:
            continue
        if include_datasets and dataset not in include_datasets:
            continue
        if include_tasks and task_name not in include_tasks:
            continue
        out.append(row)
    return out


cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
common_cfg = cfg.get("common", {}) or {}
run_cfg = cfg.get("run", {}) or {}
selection_cfg = cfg.get("selection", {}) or {}
data_cfg = cfg.get("data", {}) or {}

summary_json = selection_cfg.get("summary_json")
domain = str(run_cfg.get("domain", "")).strip().lower()
experiment_name = str(common_cfg.get("experiment_name", "optuna-seed-retrain"))
job_dir = os.environ.get("CSI_RESULTS_JOB_DIR")
if not summary_json or not domain or not job_dir:
    raise SystemExit(0)

seeds = [int(seed) for seed in _coerce_seed_list(run_cfg.get("seeds"))]
targets = _filter_targets(_load_selection(str(summary_json), domain), run_cfg)
if not targets:
    raise SystemExit(0)

export_cfg_raw = cfg.get("export", {}) or {}
export_cfg = {
    "enabled": str(export_cfg_raw.get("enabled", True)).strip().lower() in {"1", "true", "yes", "y", "on"},
    "clean_weights_dir_name": str(export_cfg_raw.get("clean_weights_dir_name", DEFAULT_CLEAN_WEIGHTS_DIR)).strip(),
    "canonical_seed": int(export_cfg_raw.get("canonical_seed", DEFAULT_CLEAN_EXPORT_SEED)),
    "datasets_root": export_cfg_raw.get("datasets_root"),
    "dataset_root": export_cfg_raw.get("dataset_root"),
}
job_base = Path(job_dir).resolve() / _sanitize_path_component(experiment_name)

all_complete = True
messages = []
for target in targets:
    model_name = str(target.get("model", ""))
    if domain == "ntu":
        label = str(target.get("dataset", ""))
    else:
        label = normalize_csi_bench_task_name(str(target.get("task_name", "")))
    model_scope_dir = (
        job_base
        / _sanitize_path_component(domain)
        / _sanitize_path_component(label)
        / _sanitize_path_component(model_name)
    )
    summary_path = model_scope_dir / "seed_runs_summary.json"
    seed_dir = model_scope_dir / "seed_weights"
    missing = []
    if not summary_path.is_file():
        missing.append(str(summary_path))
    for seed in seeds:
        seed_path = seed_dir / f"seed_{int(seed):04d}.pt"
        if not seed_path.is_file():
            missing.append(str(seed_path))

    if export_cfg.get("enabled") and int(export_cfg.get("canonical_seed")) in seeds:
        weights_dir_name = str(export_cfg.get("clean_weights_dir_name"))
        if domain == "ntu":
            datasets_root = _resolve_project_path(
                str(export_cfg.get("datasets_root") or data_cfg.get("root") or common_cfg.get("root") or "datasets")
            )
            export_alias = datasets_root / label / weights_dir_name / f"{label}_{model_name}.pt"
        else:
            dataset_root = _resolve_project_path(
                str(export_cfg.get("dataset_root") or data_cfg.get("dataset_root") or common_cfg.get("dataset_root") or "datasets/CSI-Bench")
            )
            csi_root = dataset_root if dataset_root.name == "CSI-Bench" else (dataset_root / "CSI-Bench")
            export_alias = csi_root / weights_dir_name / label / f"CSI-Bench_{model_name}.pt"
        if not export_alias.is_file():
            missing.append(str(export_alias))

    if missing:
        all_complete = False
    else:
        messages.append(f"[INFO] Seed retrain already complete for {domain}:{label}/{model_name} seeds={seeds}")

if all_complete:
    for line in messages:
        print(line)
    raise SystemExit(10)
PY
  ) || PRECHECK_STATUS=$?

  if [[ "${PRECHECK_STATUS}" -eq 10 ]]; then
    [[ -n "${PRECHECK_OUTPUT}" ]] && printf '%s\n' "${PRECHECK_OUTPUT}"
    echo "[INFO] Skipping Optuna seed retrain because all requested seed outputs already exist."
    exit 0
  fi
  if [[ "${PRECHECK_STATUS}" -ne 0 ]]; then
    [[ -n "${PRECHECK_OUTPUT}" ]] && printf '%s\n' "${PRECHECK_OUTPUT}" >&2
    echo "[WARN] Optuna seed retrain completion precheck failed; continuing with retrain." >&2
  fi
fi

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "[WARN] nvidia-smi not available"

CMD=("${PYTHON_BIN}" -u -m experiments.train_models.optuna_seed_retrain "${TMP_CONFIG}")
echo "[INFO] Executing: ${CMD[*]}"
"${CMD[@]}"

echo "[INFO] End=$(date)"
