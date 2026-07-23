#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX_DIR="${PX_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# All settings can be overridden without editing this file, for example:
# TASK_SUITES="libero_goal" EVAL_GPUS="0" NUM_TRIALS_PER_TASK=5 \
#   bash examples/LIBERO/eval_files/eval_libero.sh
LIBERO_HOME="${LIBERO_HOME:-/mnt/users/cfr_project/vla/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/mnt/users/envs/libero/bin/python}"
CKPT="${CKPT:-${PX_DIR}/checkpoints/ABot-M0-LIBERO/checkpoints/steps_40000_pytorch_model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PX_DIR}/outputs/libero_eval}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-2375}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
TASK_SUITES="${TASK_SUITES:-libero_goal libero_spatial libero_object libero_10}"
EVAL_GPUS="${EVAL_GPUS:-0 0 0 0}"
MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-egl}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-egl}"

if [[ ! -d "${LIBERO_HOME}/libero" ]]; then
  echo "Error: LIBERO_HOME is invalid: ${LIBERO_HOME}" >&2
  echo "Expected directory: ${LIBERO_HOME}/libero" >&2
  exit 1
fi

if [[ ! -x "${LIBERO_PYTHON}" ]]; then
  echo "Error: LIBERO Python is not executable: ${LIBERO_PYTHON}" >&2
  echo "Set it with LIBERO_PYTHON=/path/to/python." >&2
  exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Error: checkpoint not found: ${CKPT}" >&2
  echo "Download the ABot-M0-LIBERO checkpoint or set CKPT=/path/to/model.pt." >&2
  exit 1
fi

if [[ ! "${NUM_TRIALS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: NUM_TRIALS_PER_TASK must be a positive integer." >&2
  exit 1
fi

read -r -a task_suites <<< "${TASK_SUITES}"
read -r -a eval_gpus <<< "${EVAL_GPUS}"

if (( ${#task_suites[@]} == 0 )); then
  echo "Error: TASK_SUITES cannot be empty." >&2
  exit 1
fi

if (( ${#task_suites[@]} != ${#eval_gpus[@]} )); then
  echo "Error: TASK_SUITES and EVAL_GPUS must contain the same number of entries." >&2
  echo "TASK_SUITES=${TASK_SUITES}" >&2
  echo "EVAL_GPUS=${EVAL_GPUS}" >&2
  exit 1
fi

cd "${PX_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${LIBERO_HOME}:${PX_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"

ckpt_file="$(basename "${CKPT}")"
ckpt_parent_dir="$(dirname "${CKPT}")"
ckpt_parent="$(basename "${ckpt_parent_dir}")"
ckpt_grandparent="$(basename "$(dirname "${ckpt_parent_dir}")")"
folder_name="${ckpt_grandparent}_${ckpt_parent}_${ckpt_file}"

timestamp="$(date +'%Y%m%d_%H%M%S')"
log_dir="${OUTPUT_DIR}/logs/${timestamp}"
mkdir -p "${log_dir}"

pids=()
cleanup() {
  if (( ${#pids[@]} > 0 )); then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for i in "${!task_suites[@]}"; do
  task_suite="${task_suites[$i]}"
  gpu_id="${eval_gpus[$i]}"
  video_out_path="${OUTPUT_DIR}/${task_suite}/${folder_name}"
  log_file="${log_dir}/${task_suite}.log"

  echo "Starting ${task_suite} on GPU ${gpu_id}; log: ${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${LIBERO_PYTHON}" \
    ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${CKPT}" \
    --args.unnorm-key "${UNNORM_KEY}" \
    --args.host "${HOST}" \
    --args.port "${PORT}" \
    --args.task-suite-name "${task_suite}" \
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
    --args.video-out-path "${video_out_path}" \
    2>&1 | tee "${log_file}" &
  pids+=("$!")
done

result=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    result=1
  fi
done

if (( result != 0 )); then
  echo "One or more LIBERO evaluations failed. Logs: ${log_dir}" >&2
  exit "${result}"
fi

echo "All LIBERO evaluations completed. Logs: ${log_dir}"
