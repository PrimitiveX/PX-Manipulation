#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX_DIR="${PX_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# All settings can be overridden without editing this file, for example:
# CKPT=/path/to/model.pt GPU_ID=0 bash examples/LIBERO/eval_files/run_policy_server.sh
PX_PYTHON="${PX_PYTHON:-/mnt/users/envs/PX/bin/python}"
CKPT="${CKPT:-${PX_DIR}/checkpoints/ABot-M0-LIBERO/checkpoints/steps_40000_pytorch_model.pt}"
PORT="${PORT:-2375}"
GPU_ID="${GPU_ID:-0}"
USE_BF16="${USE_BF16:-1}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-1800}"
DEBUG="${DEBUG:-1}"

if [[ ! -x "${PX_PYTHON}" ]]; then
  echo "Error: PX Python is not executable: ${PX_PYTHON}" >&2
  echo "Set it with PX_PYTHON=/path/to/python." >&2
  exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Error: checkpoint not found: ${CKPT}" >&2
  echo "Download the ABot-M0-LIBERO checkpoint or set CKPT=/path/to/model.pt." >&2
  exit 1
fi

cd "${PX_DIR}"
export PYTHONPATH="${PX_DIR}:${PYTHONPATH:-}"
export DEBUG

cmd=(
  "${PX_PYTHON}"
  deployment/model_server/server_policy.py
  --ckpt_path "${CKPT}"
  --port "${PORT}"
  --idle_timeout "${IDLE_TIMEOUT}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  cmd+=(--use_bf16)
fi

echo "Starting policy server on port ${PORT} (GPU ${GPU_ID})"
echo "Checkpoint: ${CKPT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" exec "${cmd[@]}"
