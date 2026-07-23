#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX_DIR="${PX_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# Defaults below are for a single-GPU smoke test. Every setting can be overridden.
PX_PYTHON="${PX_PYTHON:-/mnt/users/envs/PX/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$(dirname "${PX_PYTHON}")/accelerate}"
BASE_VLM="${BASE_VLM:-${PX_DIR}/base_vlms/Qwen3-VL-4B-Instruct-Action}"
CONFIG_YAML="${CONFIG_YAML:-${PX_DIR}/examples/LIBERO/train_files/PX_libero.yaml}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/mnt/users/cfr_project/vla/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-${PX_DIR}/checkpoints/ABot-M0-Pretrain/checkpoints/ABot_M0_Pretrain.pt}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${PX_DIR}/outputs/libero_train}"
RUN_ID="${RUN_ID:-smoke_libero_$(date +'%Y%m%d_%H%M%S')}"

TRAIN_GPUS="${TRAIN_GPUS:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-1}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-0}"

FRAMEWORK_NAME="${FRAMEWORK_NAME:-PX_M0}"
DATA_MIX="${DATA_MIX:-libero}"
FREEZE_MODULES="${FREEZE_MODULES:-}"
RELOAD_MODULES="${RELOAD_MODULES:-qwen_vl_interface,action_model}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTHONPATH="${PX_DIR}:${PYTHONPATH:-}"
if [[ -d /usr/local/cuda-12 ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

fail() {
  echo "Error: $*" >&2
  exit 1
}

[[ -x "${PX_PYTHON}" ]] || fail "PX Python is not executable: ${PX_PYTHON}"
[[ -x "${ACCELERATE_BIN}" ]] || fail "accelerate is not executable: ${ACCELERATE_BIN}"
[[ -f "${CONFIG_YAML}" ]] || fail "training config not found: ${CONFIG_YAML}"
[[ -f "${BASE_VLM}/config.json" ]] || fail "Action base VLM not found: ${BASE_VLM}
Download StarVLA/Qwen3-VL-4B-Instruct-Action or set BASE_VLM=/path/to/model."
[[ -f "${PRETRAIN_CKPT}" ]] || fail "pretrained checkpoint not found: ${PRETRAIN_CKPT}"
[[ -d "${LIBERO_DATA_ROOT}" ]] || fail "LIBERO data root not found: ${LIBERO_DATA_ROOT}"

dataset_names=(
  libero_10_no_noops_1.0.0_lerobot
  libero_goal_no_noops_1.0.0_lerobot
  libero_object_no_noops_1.0.0_lerobot
  libero_spatial_no_noops_1.0.0_lerobot
)
for dataset_name in "${dataset_names[@]}"; do
  [[ -f "${LIBERO_DATA_ROOT}/${dataset_name}/meta/modality.json" ]] ||
    fail "dataset is incomplete: ${LIBERO_DATA_ROOT}/${dataset_name}"
done

for value_name in NUM_PROCESSES PER_DEVICE_BATCH_SIZE MAX_TRAIN_STEPS; do
  value="${!value_name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "${value_name} must be a positive integer"
done

IFS=',' read -r -a gpu_ids <<< "${TRAIN_GPUS}"
(( ${#gpu_ids[@]} == NUM_PROCESSES )) ||
  fail "TRAIN_GPUS and NUM_PROCESSES disagree: ${TRAIN_GPUS} vs ${NUM_PROCESSES}"

output_dir="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${output_dir}"
cp "${BASH_SOURCE[0]}" "${output_dir}/"

cd "${PX_DIR}"

cmd=(
  "${ACCELERATE_BIN}" launch
  --config_file PX/config/deepseeds/deepspeed_zero2.yaml
  --num_processes "${NUM_PROCESSES}"
  PX/training/train.py
  --config_yaml "${CONFIG_YAML}"
  --framework.name "${FRAMEWORK_NAME}"
  --framework.qwenvl.base_vlm "${BASE_VLM}"
  --datasets.vla_data.data_root_dir "${LIBERO_DATA_ROOT}"
  --datasets.vla_data.data_mix "${DATA_MIX}"
  --datasets.vla_data.num_workers "${NUM_WORKERS}"
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --datasets.vla_data.include_state false
  --datasets.vla_data.video_backend torchvision_av
  --trainer.pretrained_checkpoint "${PRETRAIN_CKPT}"
  --trainer.reload_modules "${RELOAD_MODULES}"
  --trainer.freeze_modules "${FREEZE_MODULES}"
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.eval_interval "${EVAL_INTERVAL}"
  --trainer.logging_frequency "${LOGGING_FREQUENCY}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --run_id "${RUN_ID}"
)

echo "Starting LIBERO training smoke test"
echo "GPUs: ${TRAIN_GPUS}; processes: ${NUM_PROCESSES}; steps: ${MAX_TRAIN_STEPS}"
echo "Output: ${output_dir}"
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" exec "${cmd[@]}"
