#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PX_DIR}"
export PYTHONPATH=${PX_DIR}:${PYTHONPATH:-} # let RoboTwin find the websocket tools from main repo
export PX_python=/mnt/users/envs/PX/bin/python
your_ckpt=${PX_DIR}/checkpoints/ABot-M0-RoboTwin2/checkpoints/steps_125000_pytorch_model.pt
gpu_id=0
port=5694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id "${PX_python}" deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    --use_bf16

# #################################
