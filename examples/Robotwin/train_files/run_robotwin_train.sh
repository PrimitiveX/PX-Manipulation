
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=PX_M0
freeze_module_list="qwen_vl_interface"
base_vlm=/mnt/users/cfr_project/vla/px/PX-Manipulation/base_vlms/Qwen3-VL-4B-Instruct-Action
config_yaml=./examples/Robotwin/train_files/PX_robotwin.yaml
robotwin_data_root=/mnt/users/cfr_project/vla/starVLA/playground/Datasets/robotwin
data_mix=robotwin
pretrain_ckpt=/mnt/users/cfr_project/vla/px/PX-Manipulation/checkpoints/ABot-M0-Pretrain/checkpoints/ABot_M0_Pretrain.pt
run_root_dir=/mnt/users/cfr_project/vla/px/PX-Manipulation/outputs/robotwin_train
run_id=PX_M0_robotwin
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=disabled
export WANDB_MODE=offline
export WANDB_DISABLED=true
export CUDA_HOME=/usr/local/cuda-12
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com 


for dataset_split in Clean Randomized; do
  if [ ! -d "${robotwin_data_root}/${dataset_split}" ]; then
    echo "Error: RoboTwin dataset split not found: ${robotwin_data_root}/${dataset_split}" >&2
    exit 1
  fi
done

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


/mnt/users/envs/PX/bin/accelerate launch \
  --config_file PX/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  PX/training/train.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.use_vggt false \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${robotwin_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.pretrained_checkpoint ${pretrain_ckpt} \
  --trainer.reload_modules qwen_vl_interface \
  --datasets.vla_data.num_workers 4 \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vla_data.include_state false \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 150000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 5000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
