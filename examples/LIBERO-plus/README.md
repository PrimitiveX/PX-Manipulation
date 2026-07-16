# 🚀 LIBERO-plus zero shot Evaluation

This document provides instructions for reproducing our **experimental results** with LIBERO-plus.  
The evaluation process consists of two main parts:  

1. Setting up the `LIBERO-plus` environment and dependencies.  
2. Running the evaluation by launching services in both `PX` and `LIBERO-plus` environments.  

We have verified that this workflow runs successfully on both **NVIDIA A100** and **RTX 4090** GPUs.  

---


## ⬇️ 0. Download Checkpoints

Please download Checkpoint from [🤗 PX-M0-LIBERO](https://huggingface.co/acvlab/PX-M0-LIBERO). You should replace the `base_vlm` in the `config.yaml` file with your own path.


---


## 📦 1. Environment Setup

To set up the environment, please first follow the official [LIBERO-plus repository](https://github.com/sylvestf/LIBERO-plus) to install the base `LIBERO-plus` environment.  



Afterwards, inside the `LIBERO-plus` environment, install the following dependencies:  

```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

---

## 🚀 2. Evaluation Workflow

The evaluation should be run **from the repository root** using **two separate terminals**, one for each environment:  

- **PX environment**: runs the inference server.  
- **LIBERO-plus environment**: runs the simulation.  

### Step 1. Start the server (PX environment)

In the first terminal, activate the `PX` conda environment and run:  

```bash
bash examples/LIBERO-plus/eval_files/run_policy_server.sh
```

⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `examples/LIBERO-plus/eval_files/run_policy_server.sh`  


---

### Step 2. Start the simulation (LIBERO-plus environment)

In the second terminal, activate the `LIBERO-plus` conda environment and run:  

```bash
bash examples/LIBERO-plus/eval_files/eval_libero.sh
```
⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `eval_libero.sh` to load action unnormalization stats. 

Also ensure the environment variables at the top of `eval_libero.sh` are correctly set.


---

⚠️ **Note:** Since LIBERO-plus has 10,030 tasks, completing all the evaluations will take an extremely long time. It is recommended to run multiple model instances in parallel for the evaluations. We provide code and scripts for parallel testing on cluster `./parallel_eval/run_nebula_libero_plus`. Please modify them to fit your own cluster.

🚀 PX-M0 performs zero-shot evaluation on LIBERO-plus, therefore using the model only trained on LIBERO.


