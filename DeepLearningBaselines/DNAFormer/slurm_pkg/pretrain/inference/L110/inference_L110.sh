#!/bin/bash
#SBATCH -p <YOUR_PARTITION> # set to your partition
#SBATCH --qos=<YOUR_QOS> # set to your QOS if needed
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o <LOG_DIR>/experiment.out # set output log directory
#SBATCH -e <LOG_DIR>/experiment.err # set error log directory
#SBATCH --time=00:45:00

srun \
  --container-image=<PATH_TO_CONTAINER_IMAGE> \
  --container-mounts="<PATH_TO_PROJECT>:<CONTAINER_PROJECT_PATH>,<ADDITIONAL_MOUNTS>" \
  bash -c '

cd <CONTAINER_PROJECT_PATH>/DeepLearningBaselines/DNAFormer
source /opt/conda/etc/profile.d/conda.sh
conda activate treconlm

# W&B env (set inside container)
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"

# Install required packages
python -m pip install transformers torchmetrics einops -q
python -m pip install gpustat -q

# Start GPU monitoring in background
GPU_LOG="<CONTAINER_PROJECT_PATH>/DeepLearningBaselines/DNAFormer/slurm_pkg/pretrain/inference/L110/logs/gpu_stats.log"
echo "Starting GPU monitoring, logging to: $GPU_LOG"

gpustat -i 1 --no-color >> "$GPU_LOG" 2>&1 &
MONITOR_PID=$!

# Run inference script
python -u slurm_pkg/pretrain/inference/L110/inference_110.py

# Kill the monitoring process when done
kill $MONITOR_PID 2>/dev/null
'