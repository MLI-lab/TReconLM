#!/bin/bash
#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o <your.slurm.home.path>/TReconLM/DeepLearningBaselines/RobuSeqNet/slurm_pkg/pretrain/inference/110/logs/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/DeepLearningBaselines/RobuSeqNet/slurm_pkg/pretrain/inference/110/logs/experiment.err
#SBATCH --time=00:60:00

srun \
  --container-image=<your.storage.path>/enroot/Treconlm.sqsh \
  --container-mounts="$PWD/TReconLM:/TReconLM,<your.storage.path>:<your.storage.path>" \
  bash -c '
cd /TReconLM/DeepLearningBaselines/RobuSeqNet/examples
source /opt/conda/etc/profile.d/conda.sh
conda activate treconlm

# W&B env (set inside container)
export WANDB_API_KEY="<your.wandb.api.key>8"

# Install required packages
python -m pip install transformers torchmetrics einops -q
python -m pip install gpustat -q

# Start GPU monitoring in background
GPU_LOG="<your.slurm.home.path>/TReconLM/DeepLearningBaselines/RobuSeqNet/slurm_pkg/pretrain/inference/110/logs/gpu_stats.log"
echo "Starting GPU monitoring, logging to: $GPU_LOG"

gpustat -i 1 --no-color >> "$GPU_LOG" 2>&1 &
MONITOR_PID=$!

# Run inference script
python inference.py \
  --checkpoint <your.storage.path>/model_checkpoints_RobuSeqNet/RobuSeqNet_20250626_211721_gt110/checkpoint_best_val_loss.pt \
  --artifact_name test_dataset_seed34721_gl110_bs1500_ds50000 \
  --test_project TRACE_RECONSTRUCTION \
  --project TimingCost \
  --wandb_entity <your.wandb.entity> \
  --padding_length 150 \
  --label_length 110 \
  --batch_size 400 \
  --timing \

# Kill the monitoring process when done
kill $MONITOR_PID 2>/dev/null
'

