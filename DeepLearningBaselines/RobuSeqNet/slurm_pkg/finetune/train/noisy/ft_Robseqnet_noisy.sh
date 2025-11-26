#!/bin/bash
#SBATCH -p mcml-hgx-a100-80x4-mig
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o <your.slurm.home.path>/TReconLM/DeepLearningBaselines/RobuSeqNet/slurm_pkg/finetune/train/noisy/logs/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/DeepLearningBaselines/RobuSeqNet/slurm_pkg/finetune/train/noisy/logs/experiment.err
#SBATCH --time=48:00:00

srun \
  --container-image=<your.storage.path>/enroot/Treconlm.sqsh \
  --container-mounts="$PWD/TReconLM:/TReconLM,<your.storage.path>:<your.storage.path>" \
  --container-env WANDB_API_KEY=<your.wandb.api.key> \
  bash -c "
    cd /TReconLM/DeepLearningBaselines/RobuSeqNet/examples
    source /opt/conda/etc/profile.d/conda.sh
    conda activate treconlm
    pip install transformers torchmetrics einops
    torchrun --nproc_per_node=1 --master_port=29501 finetune.py \
      --train_data <your.storage.path>/data/noisy/train.txt \
      --val_data <your.storage.path>/data/noisy/val.txt \
      --padding_length 80 \
      --label_length 60 \
      --base_out_dir <your.storage.path>/model_checkpoints_RobuSeqNet/ \
      --wandb_log \
      --wandb_project Baselines \
      --wandb_entity <your.wandb.entity> \
      --max_iter 685307 \
      --batch_size_all 8 \
      --max_lr 1e-5 \
      --pretrain_run_name RobuSeqNet_20250515_020145
  "
