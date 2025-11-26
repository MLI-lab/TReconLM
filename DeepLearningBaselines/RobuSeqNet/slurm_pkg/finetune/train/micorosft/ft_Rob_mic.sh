#!/bin/bash
#SBATCH -p mcml-hgx-a100-80x4-mig
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o <your.slurm.home.path>/TReconLM/DeepLearningBaselines/RobuSeqNet/slurm_pkg/finetune/train/micorosft/logs/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/DeepLearningBaselines/RobuSeqNet/slurm_pkg/finetune/train/micorosft/logs/experiment.err
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
    torchrun --nproc_per_node=1 --master_port=29511 finetune.py \
      --train_data <your.storage.path>/data/no_subclusters_microsoft/train.txt \
      --val_data <your.storage.path>/data/no_subclusters_microsoft/val.txt \
      --padding_length 150 \
      --label_length 110 \
      --base_out_dir <your.storage.path>/model_checkpoints_RobuSeqNet/ \
      --wandb_log \
      --wandb_project Baselines \
      --wandb_entity <your.wandb.entity> \
      --max_iter 566046 \
      --batch_size_all 52 \
      --max_lr 1e-5 \
      --pretrain_run_name RobuSeqNet_20250514_203119 \
  "
