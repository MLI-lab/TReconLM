#!/bin/bash
#SBATCH -p mcml-hgx-a100-80x4-mig
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o <your.slurm.home.path>/TReconLM/DeepLearningBaselines/RobuSeqNet/slurm_pkg/pretrain/inference/60/logs/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/DeepLearningBaselines/RobuSeqNet/slurm_pkg/pretrain/inference/60/logs/experiment.err
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
    python inference.py \
      --checkpoint <your.storage.path>/model_checkpoints_RobuSeqNet/RobuSeqNet_20250627_135209_gt60/checkpoint_best_val_loss.pt \
      --artifact_name test_dataset_seed34721_gl60_bs800_ds50000 \
      --test_project TRACE_RECONSTRUCTION \
      --project Baselines \
      --wandb_entity <your.wandb.entity> \
      --padding_length 80 \
      --label_length 60 \
  "