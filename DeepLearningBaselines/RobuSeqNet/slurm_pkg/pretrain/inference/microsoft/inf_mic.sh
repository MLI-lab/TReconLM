#!/bin/bash
#SBATCH -p mcml-hgx-a100-80x4-mig
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o <your.slurm.home.path>/TReconLM/DeepLearningBaselines/RobuSeqNet/slurm_pkg/pretrain/inference/microsoft/logs/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/DeepLearningBaselines/RobuSeqNet/slurm_pkg/pretrain/inference/microsoft/logs/experiment.err
#SBATCH --time=48:00:00

srun \
  --container-image=<your.storage.path>/enroot/Treconlm.sqsh \
  --container-mounts="$PWD/TReconLM:/TReconLM,/dss/dssmcmlfs01/pn57vo:/dss/dssmcmlfs01/pn57vo" \
  --container-env WANDB_API_KEY=<your.wandb.api.key> \
  bash -c "
    cd /TReconLM/DeepLearningBaselines/RobuSeqNet/examples
    source /opt/conda/etc/profile.d/conda.sh
    conda activate treconlm
    pip install transformers torchmetrics einops
    python inference.py \
      --checkpoint <your.storage.path>/model_checkpoints_RobuSeqNet/RobuSeqNet_finet_20250605_190529/checkpoint_best_val_loss.pt \
      --artifact_name Microsoft-test-20250502_132818 \
      --test_project TRACE_RECONSTRUCTION \
      --project Baselines \
      --wandb_entity <your.wandb.entity> \
      --padding_length 150 \
      --label_length 110
  "
