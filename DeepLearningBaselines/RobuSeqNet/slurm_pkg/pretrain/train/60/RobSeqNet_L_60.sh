#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8
#SBATCH -o <your.slurm.home.path>/TReconLM/DeepLearningBaselines/RobuSeqNet/slurm_pkg/pretrain/train/60/logs/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/DeepLearningBaselines/RobuSeqNet/slurm_pkg/pretrain/train/60/logs/experiment.err
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
    export PYTHONPATH=/TReconLM/src:\$PYTHONPATH
    torchrun --nproc_per_node=8 --master_port=29515 train_robu_seqnet_ddp.py \
      --train_data None \
      --ground_truth None \
      --dynamic \
      --padding_length 80 \
      --label_length 60 \
      --wandb_log \
      --wandb_project Baselines \
      --wandb_entity <your.wandb.entity> \
      --max_iter 367333 \
      --base_out_dir "<your.storage.path>/model_checkpoints_RobuSeqNet/" \
      --batch_size_all 1500 \
      --max_lr 0.0009682458365518542 \
  "
