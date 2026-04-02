#!/bin/bash
#SBATCH -p <YOUR_PARTITION> # set to your partition
#SBATCH --qos=<YOUR_QOS> # set to your QOS if needed
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8
#SBATCH -o <LOG_DIR>/experiment.out # set output log directory
#SBATCH -e <LOG_DIR>/experiment.err # set error log directory
#SBATCH --time=48:00:00

srun \
  --container-image=<PATH_TO_CONTAINER_IMAGE> \
  --container-mounts="<PATH_TO_PROJECT>:<CONTAINER_PROJECT_PATH>,<ADDITIONAL_MOUNTS>" \
  --container-env WANDB_API_KEY=<YOUR_WANDB_API_KEY> \
  bash -c "
    cd <CONTAINER_PROJECT_PATH>/DeepLearningBaselines/DNAFormer
    source /opt/conda/etc/profile.d/conda.sh
    conda activate treconlm
    pip install transformers torchmetrics einops
    export PYTHONPATH=<CONTAINER_PROJECT_PATH>/src:\$PYTHONPATH
    torchrun --nproc_per_node=8 --master_port=29511 <CONTAINER_PROJECT_PATH>/DeepLearningBaselines/DNAFormer/slurm_pkg/pretrain/L110/train_110.py \
  "

