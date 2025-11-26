#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH -o <your.slurm.home.path>/TReconLM/src/slurm_pkg/pretrain/inference/lstm/logs_4/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/src/slurm_pkg/pretrain/inference/lstm/logs_4/experiment.err
#SBATCH --time=02:00:00

srun \
  --container-image=<your.storage.path>/enroot/Treconlm.sqsh \
  --container-mounts="$PWD/TReconLM:/TReconLM,<your.storage.path>:<your.storage.path>" \
  --container-env WANDB_API_KEY=<your.wandb.api.key> \
  bash -c "
    cd /TReconLM
    source /opt/conda/etc/profile.d/conda.sh
    conda activate treconlm
    pip install transformers torchmetrics einops tabulate
    export PYTHONPATH=/TReconLM/src:\$PYTHONPATH
    torchrun --nproc_per_node=2 --master_port=29459 -m src.inference exps=lstm
  "
