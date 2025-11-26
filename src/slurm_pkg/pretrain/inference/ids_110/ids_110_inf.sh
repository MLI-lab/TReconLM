#!/bin/bash
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o <your.slurm.home.path>/TReconLM/src/slurm_pkg/pretrain/inference/ids_110/logs_beam/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/src/slurm_pkg/pretrain/inference/ids_110/logs_beam/experiment.err
#SBATCH --time=11:00:00

srun \
  --container-image=<your.storage.path>/enroot/Treconlm.sqsh \
  --container-mounts="$PWD/TReconLM:/TReconLM,<your.storage.path>:<your.storage.path>" \
  bash -c '

cd /TReconLM
source /opt/conda/etc/profile.d/conda.sh
conda activate treconlm

python3 -m pip install bitsandbytes

# W&B env (set inside container)
export WANDB_API_KEY="<your.wandb.api.key>8"
echo "WANDB_API_KEY length: ${#WANDB_API_KEY}"

export PYTHONPATH=/TReconLM/src:$PYTHONPATH
torchrun --nproc_per_node=1 --master_port=2815 -m src.inference exps=ids_110
'

