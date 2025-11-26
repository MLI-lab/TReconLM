#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH -o <your.slurm.home.path>/TReconLM/src/slurm_pkg/pretrain/train/mamba/logs/mamba.out
#SBATCH -e <your.slurm.home.path>/TReconLM/src/slurm_pkg/pretrain/train/mamba/logs/mamba.err

IMG=<your.storage.path>/enroot/mamba.sqsh
CODE=$PWD/TReconLM
DATA=<your.storage.path>

# sanity check
if [[ ! -f "$IMG" ]]; then
  echo "ERROR: container image not found at $IMG" >&2
  exit 1
fi

srun \
  --container-image="$IMG" \
  --container-mounts="$CODE:/TReconLM,$DATA:$DATA" \
  --container-workdir=/TReconLM \
  --container-env WANDB_API_KEY=<your.wandb.api.key> \
  bash -lc "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate treconlm
    python -m pip install --no-cache-dir omegaconf hydra-core wandb psutil pytz
    export TORCH_CUDA_ARCH_LIST=8.0   # Ensure correct Triton arch for A100
    python src/pretrain.py exps=mamba/mamba_J
  "
