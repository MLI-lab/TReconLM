#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH -o <your.slurm.home.path>/TReconLM/src/slurm_pkg/pretrain/inference/mamba/logs/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/src/slurm_pkg/pretrain/inference/mamba/logs/experiment.err

# ----------------------------------------------------------------------
# Paths (edit only these three if your layout changes)
# ----------------------------------------------------------------------
IMG=<your.storage.path>/enroot/mamba.sqsh
CODE=<your.slurm.home.path>/TReconLM          # project *root* on the host
DATA=<your.storage.path>

# sanity check ----------------------------------------------------------
if [[ ! -f "$IMG" ]]; then
  echo "ERROR: container image not found at $IMG" >&2
  exit 1
fi
if [[ ! -d "$CODE" ]]; then
  echo "ERROR: project root not found at $CODE" >&2
  exit 1
fi

# ----------------------------------------------------------------------
# Run inside the container
# ----------------------------------------------------------------------
srun \
  --container-image="$IMG" \
  --container-mounts="$CODE:/TReconLM,$DATA:$DATA" \
  --container-workdir=/TReconLM \
  --container-env WANDB_API_KEY=<your.wandb.api.key> \
  bash -lc '
    # ---- Conda env -----------------------------------------------------
    source /opt/conda/etc/profile.d/conda.sh
    conda activate treconlm

    # ---- Python deps ---------------------------------------------------
    python -m pip install --no-cache-dir \
        omegaconf hydra-core wandb psutil pytz Levenshtein

    # ---- Ensure repo root is on sys.path -------------------------------
    export PYTHONPATH=/TReconLM:$PYTHONPATH

    # ---- Triton kernels for A100 ---------------------------------------
    export TORCH_CUDA_ARCH_LIST=8.0

    # ---- Launch inference ----------------------------------------------
    torchrun --nproc_per_node=1 --master_port=29598 \
             -m src.inference exps=mamba
  '
