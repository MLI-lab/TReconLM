#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH -o <your.slurm.home.path>/TReconLM/src/slurm_pkg/pretrain/train/mamba/slurm_logs/mamba.out
#SBATCH -e <your.slurm.home.path>/TReconLM/src/slurm_pkg/pretrain/train/mamba/slurm_logs/mamba.err

IMG=<your.storage.path>/enroot/Treconlm.sqsh
CODE=$PWD/TReconLM
DATA=<your.storage.path>

# sanity‑check the container image exists
if [[ ! -f "$IMG" ]]; then
  echo "ERROR: container image not found at $IMG" >&2
  exit 1
fi

srun \
  --container-image="$IMG" \
  --container-mounts="$CODE:/TReconLM,$DATA:$DATA" \
  --container-workdir=/TReconLM \
  --container-env WANDB_API_KEY=<your.wandb.api.key> \
  bash -lc '
    set -euo pipefail
    source /opt/conda/etc/profile.d/conda.sh
    conda activate treconlm

    # 1) Upgrade pip & setuptools so we get the right wheels
    python -m pip install --upgrade pip setuptools

    # 2) Install PyTorch 2.4.0+cu118, TorchVision 0.19.0+cu118, TorchAudio 2.4.0+cu118
    #    all from the official PyTorch CUDA 11.8 index :contentReference[oaicite:0]{index=0}
    python -m pip install --no-cache-dir \
      torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 \
      --index-url https://download.pytorch.org/whl/cu118

    # 3) Install the pre‑built mamba_ssm CUDA extension wheel (no build needed)
    python -m pip install -q --no-cache-dir \
      https://github.com/state-spaces/mamba/releases/download/v2.2.5/\
mamba_ssm-2.2.5+cu11torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

    # 4) Sanity‑check the extension
    python - <<PY
import importlib.util
spec = importlib.util.find_spec("selective_scan_cuda")
if spec is None:
    raise ImportError("selective_scan_cuda not found")
print("Found selective_scan_cuda at", spec.origin)
PY

    # 5) Launch your training
    python src/pretrain.py exps=mamba/mamba_J
  '
