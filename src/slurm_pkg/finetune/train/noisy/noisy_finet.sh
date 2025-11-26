#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --mem=30GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o <your.slurm.home.path>/TReconLM/src/slurm_pkg/finetune/train/noisy/logs/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/src/slurm_pkg/finetune/train/noisy/logs/experiment.err
#SBATCH --time=48:00:00

# Extract node lists for node groups
NODE_LIST=($(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0)) || { echo "Error fetching node list"; exit 1; }
NODE_1=${NODE_LIST[0]} || { echo "Error assigning NODE_1"; exit 1; }
REMAINING=("${NODE_LIST[@]:1}")

echo "Primary node: $NODE_1"
echo "Remaining nodes: ${REMAINING[@]}"

srun -N1 -n1 --nodelist="$NODE_1" \
  --container-image=<your.storage.path>/enroot/Treconlm.sqsh \
  --container-mounts="$PWD/TReconLM:/TReconLM,<your.storage.path>:<your.storage.path>" \
  --container-env WANDB_API_KEY=<your.wandb.api.key> \
  bash -c '
    cd /TReconLM

    # activate Conda env
    source /opt/conda/etc/profile.d/conda.sh
    conda activate treconlm

    # install extra deps
    python -m pip install transformers torchmetrics einops

    # run inference
    HYDRA_FULL_ERROR=1 HYDRA_VERBOSE=1 \
    python src/finetune.py exps=noisyDNA/noisy
  '
