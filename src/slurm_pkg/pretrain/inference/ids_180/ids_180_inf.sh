#!/bin/bash
#SBATCH -p lrz-dgx-a100-40x8-mig
#SBATCH --qos=mig
#SBATCH --nodes=1
#SBATCH --mem=60GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o <your.slurm.home.path>/TReconLM/src/slurm_pkg/pretrain/inference/ids_180/logs/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/src/slurm_pkg/pretrain/inference/ids_180/logs/experiment.err
#SBATCH --time=40:00:00

# Extract node lists for node groups
NODE_LIST=($(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0)) || { echo "Error fetching node list"; exit 1; }
NODE_1=${NODE_LIST[0]} || { echo "Error assigning NODE_1"; exit 1; }
REMAINING=("${NODE_LIST[@]:1}")

echo "Primary node: $NODE_1"
echo "Remaining nodes: ${REMAINING[@]}"

srun -N1 -n1 --nodelist="$NODE_1" \
  --container-image=<your.storage.path>/enroot/Treconlm.sqsh \
  --container-mounts="$PWD/TReconLM:/TReconLM,<your.storage.path>:<your.storage.path>" \
  bash -c '
    cd /TReconLM

    # activate Conda env
    source /opt/conda/etc/profile.d/conda.sh
    conda activate treconlm

    # install extra deps
    pip install --quiet torchmetrics
    export WANDB_API_KEY="<your.wandb.api.key>8"
    echo "WANDB_API_KEY length: ${#WANDB_API_KEY}"

    # run inference
    torchrun --nproc_per_node=1 --master_port=28181 -m src.inference exps=ids_180
  '
