#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --mem=30GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH -o <your.slurm.home.path>/TReconLM/src/slurm_pkg/fixed_cluster_size/N60/logs_inf2/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/src/slurm_pkg/fixed_cluster_size/N60/logs_inf2/experiment.err
#SBATCH --time=48:00:00


# Extract node lists for node groups
NODE_LIST=($(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0)) || { echo "Error fetching node list"; exit 1; }

# Assign NODE_1 and remaining NODES
NODE_1=${NODE_LIST[0]} || { echo "Error assigning NODE_1"; exit 1; }
REMAINING=("${NODE_LIST[@]:1}")

echo "Primary node: $NODE_1"
echo "Remaining nodes: ${REMAINING[@]}"

# Run the main setup process on Node 1
srun -N1 -n1 --nodelist=$NODE_1 \
     --container-image=<your.storage.path>/enroot/Treconlm.sqsh \
     --container-mounts="$PWD/TReconLM:/TReconLM,<your.storage.path>:<your.storage.path>" \
     --container-env WANDB_API_KEY=<your.wandb.api.key> \
     bash -c "
      cd /TReconLM
      source /opt/conda/etc/profile.d/conda.sh
      conda activate treconlm
      torchrun --nproc_per_node=2 --master_port=29582 src/inference.py exps=fixedN/fixedN60_2
     "


