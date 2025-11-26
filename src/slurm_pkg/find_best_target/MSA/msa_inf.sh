#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:8
#SBATCH -o <your.slurm.home.path>/TReconLM/src/slurm_pkg/find_best_target/MSA_smalllr/logs_inf4/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/src/slurm_pkg/find_best_target/MSA_smalllr/logs_inf4/experiment.err
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
      cd /TReconLM/src
      source /opt/conda/etc/profile.d/conda.sh
      conda activate treconlm
      python inference_all.py exps=inference/msa
     "


