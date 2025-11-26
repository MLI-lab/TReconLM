#!/bin/bash
#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH -o <your.slurm.home.path>/TReconLM/src/slurm_pkg/pretrain/train/L110/logs42/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/src/slurm_pkg/pretrain/train/L110/logs42/experiment.err
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
     bash -c "
      cd /TReconLM
      source /opt/conda/etc/profile.d/conda.sh
      conda activate treconlm
      export WANDB_API_KEY="<your.wandb.api.key>8"
      python -m pip install transformers torchmetrics einops tabulate
      torchrun --nproc_per_node=4 --master_port=15772 src/pretrain.py exps=ids_110nt/ids_110_seed42
     "
CUDA_VISIBLE_DEVICES=0,2,3 torchrun --nproc_per_node=3 src/inference.py exps=ids_110nt
