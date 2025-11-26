#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o <your.slurm.home.path>/TReconLM/src/slurm_pkg/finetune/inference/noisy/logs/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/src/slurm_pkg/finetune/inference/noisy/logs/experiment.err
#SBATCH --time=00:35:00

# Extract node lists for node groups
NODE_LIST=($(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0)) || { echo "Error fetching node list"; exit 1; }
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

      # activate Conda env
      source /opt/conda/etc/profile.d/conda.sh
      conda activate treconlm

      # install extra deps
      pip install transformers torchmetrics einops tabulate

      export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
      export CUDA_LAUNCH_BLOCKING=1


      # run inference
      torchrun --nproc_per_node=1 src/inference.py exps=noisy_DNA/pretrained_inference

  "
