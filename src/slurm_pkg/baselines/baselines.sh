#!/bin/bash
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --mem=400GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -o <your.slurm.home.path>/TReconLM/src/slurm_pkg/baselines/logstrellisbma/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/src/slurm_pkg/baselines/logstrellisbma/experiment.err
#SBATCH --time=00:40:00

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
    python -m pip install torchmetrics

    # W&B env (set inside container)
    export WANDB_API_KEY="<your.wandb.api.key>8"
    echo "WANDB_API_KEY length: ${#WANDB_API_KEY}"

    # run inference
    python -u -m src.eval_pkg.eval_all_baselines \
        --alg trellisbma \
        --workers $SLURM_CPUS_PER_TASK \
        --project Timing \
        --timing \
  '
