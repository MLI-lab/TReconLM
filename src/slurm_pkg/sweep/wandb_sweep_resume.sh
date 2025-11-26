#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --nodes=1
#SBATCH --mem=30GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4                  
#SBATCH --time=48:00:00
#SBATCH -o <your.slurm.home.path>/TReconLM/src/slurm_pkg/sweep/logs/experiment.out
#SBATCH -e <your.slurm.home.path>/TReconLM/src/slurm_pkg/sweep/logs/experiment.err

NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
echo "Primary node: $NODE"

srun -N1 -n1 --nodelist="$NODE" \
     --container-image=<your.storage.path>/enroot/Treconlm.sqsh \
     --container-mounts="$PWD/TReconLM:/TReconLM,<your.storage.path>:<your.storage.path>" \
bash -c '
  set -euo pipefail

  cd /TReconLM
  source /opt/conda/etc/profile.d/conda.sh
  conda activate treconlm

  python -m pip install -q --upgrade wandb pyyaml transformers torchmetrics einops

  # W&B credentialsand sweep id if resumed from
  export WANDB_API_KEY=<your.wandb.api.key>8
  export WANDB_PROJECT=MicrosoftFinetune
  export WANDB_ENTITY=<your.wandb.entity>
  export SWEEP_ID=iryzendt

  export SWEEP_PATH="$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"
  echo "Sweep path is $SWEEP_PATH"

  # Split CUDA_VISIBLE_DEVICES 
  IFS=, read -r -a DEV_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
  NUM_DEVS=${#DEV_ARRAY[@]}
  echo "Container sees $NUM_DEVS GPU(s)/slice(s): ${DEV_ARRAY[*]}"

  # launch that many agents, one per device 
  for IDX in "${!DEV_ARRAY[@]}"; do
    (
      export CUDA_VISIBLE_DEVICES=${DEV_ARRAY[$IDX]}          # isolate
      export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512    # allocator hygiene
      echo "Agent $((IDX+1)) on $CUDA_VISIBLE_DEVICES"
      python -m wandb agent --count 30 "$SWEEP_PATH"
    ) &
  done

  wait   # keep allocation until all agents exit
'
