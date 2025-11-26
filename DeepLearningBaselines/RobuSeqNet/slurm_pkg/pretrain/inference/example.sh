#!/bin/bash
#SBATCH -p <your_partition>
#SBATCH --qos=<your_qos>
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -o logs/experiment.out
#SBATCH -e logs/experiment.err
#SBATCH --time=48:00:00

srun \
  --container-image=/path/to/your_container.sqsh \
  --container-mounts="$PWD:/workspace" \
  --container-env WANDB_API_KEY=<your_wandb_api_key_or_env_var> \
  bash -c "
    cd /workspace/RobuSeqNet/examples
    source /opt/conda/etc/profile.d/conda.sh
    conda activate <your_env_name>
    pip install transformers torchmetrics einops
    python inference.py \
      --checkpoint ./model_checkpoints/<your_checkpoint>.pt \
      --artifact_name <your_artifact_name> \
      --test_project <your_project_where_test_artifact_is> \
      --project <your_project_where_to_save_run> \
      --wandb_entity <your_wandb_entity> \
      --padding_length 150 \
      --label_length 110
  "

