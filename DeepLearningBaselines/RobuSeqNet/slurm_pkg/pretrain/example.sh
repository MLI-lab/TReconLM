#!/bin/bash
#SBATCH -p <your_partition>
#SBATCH --qos=<your_qos>
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
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
    conda activate your_env_name
    pip install transformers torchmetrics einops
    export PYTHONPATH=/workspace/src:\$PYTHONPATH
    torchrun --nproc_per_node=2 --master_port=29513 train_robu_seqnet_ddp.py \
      --train_data ./data/reads.txt \
      --ground_truth ./data/reference.txt \
      --dynamic \
      --padding_length 150 \
      --label_length 110 \
      --wandb_log \
      --wandb_project <your_project_where_to_safe_run> \
      --wandb_entity <your_entity> \
      --max_iter <your_max_iter> \
      --base_out_dir ./model_checkpoints/ \
      --batch_size_all 64 \
      --max_lr 0.005 \
  "
