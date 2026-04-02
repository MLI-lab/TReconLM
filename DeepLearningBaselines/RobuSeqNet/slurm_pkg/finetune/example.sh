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
    torchrun --nproc_per_node=1 --master_port=29511 finetune.py \
      --train_data ./data/train.txt \
      --val_data ./data/val.txt \
      --padding_length 150 \
      --label_length 110 \
      --base_out_dir ./model_checkpoints/ \
      --wandb_log \
      --wandb_project <your_wandb_project_where_to_safe_the_run> \
      --wandb_entity <your_entity> \
      --max_iter <your_max_iter> \
      --batch_size_all 64 \
      --max_lr 0.005 \
      --pretrain_run_name <your_pretrain_run_name> \
  "
