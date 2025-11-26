# TReconLM Hydra Configuration Guide

TReconLM uses Hydra for configuration management. This script explains all configuration options for pretraining, fine-tuning, and inference.

---

## Pretraining Configuration

Run pretraining with: `python src/pretrain.py exps=<experiment>`

### Experiment Config (exps/microsoft/mic.yaml)

```yaml
# @package _global_
# Example: python pretrain.py exps=test/pretrain_scratch

defaults:
  - override /general: scratch        # scratch = new training, resume = continue from checkpoint
  - override /data: ids_data          # Data type: ids_data, subs (substitution only)
  - override /model: gpt              # Model: gpt, lstm, mamba
  - override /train: base             # Training hyperparameters

project: TRACE_RECONSTRUCTION         # WandB project name
experiment: pretraining_test          # Experiment name for tracking

# General settings
general:
  init_from: scratch                  # scratch or resume
  checkpoint_path: /path/to/save      # Directory for saving checkpoints
  # For resume:
  # train_time: '20250101_120000'     # Timestamp of checkpoint to resume

# Data configuration
data:
  data_type: ids_data                 # Data generator type
  sequence_type: nuc                  # nuc = DNA nucleotides
  target_type: CPRED                  # Output format (see options below)
  observation_size: 10                # Max cluster size
  lower_bound_obs_size: 2             # Minimum cluster size 
  ground_truth_length: 110            # Sequence length (60, 110, or 180)
  block_size: 1500                    # Context window (800/1500/2400 for 60/110/180)

  # IDS noise parameters (bounds for random sampling during training)
  substitution_probability_lb: 0.01   # Min substitution rate
  substitution_probability_ub: 0.1    # Max substitution rate
  insertion_probability_lb: 0.01      # Min insertion rate
  insertion_probability_ub: 0.1       # Max insertion rate
  deletion_probability_lb: 0.01       # Min deletion rate
  deletion_probability_ub: 0.1        # Max deletion rate

  # Misclustering augmentation (optional)
  misclustering_training:
    enabled: false                    # Enable contamination during training
    contamination_rate_lb: 0.02       # Min contamination rate
    contamination_rate_ub: 0.2        # Max contamination rate

  # Test evaluation (optional)
  test: false                         # Evaluate on test set after training
  test_seed: 34721                    # Seed for test set generation
  test_dataset_size: 50000            # Number of test examples

# Training hyperparameters
train:
  # Distributed training
  ddp: ~                              # true/false/~ (auto-detect)

  # Evaluation and logging
  eval_interval: 500                  # Evaluate every N iterations
  log_interval: 10                    # Log to WandB every N iterations
  eval_iters: 1000                    # Batches for evaluation loss estimate
  eval_only: false                    # Skip training, only evaluate

  # Checkpointing
  always_save_checkpoint: true        # Save checkpoints at intervals
  always_interval: 500                # Checkpoint save interval

  # Device and memory
  device: cuda:0                      # Device (cuda:0, cuda:1, cpu)
  batch_size: 50                      # Batch size per GPU
  gradient_accumulation_steps: 16     # Effective batch = batch_size * grad_accum * num_gpus

  # Optimizer (AdamW)
  learning_rate: 0.0001               # Peak learning rate
  weight_decay: 0.1                   # L2 regularization
  beta1: 0.9                          # Adam momentum
  beta2: 0.95                         # Adam adaptive LR
  grad_clip: 1.0                      # Gradient clipping (0.0 = disabled)

  # Learning rate schedule
  max_iters: 500000                   # Total training iterations
  decay_lr: true                      # Use cosine decay
  warmup_iters: 0                     # Warmup steps (0 = auto 5% of max_iters)
  min_lr: 0.0                         # Minimum LR after decay
  lr_decay_iters: ${train.max_iters}  # Decay schedule length

  # Reproducibility
  train_seed: 1                       # Random seed
  # patience: 10                      # Early stopping (optional)

# Model architecture
model:
  model_type: gpt                     # gpt, lstm, or mamba

  # GPT parameters
  gpt_params:
    n_layer: 12                       # Number of transformer layers
    n_head: 8                         # Number of attention heads
    n_embd: 512                       # Embedding dimension
    dropout: 0.0                      # Dropout (0 for pretraining, 0.1+ for finetuning)
    bias: false                       # Bias in linear layers

  # LSTM parameters (if model_type: lstm)
  # lstm_params:
  #   hidden_size: 512
  #   num_layers: 4
  #   dropout: 0.1

  # Mamba parameters (if model_type: mamba)
  # mamba_params:
  #   n_layer: 4
  #   d_model: 384
  #   d_intermediate: 1536

# WandB logging
wandb:
  wandb_log: true                     # Enable WandB logging
  wandb_project: ${project}           # Project name
  wandb_entity: your-entity           # WandB entity/team
```

---

## Fine-tuning Configuration

Run fine-tuning with: `python src/finetune.py exps=<experiment>`

### Experiment Config (e.g., exps/microsoft/mic.yaml)

```yaml
# @package _global_
# Example: python finetune.py exps=microsoft/mic

defaults:
  - override /general: scratch        # scratch or resume
  - override /model: gpt              # Must match pretrained model
  - override /train: base             # Training config
  - override /finetune: microsoft     # Loads finetune/microsoft.yaml

project: FinetuneMicrosoft            # WandB project
experiment: finetune_microsoft        # Experiment name

# General settings
general:
  checkpoint_path: /path/to/save      # Where to save finetuned checkpoints

# Pretraining data config (for model initialization compatibility)
data:
  data_type: ids_data
  sequence_type: nuc
  observation_size: 10
  target_type: CPRED
  ground_truth_length: 110
  lower_bound_obs_size: 2
  block_size: 1500

# Training hyperparameters (fine-tuning specific)
train:
  ddp: false
  eval_interval: 250                  # More frequent evaluation
  log_interval: 10
  eval_iters: 250
  eval_only: false
  always_interval: 250
  always_save_checkpoint: true
  device: cuda:0
  gradient_accumulation_steps: 1
  batch_size: 25                      # Often smaller than pretraining
  learning_rate: 0.00001              # Lower LR (typically 10x smaller)
  max_iters: 566047
  weight_decay: 0.001                 # Often lower than pretraining
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  decay_lr: true
  warmup_iters: 0                     # Auto-set to 1% of max_iters
  min_lr: 0.0
  lr_decay_iters: ${train.max_iters}
  seed: 1

# Model architecture (MUST match pretrained checkpoint)
model:
  gpt_params:
    n_layer: 12                       # Must match checkpoint
    n_head: 8                         # Must match checkpoint
    n_embd: 512                       # Must match checkpoint
    dropout: 0.1                      # Higher dropout for finetuning
    bias: false

# WandB logging
wandb:
  wandb_log: true
```

### Finetune Dataset Config (e.g., finetune/microsoft.yaml)

Create a new file in `train_config/finetune/` for each dataset:

```yaml
defaults:
  - base

finetune_data_type: microsoft_data    # Dataset type
finetune_target_type: CPRED           # Should match pretraining
finetune_observation_size: 10         # Max cluster size
finetune_lower_bound_obs_size: 2      # Min cluster size
finetune_ground_truth_length: 110     # Sequence length
finetune_sequence_type: nuc
finetune_experiment: finetune_microsoft

data_folder: /path/to/finetune/data   # Path to fine-tuning dataset
finetune_filename_train: train.txt    # Training data filename
finetune_filename_val: val.txt        # Validation data filename

model_dir: /path/to/pretrained/       # Path to pretrained checkpoint folder
```
This file should be set in  `override /finetune` in exps

---

## Inference Configuration

Run inference with: `python src/inference.py exps=<experiment>`

### Complete Annotated Config

```yaml
# @package _global_
# Example: python inference.py exps=ids_110

defaults:
  - override /data: ids_data
  - override /model: gpt

project: Inference                    # WandB project
experiment: ids_110_inference         # Experiment name

# Data configuration
data:
  # Option 1: Local data directory 
  local_data_dir: /path/to/data       # Directory with test_x.pt and ground_truth.txt

  # Option 2: WandB artifact (comment out local_data_dir)
  # artifact_name: test_dataset_seed34721_gl110_bs1500_ds50000


  # Common settings
  ground_truth_length: 110            # Expected sequence length
  block_size: 1500                    # Must match training
  batch_size: 50                      # Inference batch size
  test_seed: 34721                    # For on-the-fly generation
  observation_size: 10                # Cluster size
  lower_bound_obs_size: 2             # Min cluster size
  test_dataset_size: 50000            # For on-the-fly generation

# Model and sampling configuration
model:
  checkpoint_path: /path/to/checkpoint_best.pt

  # Performance optimization
  compile: false                      # torch.compile for 30-200% speedup
  compile_mode: default               # default, reduce-overhead, max-autotune
  use_int8_quantization: false        # INT8 quantization, keep false is actually slower with true

  # Sampling strategy
  sampling:
    temperature: 1.0                 
    top_k: null                       # Top-k sampling (null = disabled)
    strategy: greedy                  # greedy or beam_search 
    max_new_tokens: ${data.ground_truth_length}
    constrained_generation: false     # Constrain output to ACTG only

    # Beam search (if strategy: beam_search)
    # beam_width: 6

    # Majority voting decoding
    majority_voting:
      enabled: false                  # Enable permutation-based voting
      max_permutations: 10            # Max permutations (actual = min(max, N!))
      seed: 42                        # Reproducibility
      tie_breaking_strategy: random   # random or first_prediction

# Attention tracking (optional)
attention:
  track_attention: false              # Save attention maps
  track_all_layers: false             # All layers (true) or last only (false)
  attention_output_dir: ./attention_output

# Misclustering simulation (optional)
misclustering:
  enabled: false                      # Enable contamination simulation
  contamination_rate: 0.1             # Fixed contamination rate
  contaminated_attention_output_dir: ./contaminated_attention
  run_baseline_on_subset: false       # Also run clean inference for comparison

# Timing mode (optional)
timing:
  enabled: false                      # Throughput measurement mode
  run_duration: 300                   # Seconds per run
  num_runs: 6                         # Total runs
  warmup_runs: 1                      # Warmup runs (discarded)
  log_interval: 100                   # Log every N examples

# WandB logging
wandb:
  wandb_log: true
  wandb_project: ${project}
```
