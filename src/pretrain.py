import os

import time
import math
import pickle
from contextlib import nullcontext # a dummy context manager that does nothing but keeps valid conditional logic in case we wanna use it (for mixed precision training)

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group # init process group initializes the default distributed process group, which is required to enable communication between multiple processes and destroy cleans them up once training is done
import torch.distributed as dist

print('device count: ', torch.cuda.device_count())

import json
import random

import hydra
from omegaconf import DictConfig, OmegaConf # to create, merge, and manipulate configs


from src.data_pkg.data_generation import data_generation, validate_block_size_for_variable_length
from src.data_pkg.prepare import encode_list, pad_encoded_data
from src.utils.helper_functions import extract_elements
from src.utils.wandb_utils import wandb_kwargs_via_cfg




def init_model(cfg, block_size, meta_vocab_size, stoi):
    """
    Build the model that will be trained from‑scratch.

    Returns
    -------
    model          : torch.nn.Module
    model_args     : dict   # will be written into the checkpoint
    compile        : bool   # should we torch.compile() the model?
    """
    model_type = cfg.model.model_type

    # ───────────────────────────── GPT ─────────────────────────────
    if model_type == "gpt":
        from src.gpt_pkg.model import GPTConfig, GPT

        gpt_cfg = GPTConfig(
            n_layer  = cfg.model.gpt_params.n_layer,
            n_head   = cfg.model.gpt_params.n_head,
            n_embd   = cfg.model.gpt_params.n_embd,
            block_size = block_size,
            bias     = cfg.model.gpt_params.bias,
            vocab_size = meta_vocab_size,
            dropout  = cfg.model.gpt_params.dropout,
            label_smoothing = 0,           # stays zero for pre‑training
        )
        model       = GPT(gpt_cfg)
        model_args  = vars(gpt_cfg).copy()
        model_args["model_type"] = "gpt"
        compile = True   # torch.compile brings speed‑up for Transformers

    # ───────────────────────────── LSTM ────────────────────────────
    elif model_type == "lstm":
        from src.rnn_pkg.lstm_model import LSTMConfig, LSTMConsensus

        lstm_cfg = LSTMConfig(
            vocab_size = meta_vocab_size,
            n_layer    = cfg.model.lstm_params.n_layer,
            n_embd     = cfg.model.lstm_params.n_embd,
            dropout    = cfg.model.lstm_params.dropout,
        )
        pad_id = stoi['#']  # Get padding token ID
        model       = LSTMConsensus(lstm_cfg, pad_id)
        model_args  = vars(lstm_cfg).copy()
        model_args["model_type"] = "lstm"
        compile = False  # compile() rarely helps small RNNs

    # ───────────────────────────── Mamba ───────────────────────────
    elif model_type == "mamba":
        from mamba_pkg.my_mamba_model import MambaLMHeadModel
        from mamba_pkg.my_config_mamba  import MambaConfig

        mparams = cfg.model.mamba_params
        mcfg = MambaConfig(
            d_model                  = mparams.d_model,
            d_intermediate           = mparams.d_intermediate,
            n_layer                  = mparams.n_layer,
            vocab_size               = meta_vocab_size,

            ssm_cfg                  = mparams.ssm_cfg,
            attn_layer_idx           = mparams.attn_layer_idx,
            attn_cfg                 = mparams.attn_cfg,
            rms_norm                 = mparams.rms_norm,
            residual_in_fp32         = mparams.residual_in_fp32,
            fused_add_norm           = mparams.fused_add_norm,
            pad_vocab_size_multiple  = mparams.pad_vocab_size_multiple,
            tie_embeddings           = mparams.tie_embeddings,
        )
        model       = MambaLMHeadModel(mcfg)
        model_args  = vars(mcfg).copy()
        model_args["model_type"] = "mamba"
        compile     = False



    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Expected 'gpt', 'lstm', or 'mamba'."
        )

    return model, model_args, compile


def load_model(cfg, device, out_dir, block_size, rank, stoi):
    """
    Loads a model checkpoint from out_dir/all_checkpoints/checkpoint_always.pt.
    Restores RNG states first (via a CPU load), then reloads the full checkpoint.

    Returns
    -------
    model : torch.nn.Module
    checkpoint : dict          # full checkpoint (already on correct device)
    checkpoint_model_args : dict
    compile : bool
    best_val_loss : float
    iter_num : int
    total_time_elapsed : float
    """

    from src.gpt_pkg.model import GPTConfig, GPT

    ckpt_path = os.path.join(out_dir, "all_checkpoints", "checkpoint_always.pt")
    print("ckpt_path:", ckpt_path)
    print("file exists:", os.path.exists(ckpt_path))
    print("device:", device)

    _cpu_ckpt = torch.load(ckpt_path, map_location="cpu")

    if "rng_states_per_rank" in _cpu_ckpt:
        rng_list = _cpu_ckpt["rng_states_per_rank"]
        if rank < len(rng_list):
            set_rng_state_dict(rng_list[rank])
        else:
            print(f"[Rank {rank}] WARNING: no RNG state found for this rank.")
    else:
        print(f"[Rank {rank}] WARNING: checkpoint contains no RNG states.")

    checkpoint = torch.load(ckpt_path, map_location=device)
    print("checkpoint loaded onto", device)

    checkpoint_model_args = checkpoint["model_args"]
    model_type            = checkpoint_model_args.get("model_type", "gpt")  # fall‑back

    # Rebuild correct model
    if model_type == "gpt":
        compile = True
        model_args = dict(
            n_layer    = cfg.model.gpt_params.n_layer,
            n_head     = cfg.model.gpt_params.n_head,
            n_embd     = cfg.model.gpt_params.n_embd,
            block_size = block_size,
            bias       = cfg.model.gpt_params.bias,
            vocab_size = None,                       # will be overwritten below
            dropout    = cfg.model.gpt_params.dropout,
        )
        for k in [
            "n_layer",
            "n_head",
            "n_embd",
            "block_size",
            "bias",
            "vocab_size",
            "dropout",
        ]:
            model_args[k] = checkpoint_model_args[k]

        gptconf = GPTConfig(**model_args)
        model   = GPT(gptconf)

    elif model_type == "lstm":
        from src.rnn_pkg.lstm_model import LSTMConfig, LSTMConsensus

        # strip helper key that the dataclass does not expect
        cfg_args  = {k: v for k, v in checkpoint_model_args.items()
                     if k != "model_type"}
        lstm_cfg  = LSTMConfig(**cfg_args)
        pad_id = stoi['#']  # Get padding token ID
        model     = LSTMConsensus(lstm_cfg, pad_id)

        compile    = False
        model_args = checkpoint_model_args

    elif model_type == "mamba":
        from mamba_pkg.my_mamba_model import Mamba
        from mamba_pkg.my_config_mamba import MambaConfig

        # strip helper key that the dataclass does not expect
        cfg_args = {k: v for k, v in checkpoint_model_args.items()
                    if k != "model_type"}
        mcfg  = MambaConfig(**cfg_args)
        model = Mamba(mcfg)

        compile    = False  
        model_args = checkpoint_model_args

    else:
        raise ValueError(f"Unsupported model_type '{model_type}' in checkpoint")

    state_dict      = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)


    iter_num           = checkpoint["iter_num"]
    best_val_loss      = checkpoint["best_val_loss"]
    total_time_elapsed = checkpoint.get("total_time_elapsed", 0.0)

    return (
        model,
        checkpoint,
        checkpoint_model_args,
        compile,
        best_val_loss,
        iter_num,
        total_time_elapsed,
    )


  
def set_rng_state_dict(rng_dict):
    import random, numpy as np, torch, warnings

    if 'py' in rng_dict:
        random.setstate(rng_dict['py'])
    if 'np' in rng_dict:
        np.random.set_state(rng_dict['np'])
    if 'torch_cpu' in rng_dict:
        torch.set_rng_state(rng_dict['torch_cpu'])

    if torch.cuda.is_available() and 'torch_cuda' in rng_dict:
        states = rng_dict['torch_cuda']
        # Support old checkpoints that stored a single tensor
        if torch.is_tensor(states):
            states = [states]
        n_visible = torch.cuda.device_count()
        if len(states) != n_visible:
            warnings.warn(
                f"Checkpoint has {len(states)} CUDA RNG states but {n_visible} GPUs are visible; "
                f"restoring only the first {min(len(states), n_visible)}."
            )
        for i in range(min(len(states), n_visible)):
            torch.cuda.set_rng_state(states[i], device=i)

def get_rng_state_dict():
    import random, numpy as np, torch
    out = {
        'py': random.getstate(),
        'np': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out['torch_cuda'] = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
    return out
    
def safe_save(checkpoint, dirpath, base_name="checkpoint_always.pt", backup_name="checkpoint_prev.pt"):
    """
    Write a new checkpoint and keep the previous one as a fallback.
    """
    import os, torch

    tmp_path    = os.path.join(dirpath, base_name + ".tmp")
    new_path    = os.path.join(dirpath, base_name)
    backup_path = os.path.join(dirpath, backup_name)

    # Save to a temp file first
    torch.save(checkpoint, tmp_path)

    # fsync to ensure the content is flushed
    with open(tmp_path, 'rb') as f:
        os.fsync(f.fileno())

    # Rotate: delete any old backup, move current to backup
    if os.path.exists(backup_path):
        os.remove(backup_path)
    if os.path.exists(new_path):
        os.replace(new_path, backup_path)

    # move temp to real
    os.replace(tmp_path, new_path)


@hydra.main(config_path='hydra/train_config', config_name='train_config.yaml', version_base=None) # reads the configuration (train_config.yaml) and initializes training.
def train(cfg: DictConfig) -> None:
    config_dict = wandb_kwargs_via_cfg(cfg)
    print(config_dict)

    # Checks if hydra config has field named additional_tags
    if hasattr(cfg, 'additional_tags'):
        additional_tags = list(cfg.additional_tags)
        print('additional_tags: ', additional_tags)
        print(type(additional_tags))
        config_dict['additional_tags'] = additional_tags
    else:
        additional_tags = []
    print(f"Additional tags are {additional_tags}")

    config_dict = OmegaConf.to_container(
        OmegaConf.create(config_dict),   # wrap so to_container works
        resolve=True,                    # ${foo.bar} -> actual values
        enum_to_str=True                 # just in case 
    )

    # region my_config
    # ---------------------------DIR--------------------------------------------------
    script_dir = os.path.dirname(__file__) # path of the folder where this script is located
    print("script_dir: ", script_dir)

    repo_path = os.path.dirname(script_dir) # Goes one level up 
    print("repo_path: ", repo_path)

    checkpoint_path = cfg.general.get('checkpoint_path', repo_path)
    print(f"Checkpoint path: {checkpoint_path}")

    data_pkg_dir = os.path.join(script_dir,'data_pkg')
    print("data_pkg_dir: ", data_pkg_dir)

    # ---------------------------GENERAL--------------------------------------------------
    init_from = cfg.general.init_from
    experiment  = cfg.experiment

    model_type  = cfg.model.model_type
    
    if init_from == 'scratch':
        now_str = cfg.general.now_str
    elif init_from == 'resume':
        train_time = cfg.general.train_time
            
    # ----------------------------------------data variables----------------------------------------
    block_size          = cfg.data.block_size
    data_type           = cfg.data.data_type
    observation_size    = cfg.data.observation_size
    lower_bound_obs_size = cfg.data.lower_bound_obs_size
    target_type         = cfg.data.target_type
    sequence_type       = cfg.data.sequence_type
    ground_truth_length = cfg.data.ground_truth_length

    # Format ground truth length for logging/naming (handles both fixed and interval lengths)
    def format_ground_truth_length(length_config):
        if isinstance(length_config, (list, tuple)):
            return f"{length_config[0]}-{length_config[1]}"
        else:
            return str(length_config)

    ground_truth_length_str = format_ground_truth_length(ground_truth_length)

    # Validate block size can accommodate variable ground truth lengths
    validate_block_size_for_variable_length(ground_truth_length, block_size, observation_size)

    insertion_probability_lb = cfg.data.insertion_probability_lb
    deletion_probability_lb = cfg.data.deletion_probability_lb
    substitution_probability_lb = cfg.data.substitution_probability_lb
    
    insertion_probability_ub = cfg.data.insertion_probability_ub
    deletion_probability_ub = cfg.data.deletion_probability_ub
    substitution_probability_ub = cfg.data.substitution_probability_ub
    
    # -----------------------------------TRAIN VARIABLES-----------------------------------
    eval_interval = cfg.train.eval_interval # check for new best val loss and save checkpoint
    log_interval  = cfg.train.log_interval # frequency of logging loss
    eval_iters    = cfg.train.eval_iters # for estimating loss over train/val during training
    eval_only     = cfg.train.eval_only # if True, script exits right after the first eval, useful for testing the model without modifying weights

    always_save_checkpoint = cfg.train.always_save_checkpoint # if True, always save a checkpoint after each eval

    device = cfg.train.device # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    gradient_accumulation_steps = cfg.train.gradient_accumulation_steps # used to simulate larger batch sizes
    batch_size = cfg.train.batch_size # if gradient_accumulation_steps > 1, this is the micro-batch size

    # adamw optimizer
    learning_rate = cfg.train.learning_rate # max learning rate
    max_iters = cfg.train.max_iters # total number of training iterations (how many batches will be processed).

    weight_decay = cfg.train.weight_decay # Used for L2 regularization to prevent overfitting. Adds a small penalty to large weights, forcing the model to prefer smaller values.
    beta1 = cfg.train.beta1 # Controls momentum of past gradients. Higher = More smooth updates, but slower adaptation. Instead of using only the current gradient, it keeps an exponential moving average of past gradients.
    beta2 = cfg.train.beta2 # Controls adaptive learning rate based on squared past gradients. Higher = More stable updates, but slower reaction to changes
    grad_clip = cfg.train.grad_clip # clip gradients at this value, or disable if == 0.0. If a gradient is too large, it gets scaled down to avoid instability. Recommended: grad_clip = 1.0 for transformers.

    # learning rate decay settings
    decay_lr = cfg.train.decay_lr  # whether to decay the learning rate

    if cfg.train.warmup_iters == 0:
        cfg.train.warmup_iters = int(0.05 * max_iters)

    warmup_iters = cfg.train.warmup_iters  # how many steps to warm up for
    print(f"warmup_iters is {warmup_iters}")
    warmup_iters = cfg.train.warmup_iters # how many steps to warm up for
    print(f"warmup_iters is {warmup_iters}")
    min_lr = cfg.train.min_lr # minimum learning rate, should be ~= learning_rate/10 per Chinchilla    
    lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla

    backend = 'nccl' # NCCL (NVIDIA Collective Communication Library) for fastest backend for GPU-based multi-GPU training
    empty_batch_flag = 0 # used to track if a batch was skipped (e.g., due to data loading issues)

    # ---------------------------WANDB and OUT DIR--------------------------------------------------
    
    # wandb logging
    wandb_log = cfg.wandb.wandb_log  # Flag determines if logging to WandB is enabled. If True the script will log training metrics (loss, learning rate, etc.) to WandB.

    wandb_project = cfg.wandb.wandb_project # name of the project in WandB where logs will be stored.

    group = f'{data_type}_{sequence_type}_{target_type}_obs{observation_size}_gt{ground_truth_length_str}_compute_{experiment}'  # creates a unique identifier for the experiment 


    if init_from == 'scratch':
        wandb_run_name = f'train_run_{model_type}_{now_str}' 
        print('wandb_run_name: ', wandb_run_name)
        
    elif init_from == 'resume':
        wandb_run_name = f'train_run_{model_type}_{train_time}' 
        print('wandb_run_name: ', wandb_run_name)

    # output dir where we safe our model checkpoints
    if experiment == None:
        out_dir = os.path.join(checkpoint_path, 'model_checkpoints', wandb_project, group, wandb_run_name)
    else:
        out_dir = os.path.join(checkpoint_path, 'model_checkpoints', wandb_project, group, experiment, wandb_run_name)
    

    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = cfg.train.device #'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler. Also 'bfloat16 more stable than float16, better for new GPUs (A100, H100).
    config_dict['dtype'] = dtype
    
    # -----------------------------------------------------------------------------
    ddp = int(os.environ.get('WORLD_SIZE', 1)) > 1 # environment variable Rank only set if its a ddp run 
    print('ddp: ', ddp)

    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK']) # Rank across all GPUs and nodes
        ddp_local_rank = int(os.environ['LOCAL_RANK']) # Rank (GPU index) within a single node
        ddp_world_size = int(os.environ['WORLD_SIZE']) # Total number of GPUs across all nodes.
        device = f'cuda:{ddp_local_rank}'
        print('device: ', device)
        torch.cuda.set_device(device) # Set the correct GPU for this process, ensures that each process only sees one GPU, and you don’t need to manually move the model or tensors to a specific GPU.
        master_process = ddp_rank == 0 # Only the first process (ddp_rank == 0) saves checkpoints and logs metrics. So if process runs on first gpu on first node set mster process flag to True  
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0 # Gradient accumulation to simulate lrger batch size without increasing memory that means not updating the weights after every batch, but instead summing gradients over multiple batches before updating the model.
        gradient_accumulation_steps //= ddp_world_size # need to devide by world size cause each GPU processes a different batch and updates are synchronized, so accross all GPUs we want to process in total gradient_accumulation_steps before updates weights not on each gpu individually

    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_local_rank = 0
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
        print('out_dir', out_dir)
        
        #create folder where to safe the checkpoints
        if always_save_checkpoint:
            path_all_ckpts = os.path.join(out_dir, 'all_checkpoints')
            os.makedirs(path_all_ckpts, exist_ok=True)    
    torch.backends.cuda.matmul.allow_tf32 = True # accelerates computations on Ampere (A100) and newer GPUs.
    torch.backends.cudnn.allow_tf32 = True # TF32 stores numbers with 8-bit exponents and 10-bit mantissas (vs. float16, which has only 5-bit exponents).
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9 # initialized to a large value, so any valid loss is considered an improvement.

    meta_path = os.path.join(data_pkg_dir,f'meta_{sequence_type}.pkl')  
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        config_dict['meta_vocab_size'] = meta_vocab_size
        stoi, itos = meta['stoi'], meta['itos'] # stoi converts characters into token IDs and itos vice versa (token id just from 0 to 7)
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

        if sequence_type == 'nuc':
            print("encode('ACGT-|:#'): ", encode("ACGT-|:#")) # - EOS/padding token
            print('decode([0,1,2,3,4,5,6,7]): ', decode([0,1,2,3,4,5,6,7]))
            
        else:
            raise ValueError(f"Invalid sequence_type. Expected one of: 'nuc', 'nuc_extended', 'amino'")
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # if resume will override random states for all ransk for which we have one saved
    base_seed = getattr(cfg.train, 'train_seed', 100)
    seed_number = base_seed + seed_offset
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # Set global seed once at beginning
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # determine the vocab size we'll use for from-scratch training
        model, model_args, compile = init_model(cfg = cfg, meta_vocab_size = meta_vocab_size, block_size = block_size, stoi = stoi)

        total_time_elapsed = 0.0   

    elif init_from == 'resume': 
        print("Resuming training from a checkpoint")
        model, checkpoint, model_args, compile, best_val_loss, iter_num, total_time_elapsed = load_model(cfg = cfg, device = device, out_dir = out_dir, block_size = block_size, rank=ddp_local_rank, stoi = stoi)

    config_dict['compile'] = compile
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16" and device_type == "cuda"))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        torch.cuda.set_device(device)
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
        print('compiled')
    # wrap model into DDP container so gradients are correctly shared and synchronized across all GPUs (without ech GPU would train seperately)
    # Ensures that gradients are correctly shared across all GPUs.
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    # endregion

    # region ----------------------------------- FUNCTIONS -----------------------------------
    def get_batch(seed=None):

        if seed is not None: 
            rng = random.Random(seed)
        else: 
            rng=None
        
        data_list = []
        batch_size_counter = 0
        sampled_lengths_batch = []  # Track sampled lengths for debugging

        while batch_size_counter < batch_size: 
            substitution_probability = random.uniform(substitution_probability_lb, substitution_probability_ub)
            insertion_probability = random.uniform(insertion_probability_lb, insertion_probability_ub)
            deletion_probability = random.uniform(deletion_probability_lb, deletion_probability_ub)

            channel_statistics = {'substitution_probability': substitution_probability,
                                    'insertion_probability': insertion_probability,
                                    'deletion_probability': deletion_probability}

            local_obs_size = random.randint(lower_bound_obs_size, observation_size) #randomly sample observation size (including observation_size)

            # Get misclustering config if enabled
            misclustering_config = cfg.data.get('misclustering_training', None)

            # Generates one noisy sequence at a time
            data = data_generation(data_set_size = 1 , observation_size = local_obs_size,
                                   length_ground_truth = ground_truth_length, channel_statistics = channel_statistics,
                                    target_type = target_type, data_type =  data_type, rng=rng,
                                    misclustering_config=misclustering_config)
            # Extracts the target type value, i.e. concatenated noisy reads, e.g., if alignment 'T--T' for one read, if not alignment and CPRED just TT
            data = extract_elements(data, target_type)[0]

            if len(data[0]) >= block_size-1:
                print(f'len: {len(data[0])} - DATA LENGTH GREATER THAN BLOCK SIZE')
                continue
            else:
                batch_size_counter += 1 #fills batch with observation size many noisy copies for each ground truth until batch size reached i.e. can have different number of ground truth many in a batch
                # Track the sampled ground truth length (extract from generated data for CPRED)
                if target_type == 'CPRED' and ':' in data[0]:
                    gt_seq = data[0].split(':')[1]
                    sampled_lengths_batch.append(len(gt_seq))

            data_list.append(data[0])


        # Log sampled lengths for debugging (only for variable length configs)
        from omegaconf import ListConfig
        if sampled_lengths_batch and isinstance(ground_truth_length, (list, tuple, ListConfig)):
            print(f"[DEBUG] Batch sampled GT lengths: {sampled_lengths_batch}, mean: {sum(sampled_lengths_batch)/len(sampled_lengths_batch):.1f}")

        encoded_data = encode_list(data_list,stoi)
        # Compute actual sequence lengths before padding (for LSTM efficiency)
        lengths = torch.tensor([len(seq) for seq in encoded_data], dtype=torch.long)

        padded_encoded_data = pad_encoded_data(encoded_data, block_size, stoi)  # block_size is essentially the context length, defines how many tokens the model can process at once.
        np_data = np.array(padded_encoded_data, dtype = np.int64) # Converts to NumPy array for efficient tensor operations.
        # Each token in y is the next token in x
        # The model learns to predict y[i] given x[i]
        # dim 0 is batch dim
        x_temp = np_data[:, 0:block_size-1] # takes tokens from index 0 to block_size-1 (excluding the last token).
        y_temp = np_data[:, 1:block_size] #y_temp takes tokens from index 1 to block_size (shifting by one position)
        x = torch.from_numpy(x_temp.astype(np.int64))
        y = torch.from_numpy(y_temp.astype(np.int64))

        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True) --> for faster GPU transfers.
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
            lengths = lengths.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
            lengths = lengths.to(device)

        return x, y, lengths
    
    @torch.no_grad()
    def estimate_train_loss():
        model.eval()
        losses = torch.zeros(eval_iters, device=device)  # device so .mean() is fast
        for k in range(eval_iters):
            X, Y, lengths = get_batch()
            with ctx:
                # Pass lengths only for LSTM model
                if model_type == "lstm":
                    _, loss = model(input_ids=X, targets=Y, stoi=stoi, lengths=lengths)
                else:
                    _, loss = model(input_ids=X, targets=Y, stoi=stoi)
            losses[k] = loss
        model.train()
        return {"train": losses.mean().item()}


    @torch.no_grad()
    def estimate_val_loss():
        model.eval()
        val_losses = []
        for Xb_cpu, Yb_cpu in val_cache:
            Xb = Xb_cpu.to(device, non_blocking=True)
            Yb = Yb_cpu.to(device, non_blocking=True)
            with ctx:
                # Pass lengths only for LSTM model (note: lengths not available for val_cache)
                _, loss = model(input_ids=Xb, targets=Yb, stoi=stoi)
            val_losses.append(loss.item())
        model.train()
        return {"val": float(np.mean(val_losses))}


    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
    # endregion

    ########################## With fixed val set call just once and not in train loop ############################

    val_cache = []
    for _ in range(eval_iters):
        Xb, Yb, lengths_b = get_batch(seed=0)   
        val_cache.append((Xb.cpu(), Yb.cpu())) 

    if master_process:                              # print only once
        print(f"val batches on this rank     : {len(val_cache)}")
        print(f"total val batches (all GPUs): {len(val_cache)*ddp_world_size}")

    # region ----------------------------------- TRAINING LOOP -----------------------------------
    if master_process:
        if init_from == 'scratch':
            json_file_path = os.path.join(out_dir,  f'training_params_{init_from}_{now_str}.json')
        elif init_from == 'resume':
            json_file_path = os.path.join(out_dir,  f'training_params_{init_from}_{train_time}.json')
        
        with open(json_file_path, 'w') as f:
            json.dump(config_dict, f, indent = 4)

    if wandb_log and master_process:
        import wandb

        run_id = None
        run_id_path = os.path.join(out_dir, 'wandb_run_id.txt')
        if init_from == 'resume' and os.path.exists(run_id_path):
            with open(run_id_path, 'r') as f:
                run_id = f.read().strip()

        if experiment == None:
            wandb.init(project=wandb_project, entity=cfg.wandb.wandb_entity, group=group, name=wandb_run_name, job_type = 'training', config=config_dict, dir = out_dir, id=run_id, resume="must" if run_id else None)
        else:
            wandb.init(project=wandb_project, entity=cfg.wandb.wandb_entity, group=group, tags = [experiment] + additional_tags, name=wandb_run_name, job_type = 'training', config=config_dict, dir = out_dir, id=run_id, resume="must" if run_id else None)

        if init_from == 'scratch':
            run_id = wandb.run.id
            with open(run_id_path, 'w') as f:
                f.write(run_id)

    # training loop
    X, Y, lengths = get_batch() # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0

    if master_process:
        lock = True

    if master_process:
        decoded_inputs = [decode(seq.tolist()) for seq in X[:5]]
        print("\nExample training inputs (x):")
        for i, s in enumerate(decoded_inputs):
            print(f"Example {i+1}: {s}")

    start_time_for_speed = time.time()
    start_iter_num = iter_num
    speed_checked = False

    # ------ GPU TIMER START ------
    if device_type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    # ------ GPU TIMER START ------

    session_start_time = time.perf_counter()

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        # Kind of inefficient as for now as all ranks estimate same loss but overhead should be little as val and test set tiny 
        if iter_num > 0 and iter_num % eval_interval == 0:
            train_stats = estimate_train_loss() 
            val_stats   = estimate_val_loss()

            train_loss = train_stats['train']
            val_loss   = val_stats['val']
            if wandb_log and master_process:
                print(f"step {iter_num}: train {train_loss:.4f} | val {val_loss:.4f}")
                wandb.log({"iter": iter_num,
                        "train/loss": train_loss,
                        "val/loss":   val_loss,
                        "lr":         lr})

            if val_loss < best_val_loss and master_process: # or always_save_checkpoint:
                best_val_loss = val_loss
                if iter_num > 0:
                    print('NEW VAL LOSS MINIMUM: ', best_val_loss)
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config_dict,
                        'wandb_run_id': run_id if wandb_log else None, 
                    }
                    print(f"Save checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_best.pt')) 

        if always_save_checkpoint and iter_num % 500 == 0 and iter_num > 0:
            local_rng = get_rng_state_dict()

            if ddp:
                dist.barrier()
                print(f"[Rank {ddp_local_rank}] Gathering RNG states")
                if ddp_rank == 0:
                    all_rng_states = [None for _ in range(ddp_world_size)]
                else:
                    all_rng_states = None

                dist.gather_object(local_rng, object_gather_list=all_rng_states, dst=0)
            else:
                all_rng_states = [local_rng]  # non-DDP single process

            if master_process:                  
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss, 
                    'config': config_dict,
                    'rng_states_per_rank': all_rng_states,
                    'total_time_elapsed': total_time_elapsed, 

                }
                print('ALWAYS SAVE CHECKPOINT')
                print(f"saving checkpoint to {path_all_ckpts}")
                #if just save risk corrupted checkpoints with SLURM scripts
                #torch.save(checkpoint, os.path.join(path_all_ckpts, f'checkpoint_always.pt')) 
            
                safe_save(checkpoint,dirpath = path_all_ckpts, base_name   = "checkpoint_always.pt",backup_name = "checkpoint_prev.pt")


        if iter_num == 0 and eval_only: # eval_only=True, then only want to estimate validation loss once and not train. --> so exit the loop right away, after the first evaluation 
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1) #Ensures that gradients are only synchronized across GPUs during the last micro-step.
            with ctx:
                # Pass lengths only for LSTM model
                if model_type == "lstm":
                    logits, loss = model(input_ids = X, targets = Y, stoi = stoi, lengths = lengths) # model(X, Y) forward pass
                else:
                    logits, loss = model(input_ids = X, targets = Y, stoi = stoi) # model(X, Y) forward pass
                loss = loss / gradient_accumulation_steps # loss is sum of losses in each mini batch in gradient accumulation, so take average by dividing by number of mini batches
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y, lengths = get_batch()

            if not torch.isfinite(loss):
                print(f"[ERROR] Loss blew up at iter {iter_num}: {loss}")
                break   

            # backward pass, with gradient scaling if training in fp16 to compute gradients
            scaler.scale(loss).backward() 
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer) # Updates weights
        scaler.update() # Ensures float16 gradients don’t underflow 
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True) # Clears gradients to prevent accumulation across iterations.

        # timing and logging
        t1 = time.time()
        dt = t1 - t0                 # wall-clock time of this iteration
        total_time_elapsed += dt     # accumulate into the run-wide counter
        t0 = t1                      # set up for the next iter
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps 
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt) # estimate_estimates how efficiently the model is utilizing available FLOP ( not build in method )
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            
        iter_num += 1
        local_iter_num += 1

        # Measure how many iterations completed after 5 minutes
        if master_process and not speed_checked and (time.time() - start_time_for_speed) > 300:
            iterations_in_5min = iter_num - start_iter_num
            avg_iter_per_minute = iterations_in_5min / 5
            print(f"\nIn first 5 minutes: {iterations_in_5min} iterations completed.")
            print(f"Average iterations per minute: {avg_iter_per_minute:.2f}\n")
            speed_checked = True
        
        if master_process: #Ensures that only one process saves checkpoints and logs.
            lock = False

        # termination conditions
        if iter_num >= max_iters:
            # ------ GPU TIMER END ------
            if device_type == 'cuda':
                end_event.record()
            if device_type == 'cuda':
                torch.cuda.synchronize()  # wait for everything to finish
                gpu_time_ms = start_event.elapsed_time(end_event)
            else:
                gpu_time_ms = 0  # No GPU timing on CPU
            # ------ GPU TIMER END ------
            break
    
    if master_process:
        # region ------------------------------------ FINISH ------------------------------------
        print("Training finished")
        print('empyt_batch_flag: ', empty_batch_flag) 
    
        if wandb_log:
            wandb.log({
                "total_gpu_time_master_process": gpu_time_ms,
                "total_gpu_hours_compute": (gpu_time_ms / 1000 / 3600) * ddp_world_size,
                "iter": iter_num,
                "best_val_loss": best_val_loss,
                "wall_clock_hours": total_time_elapsed/3600,
            })

            
    if wandb_log and master_process and cfg.data.test: # for scaling laws to quickly get loss after training
        print("Testing on artifact-based test dataset.")

        # Load artifact

        if hasattr(cfg.data, "test_artifact") and cfg.data.test_artifact:
            artifact_name = cfg.data.test_artifact
        else:
            artifact_name = f"test_dataset_seed{cfg.data.test_seed}_gl{ground_truth_length_str}_bs{block_size}_ds{cfg.data.test_dataset_size}"

        test_project = cfg.wandb.test_project if hasattr(cfg.wandb, "test_project") else wandb_project
        artifact = wandb.use_artifact(f"{cfg.wandb.wandb_entity}/{test_project}/{artifact_name}:latest", type="dataset")


        artifact_dir = artifact.download()

        # Clear current model to free memory before loading best
        del model
        torch.cuda.empty_cache()

        # Load best model from checkpoint_best.pt
        from gpt_pkg.model import GPT, GPTConfig
        ckpt_path = os.path.join(out_dir, 'checkpoint_best.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        model_args = checkpoint["model_args"].copy()
        model_args.pop("model_type", None)   
        gptconf    = GPTConfig(**model_args)

        model = GPT(gptconf)

        # fix unwanted '_orig_mod.' prefix ---
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        # ---------------------------------------------

        model.load_state_dict(state_dict)
        model.to(device)

        try:
            # Load full test data to CPU
            x_test = torch.load(os.path.join(artifact_dir, 'test_x.pt'), map_location='cpu')
            y_test = torch.load(os.path.join(artifact_dir, 'test_y.pt'), map_location='cpu')

            batch_size = 32
            num_batches = (x_test.size(0) + batch_size - 1) // batch_size

            model.eval()
            total_loss = 0.0
            total_samples = 0

            log_interval = 500  

            with torch.no_grad():
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, x_test.size(0))

                    # Use pin_memory and non_blocking=True
                    xb = x_test[start_idx:end_idx].pin_memory().to(device, non_blocking=True)
                    yb = y_test[start_idx:end_idx].pin_memory().to(device, non_blocking=True)

                    with torch.autocast(device_type=device_type, dtype=ptdtype):
                        # Note: For cached validation, we don't have lengths stored, so LSTM will infer them
                        logits, loss = model(input_ids=xb, targets=yb, stoi=stoi)

                    total_loss += loss.item() * (end_idx - start_idx)
                    total_samples += (end_idx - start_idx)

                    # Free the batch and empty cache
                    del xb, yb, logits
                    #torch.cuda.empty_cache()
                    if (batch_idx + 1) % 100 == 0:
                        torch.cuda.empty_cache()

                    # Log progress
                    if (batch_idx + 1) % log_interval == 0 or batch_idx == num_batches - 1:
                        print(f"Processed {batch_idx + 1}/{num_batches} batches "
                            f"({(batch_idx + 1) * batch_size}/{x_test.size(0)} samples)")

            avg_test_loss = total_loss / total_samples
            print(f"Test loss: {avg_test_loss:.4f}")

            if wandb_log:
                wandb.log({"test/loss": avg_test_loss})

        except Exception as e:
            print(f"Error loading the test data or during evaluation: {e}")

    if wandb_log and master_process:
        wandb.finish()

    if ddp:
        dist.barrier()
        destroy_process_group()

    # endregion
@hydra.main(config_path="conf", config_name="test_config", version_base=None)
def test(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg)) # Run train() with eval_only = True instead of calling test() as test only prints its configs 
    
if __name__ == "__main__":

    # dont use test() but instead train(eval_only=True)
    train()