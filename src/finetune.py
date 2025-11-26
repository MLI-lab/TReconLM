import os
import sys

import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchmetrics.classification import MulticlassAccuracy
from torch.distributed import all_reduce, ReduceOp
import torch.distributed as dist


print('device count: ', torch.cuda.device_count())

import json
import random

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


from src.data_pkg.prepare import encode_list, pad_encoded_data
from src.utils.data_functions import load_data_from_file
from src.utils.wandb_utils import wandb_kwargs_via_cfg
from src.utils.hamming_distance import hamming_distance_postprocessed

def init_model(cfg, block_size, meta_vocab_size): 
    """
    Initializes a GPT model from config with block size and vocab size as extra arguments.
    
    Returns the model, its config args, and a compile flag (for PyTorch 2.0+).
    """
    model_type = cfg.model.model_type

    if model_type == 'gpt':
        from src.gpt_pkg.model import GPTConfig, GPT
        compile = True

        n_layer  = cfg.model.gpt_params.n_layer
        n_head   = cfg.model.gpt_params.n_head
        n_embd   = cfg.model.gpt_params.n_embd
        dropout  = cfg.model.gpt_params.dropout # for pretraining 0 is good, for finetuning try 0.1+
        bias     = cfg.model.gpt_params.bias 
                                             
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=meta_vocab_size, dropout=dropout, label_smoothing = 0) 
        
        gpt_cfg = GPTConfig(**model_args)
        model = GPT(gpt_cfg)
    else:
        raise ValueError(f"Invalid model_type. Expected is 'gpt'")
    
    return model, model_args, compile # compile Requires PyTorch 2.0 is for faster training and inference


def load_model(cfg, device, model_dir, block_size, init_from, finetune_out_dir=None, rank=0): 
    """
    Loads a GPT pretrained model checkpoint from model_dir/checkpoint_best.pt.
    Or loads GPT finetuned model checkpoint from finetune_out_dir/all_checkpoints/checkpoint_always.pt.
    Restores RNG states first (via a CPU load), then reloads the full checkpoint
    Returns: model, checkpoint dict, checkpoint_model_args, compile flag, best_val_loss, iter_num
    """
    from src.gpt_pkg.model import GPTConfig, GPT

    if init_from == 'scratch':
        ckpt_path = f'{model_dir}/checkpoint_best.pt'
    elif init_from == 'resume':
        ckpt_path = f'{finetune_out_dir}/all_checkpoints/checkpoint_always.pt'
    else:
        raise ValueError(f"Invalid init_from. Expected one of: 'scratch', 'resume'")

    print('ckpt_path: ', ckpt_path)
    print('file exists:', os.path.exists(ckpt_path))


    compile = True
    model_args = dict(
        n_layer=cfg.model.gpt_params.n_layer,
        n_head=cfg.model.gpt_params.n_head,
        n_embd=cfg.model.gpt_params.n_embd,
        block_size=block_size,
        bias=cfg.model.gpt_params.bias,
        vocab_size=None,
        dropout=cfg.model.gpt_params.dropout
    )

    # Always load checkpoint to CPU
    _cpu_ckpt = torch.load(ckpt_path, map_location='cpu')

    # Restore RNG state for this rank
    if 'rng_states_per_rank' in _cpu_ckpt:
        rng_list = _cpu_ckpt['rng_states_per_rank']
        if rank < len(rng_list):
            set_rng_state_dict(rng_list[rank])
        else:
            print(f"[Rank {rank}] WARNING: No RNG state found for this rank in checkpoint.")
    else:
        print(f"[Rank {rank}] WARNING: No RNG state found in checkpoint. Starting from pretrained?")

    checkpoint = torch.load(ckpt_path, map_location=device)
    print('load okay')

    checkpoint_model_args = checkpoint['model_args']
    print('args okay')

    # Override only essential model args
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    model_args = checkpoint["model_args"].copy()
    model_args.pop("model_type", None)  
    gptconf    = GPTConfig(**model_args)

    model = GPT(gptconf)

    # Fix possible _orig_mod. prefix in state_dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    iter_num = checkpoint.get('iter_num', 0)
    print(f"iter_num is {iter_num}")
    best_val_loss = checkpoint.get('best_val_loss', 1e9)

    best_val_gen_acc   = checkpoint.get('best_val_gen_acc', 0.0)
    no_improve_counter = checkpoint.get('no_improve_counter',  0)

    total_time_elapsed = checkpoint.get('total_time_elapsed', 0.0)


    return model, checkpoint, checkpoint_model_args, compile, best_val_loss, iter_num, best_val_gen_acc, no_improve_counter, total_time_elapsed

def set_rng_state_dict(rng_dict):
    import random
    import numpy as np
    import torch

    random.setstate(rng_dict['py'])
    np.random.set_state(rng_dict['np'])
    torch.set_rng_state(rng_dict['torch_cpu'])
    torch.cuda.set_rng_state_all(rng_dict['torch_cuda'])

def get_rng_state_dict():
    return {
        'py': random.getstate(),
        'np': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all(),
    }

@hydra.main(config_path='hydra/train_config', config_name='train_config.yaml', version_base=None)
def train(cfg: DictConfig) -> None:

    config_dict = wandb_kwargs_via_cfg(cfg)
    print(config_dict)
    
    if wandb.run is not None and wandb.config is not None:
        print("Overriding Hydra config with W&B sweep values")
        sweep_cfg = wandb.config

        # Override scalar hyperparameters
        cfg.train.learning_rate = sweep_cfg.get("train.learning_rate", cfg.train.learning_rate)
        cfg.train.weight_decay  = sweep_cfg.get("train.weight_decay", cfg.train.weight_decay)
        cfg.model.gpt_params.dropout = sweep_cfg.get("model.gpt_params.dropout", cfg.model.gpt_params.dropout)

        # Override from batch_config sweep parameter
        if "batch_config" in sweep_cfg:
            cfg.train.batch_size = sweep_cfg.batch_config["batch_size"]
            cfg.train.gradient_accumulation_steps = sweep_cfg.batch_config["gradient_accumulation_steps"]
            cfg.train.max_iters = sweep_cfg.batch_config["max_iters"]


    if hasattr(cfg, 'additional_tags'):
        additional_tags = list(cfg.additional_tags)
        config_dict['additional_tags'] = additional_tags
    else:
        additional_tags = []

    print(f"Additional tags are {additional_tags}")

    
    # region my_config
    # ---------------------------DIR--------------------------------------------------
    script_dir = os.path.dirname(__file__)
    print("script_dir: ", script_dir)

    repo_path = os.path.dirname(script_dir)
    print("repo_path: ", repo_path)

    data_pkg_dir = os.path.join(script_dir,'data_pkg')
    print("data_pkg_dir: ", data_pkg_dir)

    # ---------------------------GENERAL--------------------------------------------------
    init_from = cfg.general.init_from
    
    experiment  = cfg.experiment
    model_type  = cfg.model.model_type
    
    if init_from=='resume': 
        train_time = cfg.general.train_time
        now_str=train_time
    else: 
        now_str = cfg.general.now_str
        train_time=now_str
            
    # ----------------------------------------data variables----------------------------------------
    block_size          = cfg.data.block_size
    data_type           = cfg.data.data_type
    observation_size    = cfg.data.observation_size
    lower_bound_obs_size = cfg.data.lower_bound_obs_size
    target_type         = cfg.data.target_type
    sequence_type       = cfg.data.sequence_type
    ground_truth_length = cfg.data.ground_truth_length
    group               = f'{data_type}_{sequence_type}_{target_type}_observation_size_{observation_size}_ground_truth_{ground_truth_length}'

    finetune_data_type           = cfg.finetune.finetune_data_type
    finetune_target_type         = cfg.finetune.finetune_target_type
    finetune_ground_truth_length = cfg.finetune.finetune_ground_truth_length
    finetune_sequence_type       = cfg.finetune.finetune_sequence_type
    finetune_experiment          = cfg.experiment 
    finetune_observation_size    = cfg.finetune.finetune_observation_size

    train_seed = getattr(cfg.data, "train_seed", 100)  
    model_dir                          = cfg.finetune.model_dir
        
    finetune_group = f'{finetune_data_type}_{finetune_sequence_type}_{finetune_target_type}_observation_size_{finetune_observation_size}_ground_truth_{finetune_ground_truth_length}'
    data_folder = cfg.finetune.data_folder
    data_dir = data_folder # If data is on ssd or hdd specify full path in configs otherwise use os.path.join(repo_path,'data', data_folder)

    filename_train = cfg.finetune.finetune_filename_train
    filename_val   = cfg.finetune.finetune_filename_val

    train_data = load_data_from_file(os.path.join(data_dir, filename_train))
    val_data   = load_data_from_file(os.path.join(data_dir, filename_val))

    random.seed(train_seed) 
    random.shuffle(train_data)
    random.shuffle(val_data)

    # Subsample training data if ratio is set 
    if hasattr(cfg.data, "ratio") and 0 < cfg.data.ratio < 1.0:
        num_samples = int(len(train_data) * cfg.data.ratio)
        train_data = train_data[:num_samples]  # take first N after shuffling
        print(f"Using only {len(train_data)} samples ({cfg.data.ratio*100:.1f}% of training data)")

    val_len = len(val_data)
    train_len = len(train_data)
    print(f"Validation data is length {val_len}")
    print(f"Train data is length {train_len}")


    # -----------------------------------Early Stopping-----------------------------------

    best_val_gen_acc    = 0.0  
    val_tf_post_acc     = 0.0
    no_improve_counter  = 0
    patience = cfg.train.get("patience", 10**9) # if patient set we do early stopping otherwise not 
        
    # -----------------------------------TRAIN VARIABLES-----------------------------------
    eval_interval = cfg.train.eval_interval # check for new best val loss and save checkpoint
    log_interval  = cfg.train.log_interval # frequency of logging loss
    eval_iters    = cfg.train.eval_iters # for estimating loss over train/val during training (estimate over eval_iters many batches)
    eval_only     = cfg.train.eval_only # if True, script exits right after the first eval

    always_interval = cfg.train.always_interval
    always_save_checkpoint = cfg.train.always_save_checkpoint # if True, always save a checkpoint after each eval

    device = cfg.train.device # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    gradient_accumulation_steps = cfg.train.gradient_accumulation_steps # used to simulate larger batch sizes
    batch_size = cfg.train.batch_size # if gradient_accumulation_steps > 1, this is the micro-batch size

    # adamw optimizer
    learning_rate = cfg.train.learning_rate # max learning rate
    max_iters = cfg.train.max_iters  # total number of training iterations
    steps_per_epoch = math.ceil(train_len / batch_size)

    if max_iters == 0:
        steps_per_epoch = math.ceil(train_len / batch_size)
        max_iters = steps_per_epoch * cfg.train.max_epochs
        cfg.train.max_iters = max_iters  # update config dynamically if not based on max iterations but based on epochs

    print(f"Training size: {train_len}, Batch size: {batch_size}, Steps/epoch: {steps_per_epoch}")
    print(f"Max epochs: {cfg.train.max_epochs}, Max iters: {cfg.train.max_iters}")

    # so it gets logged to wandb
    config_dict['max_iters'] = cfg.train.max_iters
    config_dict['steps_per_epoch'] = steps_per_epoch

    weight_decay = cfg.train.weight_decay
    beta1 = cfg.train.beta1
    beta2 = cfg.train.beta2
    grad_clip = cfg.train.grad_clip # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr = cfg.train.decay_lr # whether to decay the learning rate
    cfg.train.warmup_iters= int(0.05 * max_iters)
    warmup_iters = cfg.train.warmup_iters # how many steps to warm up for
    print(f"warmup_iters is {warmup_iters}")
    min_lr = cfg.train.min_lr # minimum learning rate, should be ~= learning_rate/10 per Chinchilla    
    lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla

    backend = 'nccl' # 'nccl', 'gloo', etc.
    
    # ---------------------------WANDB and OUT DIR--------------------------------------------------
    # wandb logging
    wandb_log = cfg.wandb.wandb_log 
    wandb_project = cfg.wandb.wandb_project

    if init_from == 'scratch':
        wandb_run_name = f'train_run_{model_type}_{train_time}' 
        finetune_wandb_run_name = f'train_run_{model_type}_{now_str}'
        finetune_out_dir = os.path.join(cfg.general.checkpoint_path, 'model_checkpoints', wandb_project, finetune_group, finetune_experiment, finetune_wandb_run_name)

    elif init_from == 'resume':
        wandb_run_name = ''
        finetune_wandb_run_name = f'train_run_{model_type}_{train_time}'
        finetune_out_dir = os.path.join(cfg.general.checkpoint_path, 'model_checkpoints', wandb_project, finetune_group, finetune_experiment, finetune_wandb_run_name)

  
    # region ----------------------------------TRAINING PERPARATION-----------------------------------
    device = cfg.train.device #'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    config_dict['dtype'] = dtype

    ddp = int(os.environ.get('WORLD_SIZE', 1)) > 1 # environment variable Rank only set if its a ddp run 
    print('ddp: ', ddp)

    ddp_rank = 0 

    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size

    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_local_rank = 0 
    
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    
    if master_process:
        os.makedirs(finetune_out_dir, exist_ok=True)
        if always_save_checkpoint:
            finetune_path_all_ckpts = os.path.join(finetune_out_dir, 'all_checkpoints')
            os.makedirs(finetune_path_all_ckpts, exist_ok=True)


    torch.manual_seed(train_seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # attempt to derive vocab_size from the dataset
    # meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_path = os.path.join(data_pkg_dir,f'meta_{sequence_type}.pkl')  
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        config_dict['meta_vocab_size'] = meta_vocab_size
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

        if sequence_type == 'nuc':
            print("encode('ACGT-|:#'): ", encode("ACGT-|:#"))
            print('decode([0,1,2,3,4,5,6,7]): ', decode([0,1,2,3,4,5,6,7]))
        else:
            raise ValueError(f"Invalid sequence_type. Expected one of: 'nuc', 'nuc_extended', 'amino'")
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    if init_from == 'scratch':
        print(f"model_dir is {model_dir}")

        if model_dir is not None and os.path.exists(os.path.join(model_dir, "checkpoint_best.pt")):
            print("Finetuning a model from scratch")
            # load_model(cfg, device, out_dir, block_size, init_from, finetune_flag = False, finetune_out_dir = None):
            model, checkpoint, model_args, compile, best_val_loss, iter_num, best_val_gen_acc, no_improve_counter, _ = load_model(cfg = cfg, device = device, model_dir = model_dir, block_size = block_size, init_from = init_from, finetune_out_dir = finetune_out_dir, rank=ddp_local_rank)
        else: 
            print("Pretraining model from real-world data")
            # Set global seed once at beginning
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # determine the vocab size we'll use for from-scratch training
            model, model_args, compile = init_model(cfg = cfg, meta_vocab_size = meta_vocab_size, block_size = block_size)

        iter_num = 0
        no_improve_counter=0 
        best_val_gen_acc = 0
        best_val_loss = 1e9
        total_time_elapsed = 0.0  
 
    elif init_from == 'resume':
        print("Resuming training from a checkpoint")
        # load_model(cfg, device, out_dir, block_size, init_from, finetune_flag = False, finetune_out_dir = None):
        model, checkpoint, model_args, compile, best_val_loss, iter_num, best_val_gen_acc, no_improve_counter, total_time_elapsed = load_model(cfg = cfg, device = device, model_dir = model_dir, block_size = block_size, init_from = init_from, finetune_out_dir = finetune_out_dir, rank=ddp_local_rank)
        print(f"Loaded non improvement counter from checkpoint equal to {no_improve_counter}")


    config_dict['compile'] = compile
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    # endregion

    # region ----------------------------------- FUNCTIONS -----------------------------------

    def get_batch_with_epoch_logging(*args, return_examples=False):
        """
        Fetch batch from dataset with random subsampling and shuffling after each epoch. 
        """

        split = args[0]
        if split == 'train':
            iter_num = args[1]
            data = train_data
            length = len(data)
            examples_seen = iter_num * batch_size
            index_list = [(examples_seen + i) % length for i in range(batch_size)]
            epoch = examples_seen // length
            is_new_epoch = (examples_seen % length) < batch_size

        elif split == 'val':
            k = args[1]
            data = val_data
            length = len(data)
            index_list = [(k * batch_size + i) % length for i in range(batch_size)]
            epoch = -1  # not applicable for val
            is_new_epoch = False
        else:
            raise ValueError(f"Invalid split. Expected one of: 'train', 'val'")

        data_list = []
        for index in index_list:
            data_ex = data[index]
            obs_str = data_ex.split(':')[0]
            obs_list = obs_str.split('|')

            original_len = len(obs_list)

            # Determine cluster size: sample if larger than max, otherwise use true size
            if original_len > observation_size:
                temp_obs_size = random.randint(lower_bound_obs_size, observation_size)
                subsampled = True
            else:
                temp_obs_size = original_len
                subsampled = False

            # Always shuffle (random permutation) before sampling
            if split == 'train':
                random.shuffle(obs_list)

            # Take the appropriate number of reads
            obs_list = obs_list[:temp_obs_size]

            #print(f"[{split}] Example {index}: original_len={original_len}, "
            #    f"{'subsampled_len=' + str(len(obs_list)) if subsampled else 'used full list'}")

            obs_str = '|'.join(obs_list)
            data_ex = obs_str + ':' + data_ex.split(':')[1]
            data_list.append(data_ex) 


        encoded_data = encode_list(data_list, stoi) 
        padded_encoded_data = pad_encoded_data(encoded_data, block_size, stoi) # pad with # 
        np_data = np.array(padded_encoded_data, dtype=np.int64)
        x_temp = np_data[:, 0:block_size - 1]
        y_temp = np_data[:, 1:block_size]
        x = torch.from_numpy(x_temp.astype(np.int64))
        y = torch.from_numpy(y_temp.astype(np.int64))

        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        if return_examples:
            return x, y, epoch, is_new_epoch, data_list
        else:
            return x, y, epoch, is_new_epoch


    def broadcast_stop_signal(stop: bool, device: str):
        """
        Broadcast an early stopping signal (0 or 1) from rank 0 to all ranks.
        """
        stop_tensor = torch.tensor(int(stop), dtype=torch.uint8, device=device)
        torch.distributed.broadcast(stop_tensor, src=0)
        return bool(stop_tensor.item())


    @torch.no_grad()
    def estimate_loss(iter_num, ddp=False, ddp_rank=0, ddp_world_size=1):
        out = {}
        model.eval()

        for split in ['train', 'val']:
            n_steps         = eval_iters if split == 'train' else math.floor(val_len / batch_size)
            losses          = []
            tf_acc_metric   = MulticlassAccuracy(num_classes=len(stoi)).to(device)
            post_acc_metric = MulticlassAccuracy(num_classes=len(stoi)).to(device)

            for k in range(n_steps):
                # fetch batch
                X, Y, _, _ = get_batch_with_epoch_logging(
                    split,
                    iter_num if split == 'train' else k
                )
                # forward + loss
                with ctx:
                    logits, loss = model(input_ids=X, targets=Y, stoi=stoi)
                losses.append(loss.item())

                # full-sequence accuracy
                preds = logits.argmax(dim=-1).flatten()   # [B*T]
                tf_acc_metric.update(preds, Y.flatten())

                # post-colon accuracy (fixed length)
                # find colon positions in each example
                colon_mask   = (Y == stoi[':'])             # [B, T]
                b_idxs, pos_idxs = colon_mask.nonzero(as_tuple=True)
                for bi, cp in zip(b_idxs.tolist(), pos_idxs.tolist()):
                    start = cp + 1
                    end   = start + ground_truth_length
                    # slice logits and targets
                    slice_preds   = logits[bi, start:end, :].argmax(dim=-1)  # [L]
                    slice_targets = Y[bi, start:end]                         # [L]
                    post_acc_metric.update(slice_preds, slice_targets)

            # aggregate
            out[f"{split}_loss"]        = sum(losses) / len(losses)
            out[f"{split}_tf_acc"]      = tf_acc_metric.compute().item()
            out[f"{split}_tf_post_acc"] = post_acc_metric.compute().item()

        return out


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

    # region ----------------------------------- TRAINING LOOP -----------------------------------
    if master_process:
        json_file_path = os.path.join(finetune_out_dir,  f'training_params_{init_from}_{now_str}.json')
            
        with open(json_file_path, 'w') as f:
            json.dump(config_dict, f, indent = 4)

    if wandb_log and master_process:

        run_id = None
        run_id_path = os.path.join(finetune_out_dir, 'wandb_run_id.txt')
        if init_from == 'resume' and os.path.exists(run_id_path):
            with open(run_id_path, 'r') as f:
                run_id = f.read().strip()

        wandb.init(
            project=wandb_project,
            group=finetune_group,
            tags=[finetune_experiment] + additional_tags,
            name=finetune_wandb_run_name,
            job_type='training',
            config=config_dict,
            dir=finetune_out_dir,
            id=run_id,
            resume="must" if run_id else None
        )

        if init_from == 'scratch':
            run_id = wandb.run.id
            with open(run_id_path, 'w') as f:
                f.write(run_id)


    # training loop
    X, Y, epoch, is_new_epoch = get_batch_with_epoch_logging('train', 0) # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
        
    session_start_time = time.perf_counter()

    last_best_iter = iter_num # for early stopping based on val_loss
    early_stop_patience_iters = getattr(cfg.train, "early_stop_patience_iters", 150000)  # default 150K iterations


    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
 
        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(input_ids = X, targets = Y, stoi = stoi) #model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y, epoch, is_new_epoch = get_batch_with_epoch_logging('train', iter_num)

            # Reshuffle if new epoch
            if is_new_epoch and master_process:
                random.shuffle(train_data)

            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

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
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1
        
        if master_process:
            lock = False

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            metrics = estimate_loss(iter_num, ddp=ddp, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size)

            if master_process: 
                print(
                    f"step {iter_num}: "
                    f"train loss {metrics['train_loss']:.4f}, tf_acc {metrics['train_tf_acc']:.4f}, post_acc {metrics['train_tf_post_acc']:.4f}, "
                    f"val loss {metrics['val_loss']:.4f}, val_tf_acc {metrics['val_tf_acc']:.4f}, val_post_acc {metrics['val_tf_post_acc']:.4f}, "
                )
                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss":               metrics["train_loss"],
                        "train/tf_acc":             metrics["train_tf_acc"],
                        "train/post_colon_acc":     metrics["train_tf_post_acc"],
                        "val/loss":                 metrics["val_loss"],
                        "val/tf_acc":               metrics["val_tf_acc"],
                        "val/post_colon_acc":       metrics["val_tf_post_acc"],
                        "lr":                       lr,
                    })
                    wandb.log({}, commit=True)

            if metrics['val_tf_post_acc'] > val_tf_post_acc:
                best_val_gen_acc   = metrics['val_tf_post_acc']
                no_improve_counter = 0
                if master_process:
                    torch.save({
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num':  iter_num,
                        'best_val_gen_acc': best_val_gen_acc,
                        'no_improve_counter': no_improve_counter,
                        'config': config_dict,
                    }, os.path.join(finetune_out_dir, 'checkpoint_best_gen_acc.pt'))
            else:
                no_improve_counter += 0

            if metrics['val_loss'] < best_val_loss:
                best_val_loss = metrics['val_loss']
                last_best_iter = iter_num
                if iter_num > 0 and master_process:
                    print('NEW VAL LOSS MINIMUM: ', best_val_loss)
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'best_val_gen_acc': best_val_gen_acc,
                        'no_improve_counter': no_improve_counter,
                        'config': config_dict,
                        'wandb_run_id': wandb.run.id if wandb_log else None
                    }
                    print(f"Save checkpoint to {finetune_out_dir}")
                    torch.save(checkpoint, os.path.join(finetune_out_dir, 'checkpoint_best.pt')) 

            if always_save_checkpoint and iter_num % always_interval == 0 and iter_num > 0: 
                local_rng = get_rng_state_dict()
            
                if ddp:
                    dist.barrier()
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
                        'best_val_loss': best_val_loss,
                        'best_val_gen_acc': best_val_gen_acc,
                        'no_improve_counter': no_improve_counter,
                        'config': config_dict,
                        'rng_states_per_rank': all_rng_states,
                        'total_time_elapsed': total_time_elapsed, 

                    }
                    print('ALWAYS SAVE CHECKPOINT')
                    print(f"saving checkpoint to {finetune_path_all_ckpts}")
                    torch.save(checkpoint, os.path.join(finetune_path_all_ckpts, f'checkpoint_always.pt')) 

            # ---------- Distributed early stopping sync ----------
            should_stop = iter_num > warmup_iters and (iter_num - last_best_iter) >= early_stop_patience_iters
            if master_process and should_stop:
                print(f"Early stopping: no improvement in val_gen_acc for {patience} evals.")

            if ddp:
                stop_tensor = torch.tensor(int(should_stop), device=device)
                torch.distributed.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            if should_stop:
                break

        if iter_num == 0 and eval_only:
            break


        # termination conditions
        if iter_num > max_iters:
            break
    
    if master_process:
        # region ------------------------------------ FINISH ------------------------------------
        print("Training finished")
        if wandb_log:
            wandb.log({
                        "iter": iter_num,
                        "best_val_loss": best_val_loss,
                        "wall_clock_hours": total_time_elapsed/3600,
                    })
            
            wandb.finish()

    if ddp:
        destroy_process_group()

    # endregion

@hydra.main(config_path="conf", config_name="test_config", version_base=None)
def test(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
if __name__ == "__main__":

    train()
