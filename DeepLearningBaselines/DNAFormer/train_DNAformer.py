"""
New training script, separate from the helper (prev. utilis) training script, for pretraining the DNAFormer model with IDS dynamic data generation. 

Includes:
- Multi-GPU training via DistributedDataParallel (DDP)
- Weights & Biases (W&B) experiment tracking
- IDS dynamic data generation 
- FLOPs profiling 
- Model compilation with torch.compile() for faster training 
- Mixed precision training using torch.cuda.amp (GradScaler, autocast) for faster training
"""
import os
import json
import time
import math
import random
from datetime import datetime
from collections import deque

import numpy as np
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf
from Levenshtein import distance as levenshtein_distance

import torch
import torch.distributed as dist
import torch.optim as optim
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity
from torchmetrics.classification import MulticlassAccuracy

from data_loader_IDS import DNAformerData, make_loader, collate_dna
import loss         

### Imports from TreconLM base ######
import os
import sys

# Get the directory containing this file
here = os.path.dirname(os.path.abspath(__file__))
print(f"[DEBUG] Script location: {here}")

# Add DNAformer root 
robuseqnet_root = os.path.abspath(os.path.join(here, ".."))
print(f"[DEBUG] Adding to sys.path: {robuseqnet_root}")
sys.path.insert(0, robuseqnet_root)

# Add TReconLM root (where src/ is)
project_root = os.path.abspath(os.path.join(here, "..", ".."))  # Only go up two levels
print(f"[DEBUG] Adding to sys.path: {project_root}")
sys.path.insert(0, project_root)

# Confirm src/utils/hamming_distance.py is there
src_file = os.path.join(project_root, "src", "utils", "hamming_distance.py")
if os.path.exists(src_file):
    print(f"[DEBUG] Found: {src_file}")
else:
    print(f"[WARNING] Not found: {src_file}")

from src.utils.hamming_distance import hamming_distance_postprocessed


# Deterministic validation loader for (single- or multi-GPU) training
def make_fixed_val_loader(config, num_samples: int = 1_000, batch_size: int = 500):
    dist_on = dist.is_available() and dist.is_initialized()
    config.val_dataset_size = num_samples  # needed for DNAformerData in val_fixed mode

    if (not dist_on) or dist.get_rank() == 0:
        # Only rank 0 generates the full fixed dataset
        full_val_set = DNAformerData(config, mode="val_fixed", fixed_seed=42)
        # use entire dataset, optional select subset
        subset_idx = np.arange(len(full_val_set)).tolist()
    else:
        full_val_set = None  # don't generate data on other ranks 
        subset_idx = None

    # Broadcast indices from rank 0 to all ranks
    container = [subset_idx]
    if dist_on:
        dist.broadcast_object_list(container, src=0)
    subset_idx = container[0]

    # Now all ranks create the same dataset
    full_val_set = DNAformerData(config, mode="val_fixed", fixed_seed=42)
    # later could choose exactly which samples (indices) go into the validation set if needed
    subset = torch.utils.data.Subset(full_val_set, subset_idx)

    sampler = DistributedSampler(subset, shuffle=False, drop_last=True) if dist_on else None

    return DataLoader(
        subset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler is None else None,
        drop_last=True,
        pin_memory=True,
        collate_fn=lambda b: collate_dna(b, siamese=(config.model_config == "siamese"))
    )



def save_model(path, model, optimizer, scheduler, iter_num, spent_flops, profiled_flops, approx_flops_per_iter, no_improve_counter, best_val_acc, rank, wandb_run_id=None):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'iter_num': iter_num,
        'spent_flops': spent_flops,
        'profiled_flops': profiled_flops,
        'approx_flops_per_iter': approx_flops_per_iter,
        'no_improve_counter': no_improve_counter,
        'best_val_acc': best_val_acc,
        'rng_states': {
            'py': random.getstate(),
            'np': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all()
        },
        'rank': rank,
        'wandb_run_id': wandb_run_id,
    }, path)


def load_model(path, model, optimizer, scheduler, device, rank):
    ckpt_cpu = torch.load(path, map_location='cpu', weights_only=False)
    ckpt_dev = torch.load(path, map_location=device, weights_only=False)

    # Restore RNG states
    if 'rng_states' in ckpt_cpu:
        print(f"Loading random states.")
        rng = ckpt_cpu['rng_states']
        if 'py' in rng: random.setstate(rng['py'])
        else: print(f"[Rank {rank}] RNG: no 'py' state found.")
        
        if 'np' in rng: np.random.set_state(rng['np'])
        else: print(f"[Rank {rank}] RNG: no 'np' state found.")
        
        if 'torch_cpu' in rng: torch.set_rng_state(rng['torch_cpu'])
        else: print(f"[Rank {rank}] RNG: no 'torch_cpu' state found.")
        
        if 'torch_cuda' in rng:
            loaded_states = rng['torch_cuda']
            num_devices = torch.cuda.device_count()
            if len(loaded_states) != num_devices:
                print(f"[Rank {rank}] WARNING: Checkpoint had {len(loaded_states)} CUDA RNG states, but {num_devices} GPUs are available.")
                loaded_states = loaded_states[:num_devices]
            torch.cuda.set_rng_state_all(loaded_states)
        else:
            print(f"[Rank {rank}] RNG: no 'torch_cuda' state found.")
    else:
        print(f"[Rank {rank}] WARNING: Checkpoint lacks RNG state.")


    # Strip all prefixes and load
    state_dict = ckpt_dev["model_state_dict"]

    # Strip unwanted prefix if present
    unwanted_prefix = "module._orig_mod."
    cleaned_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith(unwanted_prefix):
            cleaned_state_dict[k[len(unwanted_prefix):]] = v
        else:
            cleaned_state_dict[k] = v

    # Now load cleaned state_dict
    model.load_state_dict(cleaned_state_dict)

    optimizer.load_state_dict(ckpt_dev['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt_dev:
        scheduler.load_state_dict(ckpt_dev['scheduler_state_dict'])
    else:
        print(f"[Rank {rank}] WARNING: Checkpoint missing 'scheduler_state_dict' ,  skipping scheduler restore.")


    return (
        ckpt_dev.get('iter_num', 0),
        ckpt_dev.get('spent_flops', 0),
        ckpt_dev.get('profiled_flops', []),
        ckpt_dev.get('approx_flops_per_iter', 0.0), 
        ckpt_dev.get('no_improve_counter', 0.0), 
        ckpt_dev.get('best_val_acc', 0.0),
        ckpt_dev.get('wandb_run_id', None)
    )

def run_train_ddp(config, model):
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Set up distributed GPU device
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    config.device = device

    # Initialize distributed process group (NCCL backend)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    config.local_rank = local_rank
    is_main = (local_rank == 0)  # flag for rank-0 only actions

    # Training state track
    best_val_loss = float("inf")
    best_val_acc = 0.0
    profiled_flops = []     # store early FLOP measurements
    spent_flops = 0.0       # total FLOPs spent
    no_improve_counter = 0
    iter_num = 0

    out_dir = config.out_dir

    # Optimizer & LR scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=config.lrMax, betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            step / (config.max_iters * 0.05)
            if step < (config.max_iters * 0.05)
            else 0.5 * (1 + math.cos(
                math.pi * (step - (config.max_iters * 0.05))
                / (config.max_iters - (config.max_iters * 0.05))
            ))
        ),
        last_epoch=iter_num - 1
    )  

    # Setup run name and checkpoint directory
    if config.init_from == "scratch":
        if is_main:
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"DNAFormer_run_{now_str}_gt{config.label_length}"
            ckpt_dir = os.path.join(out_dir, "model_checkpoints_DNAFormer", run_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            config.checkpoint_dir = ckpt_dir
        else:
            run_name = None
        dist.broadcast_object_list([run_name], src=0)

    elif config.init_from == "resume":
        run_name = f"DNAFormer_run_{config.train_time}_gt{config.label_length}"
        ckpt_dir = os.path.join(out_dir, "model_checkpoints_DNAFormer", run_name)
        config.checkpoint_dir = ckpt_dir
        ckpt_path = os.path.join(ckpt_dir, "checkpoint_always.pt")
        iter_num, spent_flops, profiled_flops, approx_flops, no_improve_counter, best_val_acc, config.wandb_id = \
            load_model(
                ckpt_path, model, optimizer, scheduler, device, local_rank
            )
        profiled_flops = [approx_flops] * 2 if approx_flops else []
        scheduler.last_epoch = iter_num
        if is_main:
            print(f"Resuming from iter {iter_num}")
    else:
        raise ValueError(f"Unknown init_from: {config.init_from}")

    # Move model and optimizer states to GPU and compile
    model = model.to(device)
    model = torch.compile(model)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Initialize Weights & Biases on main process
    if is_main and getattr(config, 'wandb_log', True):
        config_dict = {k: getattr(config, k) for k in dir(config)
                       if not k.startswith('__') and not callable(getattr(config, k))}
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            dir=config.checkpoint_dir,
            config=config_dict,
            id=getattr(config, "wandb_id", None),
            resume="must" if getattr(config, "wandb_id", None) else None
        )
        if not hasattr(config, "wandb_id"):
            config.wandb_id = wandb.run.id

    # Data loaders
    per_gpu_bs = config.batch_size // world_size
    train_loader = make_loader(config, mode="train", batch_size=per_gpu_bs)
    val_loader = make_fixed_val_loader(
        config, num_samples=world_size * per_gpu_bs, batch_size=per_gpu_bs
    )
    train_iter = iter(train_loader)

    # Metrics: cheap proxy (console) vs. ground-truth (W&B)
    train_acc_proxy = MulticlassAccuracy(num_classes=4, average="micro").to(device)
    train_acc_gt    = MulticlassAccuracy(num_classes=4, average="micro").to(device)
    val_acc_metric  = MulticlassAccuracy(num_classes=4, average="micro").to(device)

    proxy_loss = 0.0
    gt_loss    = 0.0

    # Progress bar on main
    pbar = tqdm(
        total=config.max_iters,
        desc="Training",
        initial=iter_num,
        dynamic_ncols=True
    ) if is_main else None

    model.train()
    while iter_num < config.max_iters:
        # Fetch next batch 
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Prepare inputs 
        label = batch["label"].to(device)
        if config.model_config == 'single':
            model_input = batch['model_input'].to(device)
        else:  # "siamese"
            left  = batch['model_input']
            right = batch['model_input_right']
            # inefficiency: duplicated code for siamese in eval; could refactor
            model_input = torch.cat([left, right], dim=0).to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward + backward 
        if len(profiled_flops) < 2:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True
            ) as prof:
                with record_function("ddp_step"):
                    out = model(model_input)
                    loss = config.loss(out, label)["loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            flops = sum(evt.flops for evt in prof.key_averages() if evt.flops)
            profiled_flops.append(flops)
            spent_flops += flops
        else:
            with torch.cuda.amp.autocast():
                out = model(model_input)
                loss = config.loss(out, label)["loss"]
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            spent_flops += np.mean(profiled_flops)

        # LR scheduler 
        scheduler.step()

        # Metrics update 
        preds  = out["pred"].argmax(dim=1)
        target = label.argmax(dim=1)
        train_acc_proxy.update(preds, target)
        train_acc_gt.update(preds, target)
        proxy_loss += loss.item()
        gt_loss    += loss.item()
        iter_num  += 1

        # Cheap proxy logging 
        if iter_num % config.log_interval == 0:
            # global reduce proxy accuracy
            acc_p = torch.tensor(train_acc_proxy.compute().item(), device=device)
            dist.all_reduce(acc_p, op=dist.ReduceOp.SUM)
            acc_p = acc_p.item() / world_size
            if is_main:
                print(f"[iter {iter_num}] loss={proxy_loss/config.log_interval:.4f} acc={acc_p:.4f}")
                pbar.update(config.log_interval)

        train_acc_proxy.reset()
        proxy_loss = 0.0

        # Validation + W&B logging 
        if iter_num % config.eval_interval == 0:
            model.eval()
            val_loss_total = 0.0
            val_acc_metric.reset()
            with torch.no_grad():
                for vb in val_loader:
                    vlabel = vb["label"].to(device)
                    if config.model_config == 'single':
                        vinput = vb['model_input'].to(device)
                    else:
                        l = vb['model_input']; r = vb['model_input_right']
                        vinput = torch.cat([l, r], dim=0).to(device)
                    with torch.cuda.amp.autocast():
                        vout = model(vinput)
                        vloss = config.loss(vout, vlabel)["loss"]
                    val_loss_total += vloss.item()
                    vpreds = vout["pred"].argmax(dim=1)
                    vtarget = vlabel.argmax(dim=1)
                    val_acc_metric.update(vpreds, vtarget)

            # reduce validation metrics across GPUs
            val_loss_t = torch.tensor(val_loss_total, device=device)
            dist.all_reduce(val_loss_t, op=dist.ReduceOp.SUM)
            val_loss_mean = val_loss_t.item() / world_size / len(val_loader)
            val_acc_t = torch.tensor(val_acc_metric.compute().item(), device=device)
            dist.all_reduce(val_acc_t, op=dist.ReduceOp.SUM)
            val_acc_mean = val_acc_t.item() / world_size

            # reduce ground-truth train metrics
            lt = torch.tensor(gt_loss, device=device)
            dist.all_reduce(lt, op=dist.ReduceOp.SUM)
            train_loss_mean = lt.item() / world_size / config.eval_interval
            at = torch.tensor(train_acc_gt.compute().item(), device=device)
            dist.all_reduce(at, op=dist.ReduceOp.SUM)
            train_acc_mean = at.item() / world_size

            if is_main and getattr(config, 'wandb_log', True):
                # log everything
                wandb.log({
                    "iter": iter_num,
                    "train/loss": train_loss_mean,
                    "train/acc": train_acc_mean,
                    "val/loss": val_loss_mean,
                    "val/acc": val_acc_mean,
                    "flops/total_spent": spent_flops,
                    "train/lr": scheduler.get_last_lr()[0],
                })

                # Checkpointing 
                ckpt_always = os.path.join(config.checkpoint_dir, "checkpoint_always.pt")
                save_model(
                    ckpt_always, model, optimizer, scheduler,
                    iter_num, spent_flops, profiled_flops,
                    np.mean(profiled_flops) if profiled_flops else 0.0,
                    no_improve_counter, best_val_acc,
                    local_rank, config.wandb_id
                )
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                    save_model(
                        os.path.join(config.checkpoint_dir, "checkpoint_best_val_loss.pt"),
                        model, optimizer, scheduler, iter_num,
                        spent_flops, profiled_flops,
                        np.mean(profiled_flops) if profiled_flops else 0.0,
                        no_improve_counter, best_val_acc,
                        local_rank, config.wandb_id
                    )
                if val_acc_mean > best_val_acc:
                    best_val_acc = val_acc_mean
                    save_model(
                        os.path.join(config.checkpoint_dir, "checkpoint_best_val_acc.pt"),
                        model, optimizer, scheduler, iter_num,
                        spent_flops, profiled_flops,
                        np.mean(profiled_flops) if profiled_flops else 0.0,
                        no_improve_counter, best_val_acc,
                        local_rank, config.wandb_id
                    )

            # reset ground-truth metrics
            train_acc_gt.reset()
            gt_loss = 0.0

            model.train()

    # Final checkpoint 
    if is_main:
        save_model(
            os.path.join(config.checkpoint_dir, "checkpoint_final.pt"),
            model, optimizer, scheduler,
            iter_num, spent_flops, profiled_flops,
            np.mean(profiled_flops) if profiled_flops else 0.0,
            no_improve_counter, best_val_acc,
            local_rank, config.wandb_id
        )
        if getattr(config, 'wandb_log', True):
            wandb.log({"final/flops": spent_flops})
            wandb.finish()
        pbar.close()
    dist.destroy_process_group()

