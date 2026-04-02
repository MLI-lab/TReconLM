"""
train_robu_seqnet_ddp.py

Reimplemented RobuSeqNet training script with:
- support for both static and dynamic datasets
- PyTorch Distributed Data Parallel (DDP) for multi-GPU training 
- cosine learning rate decay with warmup
- Weights & Biases (wandb) integration for logging
- automatic checkpointing and validation
- FLOPs profiling and compute tracking

See `slurm_pkg/` for example SLURM scripts to run training.
"""

import os
import sys
import time
import math
import random
import argparse

import numpy as np
import torch

from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist
import torch.multiprocessing as mp

from torchmetrics.classification import MulticlassAccuracy
from transformers import get_cosine_schedule_with_warmup
import wandb


# Get the directory containing finetune.py (i.e. ".../RobuSeqNet/examples")
here = os.path.dirname(os.path.abspath(__file__))

# The parent of "examples" is ".../RobuSeqNet"
robuseqnet_root = os.path.abspath(os.path.join(here, ".."))
sys.path.insert(0, robuseqnet_root)

project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))  
sys.path.insert(0, os.path.join(project_root, "src"))

# Local imports
from dataset import MyDataset, collater
from Model import Model
from dataset_dynamic import DynamicDataset # new


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--ground_truth', required=True)
    parser.add_argument('--dynamic', action='store_true', default=False)
    parser.add_argument('--padding_length', type=int, required=True)
    parser.add_argument('--label_length', type=int, required=True)
    parser.add_argument('--base_out_dir', required=True)
    parser.add_argument('--train_time', type=str, default=None) # to resume from checkpoint
    parser.add_argument('--wandb_project', default='Baselines')
    parser.add_argument("--target_type", type=str, default="CPRED")
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--wandb_entity', default=None)
    parser.add_argument('--max_iter', type=int, default=364780)
    parser.add_argument('--wandb_log', action='store_true')
    ########################## From original paper ####################
    parser.add_argument('--batch_size_all', type=int, default=64)
    parser.add_argument('--max_lr', type=float, default=5e-3) # as in paper, in code have 1e-3
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--lstm_hidden_dim', type=int, default=256)
    parser.add_argument('--conv_dropout_p', type=float, default=0.1)
    parser.add_argument('--rnn_dropout_p', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4) # as in paper, in code have 0
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    return parser.parse_args()

def get_rng_state_dict():
    return {
        'py': random.getstate(),
        'np': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all(),
    }

def save_model(path, model, optimizer, scheduler, batch_counter, examples_seen, spent_flops, profiled_flops, approx_flops_per_iter, no_improve_counter, best_val_acc, rank, wandb_run_id=None):
    local_rng = get_rng_state_dict()

    # Gather RNG states from all ranks to rank 0
    world_size = dist.get_world_size()
    all_rng_states = [None for _ in range(world_size)]
    dist.gather_object(local_rng, object_gather_list=all_rng_states if rank == 0 else None, dst=0)

    if rank == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'batch_counter': batch_counter,
            'examples_seen': examples_seen,
            'spent_flops': spent_flops,
            'profiled_flops': profiled_flops,
            'approx_flops_per_iter': approx_flops_per_iter,
            'no_improve_counter': no_improve_counter,
            'best_val_acc': best_val_acc,
            'rng_states_per_rank': all_rng_states,
            'wandb_run_id': wandb_run_id,
        }, path)



def set_rng_state_dict(rng_dict):
    import random
    import numpy as np
    import torch

    random.setstate(rng_dict['py'])
    np.random.set_state(rng_dict['np'])
    torch.set_rng_state(rng_dict['torch_cpu'])
    torch.cuda.set_rng_state_all(rng_dict['torch_cuda'])

def load_model(path, model, optimizer, scheduler, device, rank):
    def strip_module_prefix(state_dict):
        """Removes 'module.' prefix from keys."""
        return {k.replace("module.", ""): v for k, v in state_dict.items()}

    ckpt_cpu = torch.load(path, map_location='cpu')
    ckpt_dev = torch.load(path, map_location=device)

    # Restore RNG state for this rank
    if 'rng_states_per_rank' in ckpt_cpu:
        if rank in ckpt_cpu['rng_states_per_rank']:
            set_rng_state_dict(ckpt_cpu['rng_states_per_rank'][rank])
        else:
            print(f"[Rank {rank}] WARNING: No RNG state found for this rank in checkpoint.")
    elif 'rng_states' in ckpt_cpu:
        print(f"[Rank {rank}] WARNING: Only one RNG state found, restoring it to all ranks (non-deterministic for DDP).")
        set_rng_state_dict(ckpt_cpu['rng_states'])
    else:
        print(f"[Rank {rank}] WARNING: No RNG state found in checkpoint.")

    # Try to load model state dict
    try:
        try:
            model.load_state_dict(ckpt_dev["model_state_dict"])
        except RuntimeError as e:
            print(f"[Rank {rank}] Retrying after stripping 'module.' prefix due to error: {e}")
            stripped = strip_module_prefix(ckpt_dev["model_state_dict"])
            model.load_state_dict(stripped)
    except Exception as e:
        print(f"[Rank {rank}] Failed to load model state_dict: {e}")
        raise

    optimizer.load_state_dict(ckpt_dev['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt_dev['scheduler_state_dict'])

    return (
        ckpt_dev.get('batch_counter', 0),
        ckpt_dev.get('examples_seen', 0),
        ckpt_dev.get('spent_flops', 0),
        ckpt_dev.get('profiled_flops', []),
        ckpt_dev.get('approx_flops_per_iter', 0.0),
        ckpt_dev.get('no_improve_counter', 0),
        ckpt_dev.get('best_val_acc', 0.0),
        ckpt_dev.get('wandb_run_id', None)
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    return device, local_rank, dist.get_world_size()

def make_fixed_val_loader(padding_length, label_length, batch_size, device, num_samples=1000, seed=2):
    config = {'sub_lb': 0.01, 'sub_ub': 0.1, 'ins_lb': 0.01, 'ins_ub': 0.1,
              'del_lb': 0.01, 'del_ub': 0.1, 'obs_lb': 2, 'obs_ub': 10,
              'gt_length': label_length, 'padding_length': padding_length,
              'label_length': label_length, 'target_type': 'CPRED', 'data_type': 'ids_data'}

    rng = random.Random(seed)
    dataset = DynamicDataset(batch_size=batch_size, device=device, config_params=config, rng=rng)

    val_list = []
    for _ in range(num_samples // batch_size):
        x, y = dataset.get_batch_dynamic(batch_size=batch_size, device=device, config_params=config)
        val_list.append((x.cpu(), y.cpu()))

    class FixedDataset(Dataset):
        def __init__(self, samples): self.samples = samples
        def __len__(self): return len(self.samples)
        def __getitem__(self, i): return self.samples[i][0].to(device), self.samples[i][1].to(device)

    return DataLoader(FixedDataset(val_list), batch_size=None, shuffle=False)

def estimate_val_loss(model, criterion, val_loader, device, world_size):
    model.eval()
    acc_metric = MulticlassAccuracy(num_classes=4, average='micro').to(device)
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for x, y in val_loader:
            with torch.cuda.amp.autocast():
                logits = model(x) # [B, 4, L]
                loss = criterion(logits.permute(0, 2, 1).reshape(-1, 4), y.reshape(-1)) # [B, L, 4], then # [B*L, 4] and y.reshape also to [B*L]  
            acc_metric.update(logits.argmax(dim=1).view(-1), y.view(-1))
            total_loss += loss.item() * y.numel() # same as B * L
            total_samples += x.size(0)  # number of samples in this batch 

    local_loss = torch.tensor([total_loss, total_samples], device=device)
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    avg_loss = local_loss[0].item() / local_loss[1].item()

    local_acc = acc_metric.compute()
    acc_tensor = local_acc.clone().detach().to(device)
    dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
    avg_acc = acc_tensor.item() / world_size

    model.train()

    return avg_loss, avg_acc


def main():
    args = get_args()
    device, local_rank, world_size = setup_ddp()
    is_main = local_rank == 0
    set_seed(100 + local_rank)
    train_acc_metric = MulticlassAccuracy(num_classes=4, average='micro').to(device)


    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    batch_size = args.batch_size_all // world_size 

    run_name = f"RobuSeqNet_{args.train_time or time.strftime('%Y%m%d_%H%M%S')}_gt{args.label_length}"
    out_dir = os.path.join(args.base_out_dir, run_name)
    if is_main: 
        os.makedirs(out_dir, exist_ok=True)

    wandb_run_id_path = os.path.join(out_dir, "wandb_run_id.txt")
    wandb_run_id = None
    if os.path.exists(wandb_run_id_path):
        with open(wandb_run_id_path) as f:
            wandb_run_id = f.read().strip()

    if args.wandb_log and is_main:
        wandb.init(
            name=run_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            dir=out_dir,
            id=wandb_run_id,
            resume="must" if wandb_run_id else None
        )
        if wandb_run_id is None:
            with open(wandb_run_id_path, "w") as f:
                f.write(wandb.run.id)
            wandb_run_id = wandb.run.id  

    model = Model(
        noise_length=args.padding_length,
        label_length=args.label_length,
        dim=args.dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        conv_dropout_p=args.conv_dropout_p,
        rnn_dropout_p=args.rnn_dropout_p
    ).to(device)
    
    # self.Kc in AttnScore is unused in the original implementation,
    # so we set find_unused_parameters=True in DDP to avoid errors.
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    
    #Debugging
    #for idx, (name, param) in enumerate(model.named_parameters()):
    #    print(f"[index={idx}] {name}")

    if args.dynamic:
        print("Generating dynamic dataset on the fly")
        config = {
            'sub_lb': 0.01, 'sub_ub': 0.1,
            'ins_lb': 0.01, 'ins_ub': 0.1,
            'del_lb': 0.01, 'del_ub': 0.1,
            'obs_lb': 2, 'obs_ub': 10,
            'gt_length': args.label_length,
            'padding_length': args.padding_length,
            'label_length': args.label_length,
            'target_type': 'CPRED',
            'data_type': 'ids_data'
        }
        train_set = DynamicDataset(batch_size=batch_size, device=device, config_params=config)
        train_dl = DataLoader(train_set, batch_size=None, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    else:
        print("Loading from a fixed dataset")
        train_set = MyDataset(args.train_data, args.ground_truth)
        train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                              collate_fn=collater(args.padding_length, args.label_length))

    val_dl = make_fixed_val_loader(args.padding_length, args.label_length, args.batch_size_all // world_size, device)

    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.max_iter * 0.05),
        num_training_steps=args.max_iter
    )

    scaler = torch.amp.GradScaler()
    profiled_flops, spent_flops, approx_flops = [], 0, 0.0
    best_val_loss, best_val_acc = float('inf'), 0.0

    # Timing and throughput tracking
    iter_times = []  # Track iteration times for averaging

    step = 0
    if args.train_time:
        run_name = f"RobSeqNet_run_{args.train_time}_gt{args.label_length}"
        ckpt_path = os.path.join(args.base_out_dir, run_name, "checkpoint_always.pt")

        # Load everything
        step, examples_seen, spent_flops, profiled_flops, approx_flops, no_improve_counter, best_val_acc, wandb_run_id = \
            load_model(ckpt_path, model, optimizer, scheduler, device, local_rank)

        profiled_flops = [approx_flops] * 2  


    for step_, (x, y) in enumerate(train_dl, start=step):
        if step_ >= args.max_iter:
            break

        # Start timing the iteration
        torch.cuda.synchronize()  # Ensure all previous ops are done
        t0 = time.time()

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if step_ < 10 and len(profiled_flops) < 2:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
                with record_function("train_step"):
                    with torch.amp.autocast(device_type='cuda'):
                        logits = model(x)
                        train_acc_metric.update(logits.argmax(dim=1).view(-1), y.view(-1))
                        loss = criterion(logits.permute(0, 2, 1).reshape(-1, 4), y.reshape(-1))
                    scaler.scale(loss).backward()


                    # Debugging
                    #for name, param in model.named_parameters():
                    #    if param.grad is None:
                    #        print(f"[RANK {local_rank}] Unused parameter (no grad): {name}")

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
            flops = sum(e.flops for e in prof.key_averages() if e.flops)
            profiled_flops.append(flops)
            approx_flops = np.mean(profiled_flops)
        else:
            with torch.amp.autocast(device_type='cuda'):
                logits = model(x)
                train_acc_metric.update(logits.argmax(dim=1).view(-1), y.view(-1))
                loss = criterion(logits.permute(0, 2, 1).reshape(-1, 4), y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            flops = approx_flops

        # End timing the iteration
        torch.cuda.synchronize()  # Ensure all GPU ops are done
        t1 = time.time()
        iter_time = t1 - t0  # seconds per iteration

        # Track timing (skip first few warmup iterations)
        if step_ >= 10:
            iter_times.append(iter_time)
            # Keep only last 100 iterations for moving average
            if len(iter_times) > 100:
                iter_times.pop(0)

        spent_flops += flops

        # Calculate throughput and time estimates
        avg_iter_time = np.mean(iter_times) if len(iter_times) > 0 else iter_time
        if avg_iter_time > 0 and approx_flops > 0:
            # Single GPU throughput
            single_gpu_throughput = approx_flops / avg_iter_time  # FLOPs/sec on one GPU
            # Multi-GPU throughput (ideal scaling assumption)
            multi_gpu_throughput = single_gpu_throughput * world_size  # Total FLOPs/sec

            # Time estimates to 10^20 FLOPs
            target_flops = 1e20
            time_to_1e20_single_gpu_sec = target_flops / single_gpu_throughput
            time_to_1e20_multi_gpu_sec = target_flops / multi_gpu_throughput

            # Convert to hours and days
            time_to_1e20_single_gpu_hours = time_to_1e20_single_gpu_sec / 3600
            time_to_1e20_single_gpu_days = time_to_1e20_single_gpu_hours / 24
            time_to_1e20_multi_gpu_hours = time_to_1e20_multi_gpu_sec / 3600
            time_to_1e20_multi_gpu_days = time_to_1e20_multi_gpu_hours / 24
        else:
            single_gpu_throughput = 0
            multi_gpu_throughput = 0
            time_to_1e20_single_gpu_hours = 0
            time_to_1e20_single_gpu_days = 0
            time_to_1e20_multi_gpu_hours = 0
            time_to_1e20_multi_gpu_days = 0

        if args.wandb_log and step_ % args.log_interval == 0:

            acc_local = train_acc_metric.compute()
            acc_tensor = acc_local.clone().detach().to(device)
            dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
            acc = acc_tensor.item() / world_size
            train_acc_metric.reset()

            if is_main:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/acc": acc,
                    "train/lr": scheduler.get_last_lr()[0],
                    "flops/total": spent_flops,
                    "flops/per_iter": approx_flops,
                    "throughput/iter_time_ms": avg_iter_time * 1000,
                    "throughput/single_gpu_tflops_per_sec": single_gpu_throughput / 1e12,
                    "throughput/multi_gpu_tflops_per_sec": multi_gpu_throughput / 1e12,
                    "estimate/time_to_1e20_single_gpu_days": time_to_1e20_single_gpu_days,
                    "estimate/time_to_1e20_multi_gpu_days": time_to_1e20_multi_gpu_days,
                }, step=step_)

                print(f"Itr:{step_}/{args.max_iter}, lr:{scheduler.get_last_lr()[0]:.6f}, loss:{loss.item():.4f}, acc:{acc:.4f}, "
                      f"iter_time:{avg_iter_time*1000:.1f}ms, throughput:{multi_gpu_throughput/1e12:.2f}TFLOPs/s ({world_size}GPUs), "
                      f"est_time_to_1e20:{time_to_1e20_multi_gpu_days:.1f}days")

        if step_ % args.eval_interval == 0:
            val_loss, val_acc = estimate_val_loss(model, criterion, val_dl, device, world_size)
            
            improved_val_loss = val_loss < best_val_loss
            improved_val_acc = val_acc > best_val_acc

            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_acc)

            # Save always checkpoint
            ckpt_always = os.path.join(out_dir, "checkpoint_always.pt")
            save_model(
                ckpt_always, model, optimizer, scheduler,
                batch_counter=step_,
                examples_seen=step_ * args.batch_size_all,
                spent_flops=spent_flops,
                profiled_flops=profiled_flops,
                approx_flops_per_iter=approx_flops,
                no_improve_counter=0,  
                best_val_acc=best_val_acc,
                rank=local_rank,
            )

            # Save best val loss checkpoint
            if improved_val_loss:
                ckpt_val_loss = os.path.join(out_dir, "checkpoint_best_val_loss.pt")
                save_model(
                    ckpt_val_loss, model, optimizer, scheduler,
                    batch_counter=step_,
                    examples_seen=step_ * args.batch_size_all,
                    spent_flops=spent_flops,
                    profiled_flops=profiled_flops,
                    approx_flops_per_iter=approx_flops,
                    no_improve_counter=0,
                    best_val_acc=best_val_acc,
                    rank=local_rank,
                )

            # Save best val acc checkpoint 
            if improved_val_acc:
                ckpt_val_acc = os.path.join(out_dir, "checkpoint_best_val_acc.pt")
                save_model(
                    ckpt_val_acc, model, optimizer, scheduler,
                    batch_counter=step_,
                    examples_seen=step_ * args.batch_size_all,
                    spent_flops=spent_flops,
                    profiled_flops=profiled_flops,
                    approx_flops_per_iter=approx_flops,
                    no_improve_counter=0,
                    best_val_acc=best_val_acc,
                    rank=local_rank,
                )

            # Log to Weights & Biases
            if args.wandb_log and is_main:
                wandb.log({
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/best_loss": best_val_loss,
                    "val/best_acc": best_val_acc
                })


    if is_main and args.wandb_log:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()