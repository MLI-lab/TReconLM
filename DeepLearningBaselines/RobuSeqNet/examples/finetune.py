"""
finetune.py

Extension of https://github.com/qinyunnn/RobuSeqNet/tree/master to support fine-tuning a pretrained model from a checkpoint.
Supports logging to Weights & Biases and tracking validation accuracy on a fixed validation set.

See `slurm_pkg/` for example SLURM scripts to run finetuning.
"""


import sys
import os

# Get the directory containing finetune.py (i.e. ".../RobuSeqNet/examples")
here = os.path.dirname(os.path.abspath(__file__))

# The parent of "examples" is ".../RobuSeqNet"
robuseqnet_root = os.path.abspath(os.path.join(here, ".."))
sys.path.insert(0, robuseqnet_root)

project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))  
sys.path.insert(0, os.path.join(project_root, "src"))

import time
import math
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from torch import optim
from torch.profiler import profile, record_function, ProfilerActivity
from torchmetrics.classification import MulticlassAccuracy
from transformers import get_cosine_schedule_with_warmup
import wandb
from itertools import cycle


from dataset import MyDataset, collater
from Model import Model
from dataset_dynamic import DynamicDataset, encode_reads_list, decode_prediction
from src.utils.hamming_distance import hamming_distance_postprocessed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True, help='Path to train data.')
    parser.add_argument('--val_data', required=True, help='Path to val data.')
    parser.add_argument('--padding_length', type=int, required=True)
    parser.add_argument('--label_length', type=int, required=True)
    parser.add_argument('--base_out_dir', required=True)
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--wandb_project', default='Baselines')
    parser.add_argument('--wandb_entity', default=None)
    parser.add_argument('--max_iter', type=int, default=364780)
    parser.add_argument('--batch_size_all', type=int, default=640)
    parser.add_argument('--max_lr', type=float, default=3e-3)
    parser.add_argument('--pretrain_run_name', type=str, default=None, help='Run name from which resuming finetuning.')
    parser.add_argument('--train_time', type=str, default=None, help='Train time from which finetuning starts.')
    parser.add_argument("--target_type", type=str, default="CPRED")
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=500)
    ########## model params from original work #################
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--lstm_hidden_dim', type=int, default=256)
    parser.add_argument('--conv_dropout_p', type=float, default=0.1)
    parser.add_argument('--rnn_dropout_p', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)

    args = parser.parse_args()

    return args

def convert_flat_to_clustered_format(input_path, reads_out_path, gt_out_path):
    """
    Converts a flat file where each line is: read1|read2|...|readN : ground_truth
    Into:
      - reads_out_path: one read per line, clusters separated by ===============================
      - gt_out_path: one ground truth per line
    """
    Separator = "==============================="

    with open(input_path, "r") as f_in, \
         open(reads_out_path, "w") as f_reads, \
         open(gt_out_path, "w") as f_gt:

        for line in f_in:
            if ":" not in line:
                continue  # skip lines not in correct format

            reads_part, gt_part = line.strip().split(":")
            reads = reads_part.split("|")
            ground_truth = gt_part.strip()

            for read in reads:
                f_reads.write(read.strip() + "\n")
            f_reads.write(Separator + "\n")

            f_gt.write(ground_truth + "\n")

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
            'rng_states_per_rank': {i: rng for i, rng in enumerate(all_rng_states)},
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

def estimate_val_loss(model, criterion, val_loader, device, world_size, label_length):
    model.eval()
    acc_metric = MulticlassAccuracy(num_classes=4, average='micro').to(device)

    total_loss, total_tokens = 0.0, 0
    total_hamming, num_clusters = 0.0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            with torch.amp.autocast(device_type='cuda'):
                logits = model(x)  # [B, 4, L]
                loss = criterion(logits.permute(0, 2, 1).reshape(-1, 4), y.reshape(-1))

            acc_metric.update(logits.argmax(dim=1).view(-1), y.view(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            preds = logits.argmax(dim=1).cpu().numpy()  # [B, L]
            targets = y.cpu().numpy()                   # [B, L]

            for pred_seq, target_seq in zip(preds, targets):
                rec = decode_prediction(pred_seq)[:label_length]
                gt = decode_prediction(target_seq)
                h = hamming_distance_postprocessed(gt, rec)
                total_hamming += h
                num_clusters += 1

    # Average loss across all devices
    local_loss = torch.tensor([total_loss, total_tokens], device=device)
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    avg_loss = local_loss[0].item() / max(local_loss[1].item(), 1)

    # Average accuracy across devices
    local_acc = acc_metric.compute()
    acc_tensor = torch.tensor(local_acc, device=device)
    dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
    avg_acc = acc_tensor.item() / world_size

    # Average Hamming distance across devices
    local_hamming = torch.tensor([total_hamming, num_clusters], device=device)
    dist.all_reduce(local_hamming, op=dist.ReduceOp.SUM)
    avg_hamming = local_hamming[0].item() / max(local_hamming[1].item(), 1)

    model.train()
    return avg_loss, avg_acc, avg_hamming


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

    run_name = f"RobuSeqNet_finet_{args.train_time or time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(args.base_out_dir, run_name)
    if is_main: 
        os.makedirs(out_dir, exist_ok=True)

    wandb_run_id_path = os.path.join(out_dir, "wandb_run_id.txt")
    wandb_run_id = None
    if args.train_time and os.path.exists(wandb_run_id_path):
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
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    print("Loading from a fixed dataset")

    # Preprocess data
    train_base = os.path.splitext(args.train_data)[0]  # removes .txt from args path 
    val_base = os.path.splitext(args.val_data)[0]

    train_reads = train_base + "_reads.txt"
    train_gt    = train_base + "_gt.txt"
    val_reads   = val_base + "_reads.txt"
    val_gt      = val_base + "_gt.txt"

    if is_main:
        convert_flat_to_clustered_format(args.train_data, train_reads, train_gt)
        convert_flat_to_clustered_format(args.val_data, val_reads, val_gt)

    # All ranks wait until rank 0 has finished converting the files
    dist.barrier()

    # Load dataset using converted format
    train_set = MyDataset(train_reads, train_gt)
    val_set   = MyDataset(val_reads, val_gt)

    steps_per_epoch = (len(train_set) + args.batch_size_all - 1) // args.batch_size_all


    val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              collate_fn=collater(args.padding_length, args.label_length))

    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.max_iter * 0.05),
        num_training_steps=args.max_iter
    )

    scaler = torch.cuda.amp.GradScaler()
    profiled_flops, spent_flops, approx_flops = [], 0, 0.0
    best_val_loss, best_val_acc = float('inf'), 0.0

    step = 0

    if args.train_time: 
        run_name = f"RobuSeqNet_finet_{args.train_time}"
        ckpt_path = os.path.join(args.base_out_dir, run_name, "checkpoint_always.pt")

        # Load everything
        step, examples_seen, spent_flops, profiled_flops, approx_flops, no_improve_counter, best_val_acc, wandb_run_id = load_model(ckpt_path, model, optimizer, scheduler, device, local_rank)
        
    else:
        # Starting new finetuning run from pretrained weights
        assert args.pretrain_run_name is not None, "You must set --pretrain_run_name if not resuming training."
        ckpt_path = os.path.join(args.base_out_dir, args.pretrain_run_name, "checkpoint_best_val_loss.pt")
        print(f"[INFO] Loading pretrained weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    for step_ in range(step, args.max_iter):
        
        if step_ >= args.max_iter:
            break

        if step_ == step or (step_ % steps_per_epoch) == 0:
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size_all,
                shuffle=True,
                collate_fn=collater(args.padding_length, args.label_length),
                worker_init_fn=worker_init_fn,
            )
            train_dl = iter(train_loader)


        x, y = next(train_dl)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if step_ < 10 and len(profiled_flops) < 2:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
                with record_function("train_step"):
                    with torch.cuda.amp.autocast():
                        logits = model(x)
                        train_acc_metric.update(logits.argmax(dim=1).view(-1), y.view(-1))
                        loss = criterion(logits.permute(0, 2, 1).reshape(-1, 4), y.reshape(-1))
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
            flops = sum(e.flops for e in prof.key_averages() if e.flops)
            profiled_flops.append(flops)
            approx_flops = np.mean(profiled_flops)
        else:
            with torch.cuda.amp.autocast():
                logits = model(x)
                train_acc_metric.update(logits.argmax(dim=1).view(-1), y.view(-1))
                loss = criterion(logits.permute(0, 2, 1).reshape(-1, 4), y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            flops = approx_flops

        spent_flops += flops

        if args.wandb_log and step_ % args.log_interval == 0:

            acc_local = train_acc_metric.compute()
            acc_tensor = torch.tensor(acc_local, device=device)
            dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
            acc = acc_tensor.item() / world_size
            train_acc_metric.reset()

            if is_main: 
                wandb.log({
                    "train/loss": loss.item(),
                    "train/acc": acc,
                    "train/lr": scheduler.get_last_lr()[0],
                    "flops/total": spent_flops
                }, step=step_)

                print(f"Itr:{step_}/{args.max_iter}, lr:{scheduler.get_last_lr()[0]}, loss: {loss.item():.4f}, accuracy: {acc:.4f}, examples_seen: {step_ * args.batch_size_all}")

        if step_ % args.eval_interval == 0 and step_ > 0:
            val_loss, val_acc, val_ham= estimate_val_loss(model, criterion, val_dl, device, world_size,args.label_length )
                
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
                    "val/ham": val_ham,
                    "val/best_loss": best_val_loss,
                    "val/best_acc": best_val_acc
                })


    if is_main and args.wandb_log:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()