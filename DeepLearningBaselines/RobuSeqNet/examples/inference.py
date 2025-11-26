"""
inference.py

Extension of https://github.com/qinyunnn/RobuSeqNet to support inference using a trained RobuSeqNet model.

Supports:
- Logging inference results to Weights & Biases
- Downloading test datasets from W&B artifacts
- Computing Hamming and Levenshtein distances
- Aggregated statistics per cluster size

See `slurm_pkg/` for example SLURM scripts to run inference.
"""
import sys
import os
import time

# Get the directory containing this file
here = os.path.dirname(os.path.abspath(__file__))
print(f"[DEBUG] Script location: {here}")

# Add RobuSeqNet root (where dataset_dynamic.py, Model.py is)
robuseqnet_root = os.path.abspath(os.path.join(here, ".."))
print(f"[DEBUG] Adding to sys.path: {robuseqnet_root}")
sys.path.insert(0, robuseqnet_root)

# Add TReconLM root (where src/ is)
project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
print(f"[DEBUG] Adding to sys.path: {project_root}")
sys.path.insert(0, project_root)

src_file = os.path.join(project_root, "src", "utils", "hamming_distance.py")
if os.path.exists(src_file):
    print(f"[DEBUG] Found: {src_file}")
else:
    print(f"[WARNING] Not found: {src_file}")

import argparse
import torch
import numpy as np
from collections import defaultdict
from Levenshtein import distance as levenshtein_distance
from src.utils.hamming_distance import hamming_distance_postprocessed
from src.utils.helper_functions import contaminate_trace_cluster
from dataset_dynamic import encode_reads_list, decode_prediction
from Model import Model
import wandb
from dataset import MyDataset, collater
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace
from tqdm import tqdm


def load_artifact_reads_and_ground_truth(entity, test_project, artifact_name, local_data_dir=None):
    """
    Load data from wandb artifact or local directory.

    Args:
        entity: WandB entity name
        test_project: WandB test project name
        artifact_name: Artifact name to download
        local_data_dir: If provided, use this local directory instead of downloading

    Returns:
        Tuple of (reads, ground_truths)
    """
    if local_data_dir:
        print(f"Loading data from local directory: {local_data_dir}")
        data_dir = local_data_dir

        # Verify required files exist
        reads_path = os.path.join(data_dir, "reads.txt")
        gt_path = os.path.join(data_dir, "ground_truth.txt")

        if not os.path.exists(reads_path):
            raise FileNotFoundError(f"reads.txt not found in {local_data_dir}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"ground_truth.txt not found in {local_data_dir}")
    else:
        # Download from wandb
        artifact = wandb.use_artifact(
            f"{entity}/{test_project}/{artifact_name}:latest",
            type="dataset"
        )
        data_dir = artifact.download()

    with open(os.path.join(data_dir, "reads.txt")) as f:
        reads = [line.strip() for line in f]
    with open(os.path.join(data_dir, "ground_truth.txt")) as f:
        gts = [line.strip() for line in f]
    return reads, gts


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


def group_clusters(reads):
    clusters = []
    cur = []
    for line in reads:
        if line.startswith("="):
            if cur:
                clusters.append(cur)
            cur = []
        else:
            cur.append(line)
    if cur:
        clusters.append(cur)
    return clusters


class ClustersDataset(Dataset):
    def __init__(self, clusters, gts):
        self.clusters = clusters
        self.gts      = gts

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, idx):
        return self.clusters[idx], self.gts[idx]


def inference_collate(batch):
    # batch: list of (cluster: List[str], gt: str)
    clusters_batch, gt_batch = zip(*batch)
    N_batch = [len(c) for c in clusters_batch]
    max_reads = max(N_batch)
    # encode_reads_list takes List[List[str]] and returns [B, max_reads, L, 4]
    X = encode_reads_list(clusters_batch,
                          padding_length,
                          max_num_reads=max_reads).float()
    return X, gt_batch, N_batch


def run_timing_measurement(args, clusters, gts, device, model):
    """
    Run throughput measurement by cycling through dataset for fixed time windows.

    Measures pure model inference throughput by repeatedly processing examples
    for a fixed duration, then reporting examples/hour with statistics.

    Args:
        args: Parsed command-line arguments
        clusters: List of clusters (each cluster is a list of DNA read strings)
        gts: List of ground truth strings
        device: torch.device
        model: Loaded RobuSeqNet model
    """
    from itertools import cycle

    run_duration = args.timing_duration
    num_runs = args.timing_runs
    warmup_runs = args.timing_warmup
    batch_size = args.batch_size

    print(f"\n{'='*80}")
    print("TIMING MODE: THROUGHPUT MEASUREMENT")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Model: RobuSeqNet")
    print(f"  Run duration: {run_duration / 60:.1f} minutes ({run_duration}s)")
    print(f"  Number of runs: {num_runs} ({warmup_runs} warmup + {num_runs - warmup_runs} measured)")
    print(f"  Batch size: {batch_size}")
    print(f"  Dataset size: {len(clusters)} examples")
    print(f"  Device: {device}")
    print(f"")

    # Check actual input lengths and sort by total concatenated length (consistent with TReconLM)
    def get_total_input_length(cluster):
        """Get total length of all reads concatenated in cluster"""
        return sum(len(read) for read in cluster)

    all_lengths = [get_total_input_length(cluster) for cluster in clusters]
    min_length = min(all_lengths)
    max_length = max(all_lengths)
    avg_length = sum(all_lengths) / len(all_lengths)

    print(f"Total concatenated input lengths: min={min_length}, max={max_length}, avg={avg_length:.1f}")

    # Create representative subset for timing
    # Sample evenly across observation_size (cluster size) for realistic throughput
    print(f"\nCreating representative timing subset:")

    import random
    from collections import defaultdict
    timing_seed = getattr(args, 'timing_seed', 42)
    rng = random.Random(timing_seed)

    # Group by observation_size (cluster size = number of reads)
    by_obs_size = defaultdict(list)
    for idx, (cluster, gt) in enumerate(zip(clusters, gts)):
        obs_size = len(cluster)  # cluster_size = observation_size
        by_obs_size[obs_size].append((idx, cluster, gt))

    print(f"  Found observation sizes: {sorted(by_obs_size.keys())}")
    for obs in sorted(by_obs_size.keys()):
        print(f"    Obs size {obs}: {len(by_obs_size[obs])} examples")

    # Sample equal number from each observation_size
    samples_per_obs = getattr(args, 'timing_samples_per_obs', batch_size)  # Default to batch_size for clean division
    timing_subset = []

    for obs in sorted(by_obs_size.keys()):
        items = by_obs_size[obs]
        if len(items) <= samples_per_obs:
            # Take all if fewer than requested
            sampled = items
        else:
            # Randomly sample
            sampled = rng.sample(items, samples_per_obs)

        timing_subset.extend(sampled)
        lengths = [get_total_input_length(item[1]) for item in sampled]
        print(f"  Sampled {len(sampled)} from obs_size={obs}, total lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

    print(f"  Total timing subset: {len(timing_subset)} examples")

    # Sort subset by total input length for efficient batching
    timing_subset.sort(key=lambda item: get_total_input_length(item[1]))
    print(f"  Sorted by length for efficient batching")
    print(f"  This subset will be cycled for all timing runs (equal representation of all obs sizes)")
    print(f"")

    # Create initial cycle (will be reset for each measured run)
    data_cycle = cycle((item[1], item[2]) for item in timing_subset)

    # Storage for results
    all_throughputs = []
    all_example_counts = []
    all_durations = []

    # Run timing windows
    for run_idx in range(num_runs):
        is_warmup = run_idx < warmup_runs
        run_label = "WARMUP" if is_warmup else f"RUN {run_idx - warmup_runs + 1}"

        # Reset cycle at start of each measured run (not warmup)
        # This ensures all measured runs see the same data for consistent results
        if not is_warmup:
            data_cycle = cycle((item[1], item[2]) for item in timing_subset)

        print(f"\n{run_label}: Starting {run_duration / 60:.1f} minute timing window...")

        run_start = time.perf_counter()
        examples_this_run = 0

        # Process examples until time limit
        while True:
            elapsed = time.perf_counter() - run_start
            if elapsed >= run_duration:
                break

            # Collect batch from cycle
            batch_clusters = []
            batch_gts = []
            for _ in range(batch_size):
                cluster, gt = next(data_cycle)
                batch_clusters.append(cluster)
                batch_gts.append(gt)

            # Encode batch (same as collate function)
            max_reads = max(len(c) for c in batch_clusters)
            X_batch = encode_reads_list(batch_clusters, args.padding_length, max_num_reads=max_reads).float()
            X_batch = X_batch.to(device)

            # Run inference with timing
            if device.type == 'cuda':
                torch.cuda.synchronize()

            with torch.no_grad():
                t0 = time.perf_counter()
                logits = model(X_batch)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            t1 = time.perf_counter()

            examples_this_run += batch_size

            # Log progress every 100 examples
            if examples_this_run % 1000 == 0:
                current_elapsed = time.perf_counter() - run_start
                current_rate = (examples_this_run / current_elapsed) * 3600
                print(f"  [{run_label}] Progress: {examples_this_run} examples in {current_elapsed:.1f}s -> {current_rate:.0f} ex/hr (current)")

        # Compute final timing for this run
        run_end = time.perf_counter()
        run_elapsed = run_end - run_start
        throughput = (examples_this_run / run_elapsed) * 3600  # examples per hour

        print(f"  [{run_label}] Completed: {examples_this_run} examples in {run_elapsed:.1f}s -> {throughput:.0f} ex/hr")

        # Store results (skip warmup)
        if not is_warmup:
            all_throughputs.append(throughput)
            all_example_counts.append(examples_this_run)
            all_durations.append(run_elapsed)

    # Compute statistics
    mean_throughput = np.mean(all_throughputs)
    std_throughput = np.std(all_throughputs)
    cv_throughput = (std_throughput / mean_throughput) * 100  # CV as percentage

    total_examples = sum(all_example_counts)
    total_time = sum(all_durations)

    print(f"\n{'='*80}")
    print("TIMING RESULTS")
    print(f"{'='*80}")
    print(f"Measured runs: {len(all_throughputs)}")
    print(f"Individual throughputs: {[f'{t:.0f}' for t in all_throughputs]} ex/hr")
    print(f"")
    print(f"Mean throughput: {mean_throughput:.0f} ex/hr")
    print(f"Std throughput: {std_throughput:.0f} ex/hr")
    print(f"Coefficient of Variation (CV): {cv_throughput:.2f}%")
    print(f"")
    print(f"Total examples processed: {total_examples}")
    print(f"Total time (measured runs): {total_time / 60:.1f} minutes")
    print(f"{'='*80}")

    # Log to WandB
    wandb.log({
        'timing_mean_throughput_per_hour': mean_throughput,
        'timing_std_throughput_per_hour': std_throughput,
        'timing_cv_throughput_percent': cv_throughput,
        'timing_total_examples': total_examples,
        'timing_total_time_minutes': total_time / 60,
        'timing_num_measured_runs': len(all_throughputs),
    })

    # Log individual run throughputs
    for i, throughput in enumerate(all_throughputs):
        wandb.log({f'timing_run_{i+1}_throughput': throughput})


def run_inference(args):
    """
    Batched inference for RobuSeqNet with optional sweep functionality.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # W&B login
    wandb.login()

    # determine sweep indices
    ks = list(range(args.num_sweep_runs)) if args.sweep else [None]

    for k in ks:
        # build per-run artifact name and run name
        if k is not None:
            seed = args.test_seed + k
            artifact_name = (
                f"sweep{k}_seed{seed}"
                f"_gl110"
                f"_bs1500"
                f"_ds5000"
            )
            run_name      = f"Robseqnet_L{args.label_length}_inference_sweep_k{k}"
        else:
            artifact_name = args.artifact_name
            run_name      = f"Robseqnet_L{args.label_length}_inference"

        # start a new W&B run
        run = wandb.init(
            project = args.project,
            entity  = args.wandb_entity,
            name    = run_name,
            job_type= "inference",
            config  = vars(args)
        )
        if k is not None:
            wandb.config.update({"sweep_index": k}, allow_val_change=True)

        # load and cluster the reads from the specified artifact or local directory
        reads, gts = load_artifact_reads_and_ground_truth(
            args.wandb_entity,
            args.test_project,
            artifact_name,
            local_data_dir=args.local_data_dir
        )
        clusters = group_clusters(reads)

        # Optional: Sample a subset of examples for faster testing
        if args.max_samples is not None and args.max_samples < len(clusters):
            print(f"\n{'='*80}")
            print(f"SAMPLING SUBSET OF DATA")
            print(f"{'='*80}")
            print(f"Total examples available: {len(clusters)}")
            print(f"Max samples requested: {args.max_samples}")
            print(f"Random seed: {args.sampling_seed}")

            rng = np.random.RandomState(args.sampling_seed)
            sampled_indices = rng.choice(len(clusters), size=args.max_samples, replace=False)
            sampled_indices = sorted(sampled_indices)  # Sort for deterministic processing order

            clusters = [clusters[i] for i in sampled_indices]
            gts = [gts[i] for i in sampled_indices]
            print(f"Sampled {len(clusters)} examples for inference")
            print(f"{'='*80}\n")

        # Check if timing mode is enabled
        if args.timing:
            # Update wandb run name to indicate timing mode
            wandb.run.name = f"{run_name}_timing"

            # Load model for timing
            model = Model(
                noise_length=args.padding_length,
                label_length=args.label_length,
                dim=args.dim,
                lstm_hidden_dim=args.lstm_hidden_dim,
                conv_dropout_p=args.conv_dropout_p,
                rnn_dropout_p=args.rnn_dropout_p
            ).to(device)

            ckpt = torch.load(args.checkpoint, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            state_dict = {
                k.replace("module.", "").replace("_orig_mod.", ""): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict)
            model.eval()

            # Run timing measurement
            run_timing_measurement(args, clusters, gts, device, model)

            # Finish WandB and exit
            wandb.finish()
            return  # Exit early after timing measurement

        # Handle misclustering experiment if enabled
        if args.misclustering:
            # Parse contamination rates
            contamination_rates = [float(r.strip()) for r in args.contamination_rates.split(',')]
            print(f"Misclustering experiment enabled with rates: {contamination_rates}")

            # Create simple config object for contaminate_trace_cluster
            # Use the SAME error rates as training (from train_robu_seqnet_ddp.py lines 186-187)
            # RobuSeqNet trains with error rates sampled uniformly from [0.01, 0.1]
            # Provide BOTH lb and ub - contaminate_trace_cluster will sample uniformly from [lb, ub]
            cfg = SimpleNamespace()
            cfg.data = SimpleNamespace()
            cfg.data.insertion_probability_lb = 0.01
            cfg.data.insertion_probability_ub = 0.1
            cfg.data.deletion_probability_lb = 0.01
            cfg.data.deletion_probability_ub = 0.1
            cfg.data.substitution_probability_lb = 0.01
            cfg.data.substitution_probability_ub = 0.1

            print(f"  Error rate sampling ranges (sampled uniformly per contaminant):")
            print(f"    INS=[{cfg.data.insertion_probability_lb:.3f}, {cfg.data.insertion_probability_ub:.3f}], "
                  f"DEL=[{cfg.data.deletion_probability_lb:.3f}, {cfg.data.deletion_probability_ub:.3f}], "
                  f"SUB=[{cfg.data.substitution_probability_lb:.3f}, {cfg.data.substitution_probability_ub:.3f}]")

            # Set random seed for reproducibility
            rng = np.random.RandomState(args.seed)

            # Load model once (outside contamination loop)
            model = Model(
                noise_length=args.padding_length,
                label_length=args.label_length,
                dim=args.dim,
                lstm_hidden_dim=args.lstm_hidden_dim,
                conv_dropout_p=args.conv_dropout_p,
                rnn_dropout_p=args.rnn_dropout_p
            ).to(device)

            ckpt = torch.load(args.checkpoint, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            state_dict = {
                k.replace("module.", "").replace("_orig_mod.", ""): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict)
            model.eval()

            # Run contamination experiment for each rate
            for cont_rate in contamination_rates:
                print(f"\nProcessing contamination rate: {cont_rate}")

                # Contaminate all clusters
                contaminated_clusters = []
                contaminated_examples = []
                total_contaminated_traces = 0

                print("  Contaminating clusters...")
                for i, (cluster, gt) in enumerate(tqdm(zip(clusters, gts), total=len(clusters), desc="  Contaminating")):
                    # Contaminate the cluster
                    contaminated_cluster, contamination_info = contaminate_trace_cluster(
                        traces=cluster,
                        ground_truth=gt,
                        contamination_rate=cont_rate,
                        baseline_error_rate=0.055,
                        cfg=cfg,
                        rng=rng
                    )

                    contaminated_clusters.append(contaminated_cluster)

                    # Track contamination
                    num_contaminated = len(contamination_info['contaminated_positions'])
                    if num_contaminated > 0:
                        contaminated_examples.append(i)
                        total_contaminated_traces += num_contaminated

                print(f"  Contaminated {len(contaminated_examples)} examples (out of {len(clusters)})")
                print(f"  Total contaminated traces: {total_contaminated_traces}")
                print(f"  Average contaminated traces per example: {total_contaminated_traces / len(clusters):.2f}")

                # Create DataLoader for contaminated clusters
                ds_cont = ClustersDataset(contaminated_clusters, gts)
                loader_cont = DataLoader(
                    ds_cont,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=lambda batch: (
                        encode_reads_list(
                            [c for c, _ in batch],
                            args.padding_length,
                            max_num_reads=max(len(c) for c, _ in batch)
                        ).float(),
                        [gt for _, gt in batch],
                        [len(c) for c, _ in batch]
                    ),
                    num_workers=0,
                )

                # Run inference on contaminated data
                stats = defaultdict(lambda: {"hamming": [], "lev": [], "correct": 0, "correct_chars": 0, "total_chars": 0})
                inf_times = []

                print("  Running inference...")
                for X_batch, gt_batch, N_batch in tqdm(loader_cont, desc="  Inference", total=len(loader_cont)):
                    X_batch = X_batch.to(device)

                    # Synchronize GPU before timing
                    if device.type == 'cuda':
                        torch.cuda.synchronize()

                    with torch.no_grad():
                        t0 = time.perf_counter()
                        logits = model(X_batch)

                    # Synchronize GPU after inference
                    if device.type == 'cuda':
                        torch.cuda.synchronize()

                    t1 = time.perf_counter()

                    per_ex = (t1 - t0) / X_batch.size(0)
                    inf_times += [per_ex] * X_batch.size(0)

                    preds = logits.argmax(dim=1).cpu().numpy()
                    recs = [decode_prediction(p)[:args.label_length] for p in preds]

                    for rec, gt, N in zip(recs, gt_batch, N_batch):
                        correct_chars = sum(p == t for p, t in zip(rec, gt))
                        stats[N]["correct_chars"] += correct_chars
                        stats[N]["total_chars"] += len(gt)

                        h = hamming_distance_postprocessed(gt, rec)
                        l = levenshtein_distance(gt, rec) / len(gt)
                        stats[N]["hamming"].append(h)
                        stats[N]["lev"].append(l)
                        if h == 0:
                            stats[N]["correct"] += 1

                # Log metrics for this contamination rate
                condition_name = f"cont_{cont_rate:.3f}"
                print(f"\n{condition_name} Stats:")
                for N in sorted(stats):
                    h_arr = np.array(stats[N]["hamming"])
                    l_arr = np.array(stats[N]["lev"])
                    success = stats[N]["correct"] / len(h_arr)
                    accuracy = stats[N]["correct_chars"] / stats[N]["total_chars"]
                    print(
                        f"  N={N} | Success: {success:.3f} | Acc: {accuracy:.3f} | "
                        f"H: {h_arr.mean():.2f}±{h_arr.std():.2f} | "
                        f"L: {l_arr.mean():.2f}±{l_arr.std():.2f}"
                    )
                    wandb.log({
                        f"{condition_name}_avg_hamming_N={N}": h_arr.mean(),
                        f"{condition_name}_std_hamming_N={N}": h_arr.std(),
                        f"{condition_name}_avg_levenshtein_N={N}": l_arr.mean(),
                        f"{condition_name}_std_levenshtein_N={N}": l_arr.std(),
                        f"{condition_name}_success_rate_N={N}": success,
                        f"{condition_name}_accuracy_N={N}": accuracy
                    })

                # Compute overall stats (across all cluster sizes)
                all_h = []
                all_l = []
                for N in stats:
                    all_h.extend(stats[N]["hamming"])
                    all_l.extend(stats[N]["lev"])

                mean_h = np.mean(all_h)
                std_h = np.std(all_h)
                mean_l = np.mean(all_l)
                std_l = np.std(all_l)

                # Calculate overall success/failure rate (same as inference.py)
                num_successes = sum(1 for h in all_h if h == 0)
                success_rate = num_successes / len(all_h) if len(all_h) > 0 else 0.0
                failure_rate = 1 - success_rate

                wandb.log({
                    f"{condition_name}_mean_hamming_all": mean_h,
                    f"{condition_name}_std_hamming_all": std_h,
                    f"{condition_name}_mean_levenshtein_all": mean_l,
                    f"{condition_name}_std_levenshtein_all": std_l,
                    f"{condition_name}_success_rate_all": success_rate,
                    f"{condition_name}_failure_rate_all": failure_rate,
                })

                print(f"  Overall: Success: {success_rate:.3f} | Failure: {failure_rate:.3f} | H: {mean_h:.2f}±{std_h:.2f} | L: {mean_l:.2f}±{std_l:.2f}")

            wandb.finish()
            print("\nMisclustering experiment completed!")
            return  # Exit early, skip normal inference

        # create DataLoader for batched inference
        ds     = ClustersDataset(clusters, gts)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: (
                encode_reads_list(
                    [c for c, _ in batch],
                    args.padding_length,
                    max_num_reads=max(len(c) for c, _ in batch)
                ).float(),
                [gt for _, gt in batch],
                [len(c) for c, _ in batch]
            ),
            num_workers=0,
        )

        # load model
        model = Model(
            noise_length=args.padding_length,
            label_length=args.label_length,
            dim=args.dim,
            lstm_hidden_dim=args.lstm_hidden_dim,
            conv_dropout_p=args.conv_dropout_p,
            rnn_dropout_p=args.rnn_dropout_p
        ).to(device)

        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        state_dict = {
            k.replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict)
        model.eval()

        stats     = defaultdict(lambda: {"hamming": [], "lev": [], "correct": 0, "correct_chars": 0, "total_chars": 0})
        inf_times = []

        # batched loop
        for X_batch, gt_batch, N_batch in loader:
            X_batch = X_batch.to(device)

            # Synchronize GPU before timing
            if device.type == 'cuda':
                torch.cuda.synchronize()

            with torch.no_grad():
                t0     = time.perf_counter()
                logits = model(X_batch)

            # Synchronize GPU after inference
            if device.type == 'cuda':
                torch.cuda.synchronize()

            t1     = time.perf_counter()

            # divide batch latency by batch size
            per_ex = (t1 - t0) / X_batch.size(0)
            inf_times += [per_ex] * X_batch.size(0)

            preds = logits.argmax(dim=1).cpu().numpy()
            recs  = [decode_prediction(p)[:args.label_length] for p in preds]

            # compute metrics per example
            for rec, gt, N in zip(recs, gt_batch, N_batch):
                # per-char accuracy
                correct_chars = sum(p == t for p, t in zip(rec, gt))
                stats[N]["correct_chars"] += correct_chars
                stats[N]["total_chars"]   += len(gt)

                # distances
                h = hamming_distance_postprocessed(gt, rec)
                l = levenshtein_distance(gt, rec) / len(gt)
                stats[N]["hamming"].append(h)
                stats[N]["lev"].append(l)
                if h == 0:
                    stats[N]["correct"] += 1

                print(f"[N={N}] GT:{gt} PRED:{rec} H:{h} L:{l:.3f}")

        # aggregate & log
        print("\nAggregate Stats")
        for N in sorted(stats):
            h_arr = np.array(stats[N]["hamming"])
            l_arr = np.array(stats[N]["lev"])
            success  = stats[N]["correct"]      / len(h_arr)
            accuracy = stats[N]["correct_chars"] / stats[N]["total_chars"]
            print(
                f"N={N} | Success: {success:.3f} | Acc: {accuracy:.3f} | "
                f"H: {h_arr.mean():.2f}±{h_arr.std():.2f} | "
                f"L: {l_arr.mean():.2f}±{l_arr.std():.2f}"
            )
            wandb.log({
                f"avg_hamming_N={N}":     h_arr.mean(),
                f"std_hamming_N={N}":     h_arr.std(),
                f"avg_levenshtein_N={N}": l_arr.mean(),
                f"std_levenshtein_N={N}": l_arr.std(),
                f"success_rate_N={N}":    success,
                f"accuracy_N={N}":        accuracy
            })

        # timing summary
        if inf_times:
            arr = np.array(inf_times)
            print(
                f"\nPer-example time: {arr.mean():.6f} ± {arr.std():.6f} s"
            )
            wandb.log({
                "avg_time_per_example": arr.mean(),
                "std_time_per_example": arr.std()
            })

        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--artifact_name", type=str, required=False, help="W&B artifact name (not used if --local-data-dir is provided)")
    parser.add_argument("--local-data-dir", type=str, default=None, help="Path to local data directory containing reads.txt and ground_truth.txt (skips wandb download)")
    parser.add_argument("--batch_size", type=int, default=200, help="Number of clusters per forward pass")
    parser.add_argument("--test_project", type=str, required=False, help="W&B test_project name (not needed if --local-data-dir is provided)")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, required=False, help="W&B entity name (not needed if --local-data-dir is provided)")
    parser.add_argument("--padding_length", type=int, required=True)
    parser.add_argument("--label_length", type=int, required=True)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--target_type", type=str, default="CPRED")
    parser.add_argument("--lstm_hidden_dim", type=int, default=256)
    parser.add_argument("--conv_dropout_p", type=float, default=0.1)
    parser.add_argument("--rnn_dropout_p", type=float, default=0.1)
    # Sweep arguments
    parser.add_argument("--sweep", action="store_true", help="Whether to run sweep inference runs")
    parser.add_argument("--test_seed", type=int, default=34721, help="Base seed for sweep runs")
    parser.add_argument("--num_sweep_runs", type=int, default=11, help="Number of sweep runs (default=11)")
    # Misclustering arguments
    parser.add_argument("--misclustering", action="store_true", help="Whether to run misclustering robustness experiment")
    parser.add_argument("--contamination-rates", type=str, default="0.02,0.05,0.08,0.1,0.12,0.15,0.18,0.2", help="Comma-separated contamination rates")
    parser.add_argument("--seed", type=int, default=365, help="Random seed for contamination")
    # Sampling arguments
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use (default: None = use all)")
    parser.add_argument("--sampling_seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    # Timing arguments
    parser.add_argument("--timing", action="store_true", help="Enable timing mode for throughput measurement")
    parser.add_argument("--timing-duration", type=int, default=300, help="Duration of each timing run in seconds (default: 300s = 5 min)")
    parser.add_argument("--timing-runs", type=int, default=6, help="Total number of timing runs including warmup (default: 6)")
    parser.add_argument("--timing-warmup", type=int, default=1, help="Number of warmup runs to discard (default: 1)")
    parser.add_argument("--timing-seed", type=int, default=42, help="Random seed for timing subset sampling (default: 42)")
    parser.add_argument("--timing-samples-per-obs", type=int, default=50, help="Number of examples to sample per observation size for timing (default: 50)")
    args = parser.parse_args()

    # Global for collate function
    global padding_length
    padding_length = args.padding_length

    run_inference(args)
