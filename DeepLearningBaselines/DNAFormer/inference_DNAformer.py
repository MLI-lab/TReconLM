"""
Created new inference script for DNAformer for convenience.

"""
import os
import sys
import time
import requests
from requests.exceptions import ChunkedEncodingError
from urllib3.exceptions import ProtocolError
import torch.distributed as dist
import numpy as np
from collections import defaultdict

import wandb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader_IDS import PrecomputedDNAData, collate_dna
from helper import save_results, evaluate_and_log
from datetime import datetime
import pickle
from types import SimpleNamespace
from Levenshtein import distance as levenshtein_distance

# Add TReconLM root to path for importing contamination function
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, repo_root)
from src.utils.helper_functions import contaminate_trace_cluster
from src.utils.hamming_distance import hamming_distance_postprocessed

def safe_download_artifact(entity, project, artifact_name, max_retries=3, local_data_dir=None):
    """
    Download wandb artifact or use local directory.

    Args:
        entity: WandB entity name
        project: WandB project name
        artifact_name: Artifact name to download
        max_retries: Maximum number of retry attempts
        local_data_dir: If provided, use this local directory instead of downloading

    Returns:
        Path to data directory
    """
    if local_data_dir:
        print(f"Using local data directory: {local_data_dir}")
        # Verify required files exist
        test_x_path = os.path.join(local_data_dir, 'test_x.pt')
        gt_path = os.path.join(local_data_dir, 'ground_truth.txt')

        if not os.path.exists(test_x_path):
            raise FileNotFoundError(f"test_x.pt not found in {local_data_dir}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"ground_truth.txt not found in {local_data_dir}")

        return local_data_dir

    # Download from wandb
    for attempt in range(1, max_retries + 1):
        try:
            artifact = wandb.use_artifact(
                f'{entity}/{project}/{artifact_name}:latest',
                type='dataset'
            )
            return artifact.download()
        except (requests.exceptions.RequestException,
                ChunkedEncodingError,
                ProtocolError) as e:
            print(f"Attempt {attempt} failed: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to download {artifact_name} after {max_retries} attempts.")

def run_timing_measurement(config, model, x_test, ground_truths, meta):
    """
    Run throughput measurement by cycling through dataset for fixed time windows.

    Measures pure model inference throughput by repeatedly processing examples
    for a fixed duration, then reporting examples/hour with statistics.

    Args:
        config: Configuration object with timing parameters
        model: Loaded DNAFormer model
        x_test: List of input tensors
        ground_truths: List of ground truth strings
        meta: Metadata dict with stoi/itos
    """
    from itertools import cycle

    run_duration = getattr(config, 'timing_duration', 300)
    num_runs = getattr(config, 'timing_runs', 6)
    warmup_runs = getattr(config, 'timing_warmup', 1)
    batch_size = config.test_batch_size

    print(f"\n{'='*80}")
    print("TIMING MODE: THROUGHPUT MEASUREMENT")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Model: DNAFormer ({config.model_config})")
    print(f"  Run duration: {run_duration / 60:.1f} minutes ({run_duration}s)")
    print(f"  Number of runs: {num_runs} ({warmup_runs} warmup + {num_runs - warmup_runs} measured)")
    print(f"  Batch size: {batch_size}")
    print(f"  Dataset size: {len(x_test)} examples")
    print(f"  Device: {config.device}")
    print(f"")

    # Check actual unpadded input lengths and sort (consistent with TReconLM)
    stoi, itos = meta['stoi'], meta['itos']

    def get_unpadded_input_length(x_tensor):
        """Get length of actual input (after removing padding and extracting input part)"""
        decoded = ''.join(itos[i] for i in x_tensor.tolist())
        unpadded = decoded.split('#', 1)[0]  # Remove padding
        input_only = unpadded.split(':', 1)[0]  # Extract input part (before ground truth)
        return len(input_only)

    all_lengths = [get_unpadded_input_length(x) for x in x_test]
    min_length = min(all_lengths)
    max_length = max(all_lengths)
    avg_length = sum(all_lengths) / len(all_lengths)

    print(f"Unpadded input lengths: min={min_length}, max={max_length}, avg={avg_length:.1f}")

    # Create representative subset for timing
    # Sample evenly across observation_size (cluster size) for realistic throughput
    print(f"\nCreating representative timing subset:")

    import random
    from collections import defaultdict
    timing_seed = getattr(config, 'timing_seed', 42)
    rng = random.Random(timing_seed)

    # Group by observation_size (cluster size)
    # Need to extract cluster size from the input data format
    by_obs_size = defaultdict(list)
    for idx, (x, gt) in enumerate(zip(x_test, ground_truths)):
        # For DNAFormer, cluster size is embedded in the input format
        # Count number of reads by counting ':' separators before '#'
        decoded = ''.join(itos[i] for i in x.tolist())
        unpadded = decoded.split('#', 1)[0]
        obs_size = unpadded.count(':') + 1  # Number of reads = number of ':' + 1
        by_obs_size[obs_size].append((idx, x, gt))

    print(f"  Found observation sizes: {sorted(by_obs_size.keys())}")
    for obs in sorted(by_obs_size.keys()):
        print(f"    Obs size {obs}: {len(by_obs_size[obs])} examples")

    # Sample equal number from each observation_size
    samples_per_obs = getattr(config, 'timing_samples_per_obs', batch_size)  # Default to batch_size for clean division
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
        lengths = [get_unpadded_input_length(item[1]) for item in sampled]
        print(f"  Sampled {len(sampled)} from obs_size={obs}, lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

    print(f"  Total timing subset: {len(timing_subset)} examples")

    # Sort subset by input length for efficient batching
    timing_subset.sort(key=lambda item: get_unpadded_input_length(item[1]))
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
            batch_x = []
            batch_gts = []
            for _ in range(batch_size):
                x, gt = next(data_cycle)
                batch_x.append(x)
                batch_gts.append(gt)

            # Create dataset and get collated batch (same as normal inference)
            ds_batch = PrecomputedDNAData(batch_x, batch_gts, config, meta)
            # Manually collate since we're not using DataLoader
            batch_items = [ds_batch[i] for i in range(len(ds_batch))]
            batch = collate_dna(batch_items, siamese=(config.model_config == 'siamese'))

            # Prepare input based on model config
            if config.model_config == 'single':
                inp = batch['model_input'].to(config.device)
            else:
                left = batch['model_input']
                right = batch['model_input_right']
                inp = torch.cat([left, right], dim=0).to(config.device)

            # Run inference with timing
            if config.device.type == 'cuda':
                torch.cuda.synchronize()

            with torch.inference_mode():
                t0 = time.perf_counter()
                out = model(inp)

            if config.device.type == 'cuda':
                torch.cuda.synchronize()

            t1 = time.perf_counter()

            examples_this_run += batch_size

            # Log progress every 1000 examples
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


def run_inference(config, model):
    # load model checkpoint 
    ckpt   = torch.load(config.pretrained_path, map_location=config.device)
    raw_sd = ckpt['model_state_dict']

    # strip unwanted prefixes
    unwanted = "module._orig_mod."
    clean_sd = {
        (k[len(unwanted):] if k.startswith(unwanted) else k): v
        for k, v in raw_sd.items()
    }
    model.load_state_dict(clean_sd, strict=False)
    model.to(config.device).eval()

    # load vocab metadata 
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    repo_path   = os.path.dirname(os.path.dirname(script_dir))
    data_pkg_dir= os.path.join(repo_path, 'src', 'data_pkg')
    meta_path   = os.path.join(data_pkg_dir, 'meta_nuc.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    # shared stuff: timestamp & raw config dict
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_cfg = {
        k: v for k, v in vars(config).items()
        if not k.startswith("__") and not callable(v)
    }

    # determine sweep indices 
    # Requires config.sweep: bool, and config.test_seed: int
    ks = list(range(11)) if getattr(config, "sweep", False) else [None]

    for k in ks:
        # build per-run artifact name 
        if k is not None:
            seed = config.test_seed + k
            artifact_name = (
                f"sweep{k}_seed{seed}"
                f"_gl{config.label_length}"
                f"_bs1500"
                f"_ds5000"
            )
        else:
            artifact_name = config.test_artifact_name

        # build unique run-name & output directory 
        run_name = f"DNAformer_inference_{now_str}_gl{config.label_length}"
        if k is not None:
            run_name += f"_sweep_k{k}"
        out_dir_k = os.path.join(config.out_dir, config.project, run_name)
        os.makedirs(out_dir_k, exist_ok=True)

        # start a fresh W&B run 
        run = wandb.init(
            project = config.project,
            entity  = config.entity,
            dir     = out_dir_k,
            name    = run_name,
            resume  = False,
            config  = raw_cfg,
        )
        if k is not None:
            wandb.config.update({"sweep_index": k}, allow_val_change=True)

        # download this run's test data or use local directory
        local_data_dir = getattr(config, 'local_data_dir', None)
        art_dir = safe_download_artifact(
            config.entity,
            config.test_project,
            artifact_name,
            local_data_dir=local_data_dir
        )

        # load test examples
        x_test = torch.load(os.path.join(art_dir, 'test_x.pt'),
                             map_location='cpu')
        with open(os.path.join(art_dir, 'ground_truth.txt')) as f:
            ground_truths = [l.strip() for l in f]

        # Optional: Sample a subset of examples for faster testing
        max_samples = getattr(config, 'max_samples', None)
        sampling_seed = getattr(config, 'sampling_seed', 42)

        if max_samples is not None and max_samples < len(x_test):
            print(f"\n{'='*80}")
            print(f"SAMPLING SUBSET OF DATA")
            print(f"{'='*80}")
            print(f"Total examples available: {len(x_test)}")
            print(f"Max samples requested: {max_samples}")
            print(f"Random seed: {sampling_seed}")

            rng = np.random.RandomState(sampling_seed)
            sampled_indices = rng.choice(len(x_test), size=max_samples, replace=False)
            sampled_indices = sorted(sampled_indices)  # Sort for deterministic processing order

            x_test = [x_test[i] for i in sampled_indices]
            ground_truths = [ground_truths[i] for i in sampled_indices]
            print(f"Sampled {len(x_test)} examples for inference")
            print(f"{'='*80}\n")

        # Check if timing mode is enabled
        if getattr(config, 'timing', False):
            # Update wandb run name to indicate timing mode
            wandb.run.name = f"{run_name}_timing"

            # Run timing measurement
            run_timing_measurement(config, model, x_test, ground_truths, meta)

            # Finish WandB and exit
            wandb.finish()
            return  # Exit early after timing measurement

        # Handle misclustering experiment if enabled
        if getattr(config, 'misclustering', False):
            # Parse contamination rates
            contamination_rates_str = getattr(config, 'contamination_rates', '0.02,0.05,0.08,0.1,0.12,0.15,0.18,0.2')
            contamination_rates = [float(r.strip()) for r in contamination_rates_str.split(',')]
            print(f"Misclustering experiment enabled with rates: {contamination_rates}")

            # Create simple config object for contaminate_trace_cluster
            # Use error rates from the DNAFormer config (same as training)
            # Provide BOTH lb and ub - contaminate_trace_cluster will sample uniformly from [lb, ub]
            cfg = SimpleNamespace()
            cfg.data = SimpleNamespace()
            cfg.data.insertion_probability_lb = getattr(config, 'insertion_probability_lb', 0.01)
            cfg.data.insertion_probability_ub = getattr(config, 'insertion_probability_ub', 0.1)
            cfg.data.deletion_probability_lb = getattr(config, 'deletion_probability_lb', 0.01)
            cfg.data.deletion_probability_ub = getattr(config, 'deletion_probability_ub', 0.1)
            cfg.data.substitution_probability_lb = getattr(config, 'substitution_probability_lb', 0.01)
            cfg.data.substitution_probability_ub = getattr(config, 'substitution_probability_ub', 0.1)

            print(f"  Error rate sampling ranges (sampled uniformly per contaminant):")
            print(f"    INS=[{cfg.data.insertion_probability_lb:.3f}, {cfg.data.insertion_probability_ub:.3f}], "
                  f"DEL=[{cfg.data.deletion_probability_lb:.3f}, {cfg.data.deletion_probability_ub:.3f}], "
                  f"SUB=[{cfg.data.substitution_probability_lb:.3f}, {cfg.data.substitution_probability_ub:.3f}]")

            # Set random seed for reproducibility
            seed = getattr(config, 'seed', 365)
            rng = np.random.RandomState(seed)

            # Get stoi/itos for encoding/decoding
            stoi, itos = meta['stoi'], meta['itos']

            # Decode all x_test to raw sequences (clusters of traces)
            print("  Decoding test tensors to raw sequences...")
            clusters = []
            for x_tensor in tqdm(x_test, desc="  Decoding"):
                # Decode tensor to string
                original_seq = ''.join(itos[i] for i in x_tensor.tolist())
                # Remove padding
                original_seq = original_seq.split('#', 1)[0]
                # Split to get traces and ground truth
                noisy_part, gt_part = original_seq.split(':', 1)
                traces = [r for r in noisy_part.split('|') if r]
                clusters.append(traces)

            # Run contamination experiment for each rate
            for cont_rate in contamination_rates:
                print(f"\nProcessing contamination rate: {cont_rate}")

                # Contaminate all clusters
                contaminated_x_test = []
                contaminated_examples = []
                total_contaminated_traces = 0

                print("  Contaminating clusters...")
                for i, (traces, gt) in enumerate(tqdm(zip(clusters, ground_truths), total=len(clusters), desc="  Contaminating")):
                    # Contaminate the cluster
                    contaminated_traces, contamination_info = contaminate_trace_cluster(
                        traces=traces,
                        ground_truth=gt,
                        contamination_rate=cont_rate,
                        baseline_error_rate=0.055,
                        cfg=cfg,
                        rng=rng
                    )

                    # Re-encode to tensor
                    contaminated_seq = '|'.join(contaminated_traces) + ':' + gt
                    contaminated_x = torch.tensor([stoi.get(ch, stoi.get('<unk>', 0)) for ch in contaminated_seq], dtype=torch.long)
                    contaminated_x_test.append(contaminated_x)

                    # Track contamination
                    num_contaminated = len(contamination_info['contaminated_positions'])
                    if num_contaminated > 0:
                        contaminated_examples.append(i)
                        total_contaminated_traces += num_contaminated

                print(f"  Contaminated {len(contaminated_examples)} examples (out of {len(clusters)})")
                print(f"  Total contaminated traces: {total_contaminated_traces}")
                print(f"  Average contaminated traces per example: {total_contaminated_traces / len(clusters):.2f}")

                # Create DataLoader for contaminated data
                ds_cont = PrecomputedDNAData(contaminated_x_test, ground_truths, config, meta)
                loader_cont = DataLoader(
                    ds_cont,
                    batch_size=config.test_batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                    pin_memory=True,
                    collate_fn=lambda b: collate_dna(
                        b,
                        siamese=(config.model_config == 'siamese')
                    )
                )

                # Run inference on contaminated data
                print("  Running inference...")
                all_results_cont = []
                pbar_cont = tqdm(loader_cont, desc=f"  Inference", total=len(loader_cont))
                with torch.inference_mode():
                    for batch in pbar_cont:
                        if config.model_config == 'single':
                            inp = batch['model_input'].to(config.device)
                        else:
                            left = batch['model_input']
                            right = batch['model_input_right']
                            inp = torch.cat([left, right], dim=0).to(config.device)

                        # Synchronize GPU before timing
                        if config.device.type == 'cuda':
                            torch.cuda.synchronize()

                        t0 = time.perf_counter()
                        out = model(inp)

                        # Synchronize GPU after model inference
                        if config.device.type == 'cuda':
                            torch.cuda.synchronize()

                        dt = time.perf_counter() - t0

                        probs = torch.softmax(out['pred'], dim=1)
                        per_example_time = dt / inp.size(0)

                        batch_results = save_results(config, batch, probs)
                        for res in batch_results:
                            res["inf_time"] = per_example_time
                        all_results_cont.extend(batch_results)

                # Evaluate and log results for this contamination rate
                condition_name = f"cont_{cont_rate:.3f}"
                print(f"\n{condition_name} Stats:")

                # Compute per-cluster-size metrics
                stats_by_N = defaultdict(lambda: {'hamming': [], 'levenshtein': [], 'correct': 0})
                for res in all_results_cont:
                    N = res['N']
                    gt = res['ground_truth']
                    pred = res['prediction']

                    # Compute distances (same as evaluate_and_log)
                    hamming = hamming_distance_postprocessed(gt, pred)
                    lev = levenshtein_distance(gt, pred) / len(gt)

                    stats_by_N[N]['hamming'].append(hamming)
                    stats_by_N[N]['levenshtein'].append(lev)
                    if hamming == 0:
                        stats_by_N[N]['correct'] += 1

                # Log per-cluster-size metrics
                for N in sorted(stats_by_N.keys()):
                    h_arr = np.array(stats_by_N[N]['hamming'])
                    l_arr = np.array(stats_by_N[N]['levenshtein'])
                    success = stats_by_N[N]['correct'] / len(h_arr) if len(h_arr) > 0 else 0
                    print(f"  N={N} | Success: {success:.3f} | H: {h_arr.mean():.2f}±{h_arr.std():.2f} | L: {l_arr.mean():.2f}±{l_arr.std():.2f}")
                    wandb.log({
                        f"{condition_name}_avg_hamming_N={N}": h_arr.mean(),
                        f"{condition_name}_std_hamming_N={N}": h_arr.std(),
                        f"{condition_name}_avg_levenshtein_N={N}": l_arr.mean(),
                        f"{condition_name}_std_levenshtein_N={N}": l_arr.std(),
                        f"{condition_name}_success_rate_N={N}": success,
                    })

                # Compute overall stats (across all cluster sizes)
                all_h = []
                all_l = []
                for N in stats_by_N:
                    all_h.extend(stats_by_N[N]['hamming'])
                    all_l.extend(stats_by_N[N]['levenshtein'])

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

        #build DataLoader
        ds = PrecomputedDNAData(x_test, ground_truths, config, meta)
        loader = DataLoader(
            ds,
            batch_size   = config.test_batch_size,
            shuffle      = False,
            num_workers  = config.num_workers,
            pin_memory   = True,
            collate_fn   = lambda b: collate_dna(
                b,
                siamese=(config.model_config=='siamese')
            )
        )

        # inference loop 
        all_results = []
        pbar = tqdm(loader, desc=f"Inference (k={k})", total=len(loader),
                    leave=False, dynamic_ncols=True)
        with torch.inference_mode():
            for batch in pbar:
                if config.model_config == 'single':
                    inp = batch['model_input'].to(config.device)
                else:
                    left  = batch['model_input']
                    right = batch['model_input_right']
                    inp   = torch.cat([left, right], dim=0).to(config.device)

                # Synchronize GPU before timing
                if config.device.type == 'cuda':
                    torch.cuda.synchronize()

                t0 = time.perf_counter()
                out = model(inp)

                # Synchronize GPU after model inference
                if config.device.type == 'cuda':
                    torch.cuda.synchronize()

                dt = time.perf_counter() - t0

                probs = torch.softmax(out['pred'], dim=1)
                per_example_time = dt / inp.size(0)

                batch_results = save_results(config, batch, probs)
                for res in batch_results:
                    res["inf_time"] = per_example_time
                all_results.extend(batch_results)

        #evaluate & log
        evaluate_and_log(
            all_results,
            out_dir_k,
            log_to_wandb=True
        )

        #finish this W&B run
        wandb.finish()

    print("All sweep runs completed.")
