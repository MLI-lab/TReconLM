import os

# On zion and yosemite, disable NCCL P2P due to broken GPU-to-GPU communication
# This fixes hanging barriers and broadcasts in distributed mode
os.environ["NCCL_P2P_DISABLE"] = "1"

import torch
import pickle
import wandb
import numpy as np
import math
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
from collections import defaultdict
from tqdm import tqdm
import torch.distributed as dist
from Levenshtein import distance as levenshtein_distance
from requests.exceptions import ChunkedEncodingError
from urllib3.exceptions import ProtocolError
import time
import csv
import sys
import json

from src.utils.helper_functions import (
    filter_string,
    safe_download_artifact,
    split_list,
    calculate_baseline_error_rate,
    contaminate_trace_cluster,
    create_multiplier_bins,
    create_cluster_size_bins,
    exchange_positional_encoding,
    is_distributed,
    save_contaminated_attention_data,
    permute_traces_in_tensor
)
from src.utils.hamming_distance import hamming_distance_postprocessed
from src.eval_pkg.GPT_Inference import GPT_Inference
from src.gpt_pkg.model import GPT, GPTConfig
from src.rnn_pkg.lstm_model import LSTMConfig, LSTMConsensus
from src.utils.wandb_utils import wandb_kwargs_via_cfg


# Import vote confidence analysis utilities
from src.utils.vote_confidence_analysis import (
    collect_position_metrics,
    analyze_and_log_vote_confidence
)

# Import logit margin profiling utilities
from src.utils.logit_margin_profiling import (
    subsample_data_uniformly,
    profile_logit_margins_single_batch,
    aggregate_margin_statistics,
    print_margin_statistics
)


# ======================= Helper Functions =======================

def print_separator(width=80, char='=', newline_before=True, newline_after=False):
    """Print a separator line for better output formatting."""
    if newline_before:
        print()
    print(char * width)
    if newline_after:
        print()


def print_section_header(title, width=80):
    """Print a formatted section header."""
    print_separator(width=width, newline_before=True, newline_after=False)
    print(title)
    print_separator(width=width, newline_before=False, newline_after=False)


def extract_sampling_config(cfg):
    """Extract sampling configuration from Hydra config."""
    sampling_dict = OmegaConf.to_container(cfg.model.sampling, resolve=True)
    sampling_dict.update({
        'block_size': cfg.data.block_size,
        'target_type': cfg.data.target_type,
        'ground_truth_length': cfg.data.ground_truth_length,
        'greedy': cfg.model.sampling.strategy == 'greedy',
        'model_type': cfg.model.model_type,
        'constrained_generation': cfg.model.sampling.get('constrained_generation', False),
        'debug_batch_lengths': cfg.model.sampling.get('debug_batch_lengths', False),
    })
    return sampling_dict


# ======================= Main Functions =======================

def run_misclustering_robustness_experiment(all_data, cfg, model, meta, device, ctx, rank=0):
    """
    Run the misclustering robustness analysis experiment.

    Args:
        all_data: List of (x_tensor, ground_truth, cluster_size) tuples
        cfg: Hydra config
        model, meta, device, ctx: Model and inference setup
        rank: Process rank for distributed inference

    Returns:
        dict: Experiment results for logging
    """
    # Check if misclustering experiment is enabled
    misc_cfg = cfg.get('misclustering_robustness', None)
    if not misc_cfg:
        return None

    print_section_header("MISCLUSTERING ROBUSTNESS ANALYSIS", width=80)

    # Calculate baseline error rates from config
    baseline_info = calculate_baseline_error_rate(cfg)
    baseline_error_rate_per_nt = baseline_info['error_rate_per_nt']

    # Experiment parameters
    contamination_rates = misc_cfg.get('contamination_rates', [0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2])

    print(f"Experiment Configuration:")
    print(f"  Contamination rates: {contamination_rates}")
    print(f"  Natural contamination: edit distances will be recorded and grouped post-hoc")

    # Set up random number generator for reproducibility
    experiment_seed = misc_cfg.get('seed', 365)
    rng = np.random.RandomState(experiment_seed)
    print(f"  Random seed: {experiment_seed}")

    # Results storage
    experiment_results = {
        'baseline_error_rate_per_nt': baseline_error_rate_per_nt,
        'baseline_info': baseline_info,
        'contamination_rates': contamination_rates,
        'results_by_condition': {}, # one condition is one contamination rate
        'per_cluster_results': defaultdict(list),
        'contamination_details': [],  # Store all contamination info for post-hoc analysis
        'multiplier_bin_config': {'num_bins': misc_cfg.get('multiplier_bins', 10)},
        'contaminated_example_indices_per_rate': {}  # Track which examples had contamination per rate
    }

    # Filter by cluster size if specified
    cluster_size_filter = misc_cfg.get('cluster_size_filter', None)
    if cluster_size_filter is not None:
        # Convert to list if single value
        if isinstance(cluster_size_filter, int):
            cluster_size_filter = [cluster_size_filter]

        print(f"  Filtering by cluster size(s): {cluster_size_filter}")
        filtered_data = [(x, gt, cs) for x, gt, cs in all_data if cs in cluster_size_filter]
        print(f"  Filtered from {len(all_data)} to {len(filtered_data)} examples")
        all_data = filtered_data

    # Sample subset of data for experiment (to avoid long runtime)
    max_samples = misc_cfg.get('max_samples', 500000000000)
    if len(all_data) > max_samples:
        sampled_indices = rng.choice(len(all_data), size=max_samples, replace=False)
        # Store tuples of (sequential_index, x_tensor, ground_truth, cluster_size)
        # Use enumerate to get indices 0, 1, 2, ... within the sampled subset
        experiment_data = [(idx, all_data[i][0], all_data[i][1], all_data[i][2]) for idx, i in enumerate(sampled_indices)]
        print(f"  Sampled {max_samples} examples from {len(all_data)} total")
        print(f"  Sampled indices from original data: {sorted(sampled_indices)[:10]}..." if len(sampled_indices) > 10 else f"  Sampled indices: {sorted(sampled_indices)}")
    else:
        # Store tuples of (sequential_index, x_tensor, ground_truth, cluster_size)
        experiment_data = [(i, x, gt, cs) for i, (x, gt, cs) in enumerate(all_data)]
        print(f"  Using all {len(experiment_data)} examples")

    # Sort experiment data by cluster size for efficient batched processing
    experiment_data.sort(key=lambda x: x[3])  # Sort by cluster_size (fourth element now)
    print(f"  Sorted experiment data by cluster size for batched processing")

    # Run experiment across all conditions
    total_conditions = len(contamination_rates)
    condition_idx = 0

    # for each contamination rate process all examples with contamination applied and collect results in condition_results
    for contamination_rate in contamination_rates:
        condition_idx += 1
        condition_name = f"cont_{contamination_rate:.3f}"

        print(f"\nCondition {condition_idx}/{total_conditions}: {condition_name}")
        print(f"  Contamination rate: {contamination_rate:.1%}")

        # Process data in batches grouped by cluster size for efficient inference
        contamination_stats = defaultdict(list)
        condition_results = []
        condition_results_indices = []  # Track original example index for each result

        # Track contaminated examples for this specific rate
        contaminated_indices_this_rate = set()

        # Group experiment data by cluster size
        grouped_data = defaultdict(list)
        for orig_idx, x_tensor, ground_truth, cluster_size in experiment_data:
            grouped_data[cluster_size].append((orig_idx, x_tensor, ground_truth, cluster_size))

        # Process each cluster size group in batches
        batch_size = max(cfg.data.batch_size, 32)
        example_counter = 0

        for cluster_size, cluster_data in grouped_data.items():
            print(f"  Processing {len(cluster_data)} examples with cluster size {cluster_size}")

            # Process this cluster size in batches
            num_batches = math.ceil(len(cluster_data) / batch_size)

            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(cluster_data))
                batch_examples = cluster_data[start:end]

                # Apply contamination to all examples in this batch
                batch_contaminated = []
                batch_contamination_info = []
                for orig_idx, x_tensor, ground_truth, cs in batch_examples:
                    example_counter += 1
                    # Decode the original sequence to get individual traces
                    stoi, itos = meta['stoi'], meta['itos']
                    original_seq = ''.join(itos[i_char] for i_char in x_tensor.tolist())
                    prefix, gt_part = original_seq.split(':', 1)
                    traces = prefix.split('|')

                    # Apply contamination
                    contaminated_traces, contamination_info = contaminate_trace_cluster(
                        traces, ground_truth, contamination_rate, baseline_error_rate_per_nt, cfg, rng
                    )

                    # Reconstruct the input sequence
                    contaminated_seq = '|'.join(contaminated_traces) + ':' + gt_part
                    contaminated_x = torch.tensor([stoi.get(ch, stoi.get('<unk>', 0)) for ch in contaminated_seq], dtype=torch.long)

                    batch_contaminated.append((contaminated_x, ground_truth, cs))

                    # Track contamination statistics
                    contamination_stats['num_contaminants'].append(contamination_info['num_contaminants'])
                    contamination_stats['contamination_rate'].append(contamination_info['contamination_rate'])

                    # Track which examples had contamination for this specific rate
                    num_contaminants = contamination_info['num_contaminants']
                    if num_contaminants > 0:
                        contaminated_indices_this_rate.add(orig_idx)

                    # Print contamination details for this example
                    if num_contaminants > 0:
                        realized_multipliers = [pos['realized_edit_distance_multiplier']
                                              for pos in contamination_info['contaminated_positions']]
                        avg_multiplier = np.mean(realized_multipliers) if realized_multipliers else 0
                        print(f"    Example {example_counter:3d}: {num_contaminants:2d}/{cs:2d} traces contaminated, avg edit distance multiplier: {avg_multiplier:.2f}")
                    else:
                        print(f"    Example {example_counter:3d}: 0/{cs:2d} traces contaminated (no contamination)")

                    # Store contamination details for post-hoc analysis
                    contamination_info['condition_name'] = condition_name
                    contamination_info['example_index'] = orig_idx
                    experiment_results['contamination_details'].append(contamination_info)

                    # Add to batch contamination info for attention tracking
                    batch_contamination_info.append(contamination_info.copy())

                # Run batched inference on contaminated data (all same cluster size = efficient padding)
                sampling_dict = extract_sampling_config(cfg)
                sampling_dict['cross_mode'] = cfg.data.get('cross', None)
                sampling_dict['track_attention'] = cfg.model.sampling.get('track_attention', False)
                sampling_dict['track_all_layers'] = cfg.model.sampling.get('track_all_layers', False)
                sampling_dict['save_per_head_attention'] = cfg.model.sampling.get('save_per_head_attention', False)

                # Create contamination lookup for this batch
                # Use original indices so they match normal inference indices
                contamination_lookup = {}
                for info in batch_contamination_info:
                    orig_example_idx = info['example_index']
                    contamination_lookup[orig_example_idx] = info

                # Extract original indices from batch_examples for proper alignment
                batch_orig_indices = [orig_idx for orig_idx, _, _, _ in batch_examples]

                res, _, _ = run_one_batch(batch_contaminated, 0, len(batch_contaminated),
                                        sampling_dict, model, meta, device, ctx,
                                        contamination_lookup=contamination_lookup, condition_name=condition_name,
                                        example_indices=batch_orig_indices)
                condition_results.extend(res)
                condition_results_indices.extend(batch_orig_indices)  # Track original indices

        # Calculate metrics for this condition
        hamming_distances = [result[8] for result in condition_results]
        levenshtein_distances = [result[9] for result in condition_results]

        # Calculate overall success/failure rate (same as normal inference)
        num_successes = sum(1 for h in hamming_distances if h == 0)
        success_rate = num_successes / len(hamming_distances) if len(hamming_distances) > 0 else 0.0
        failure_rate = 1 - success_rate

        # Calculate overall metrics for clusters with at least one contaminated sequence
        contaminated_cluster_indices = [i for i, orig_idx in enumerate(condition_results_indices)
                                       if orig_idx in contaminated_indices_this_rate]
        if len(contaminated_cluster_indices) > 0:
            contaminated_cluster_ham = [condition_results[i][8] for i in contaminated_cluster_indices]
            contaminated_cluster_lev = [condition_results[i][9] for i in contaminated_cluster_indices]
            contaminated_cluster_successes = sum(1 for h in contaminated_cluster_ham if h == 0)
            contaminated_cluster_success_rate = contaminated_cluster_successes / len(contaminated_cluster_ham)
            contaminated_cluster_failure_rate = 1 - contaminated_cluster_success_rate
            contaminated_cluster_metrics = {
                'num_examples': len(contaminated_cluster_ham),
                'mean_hamming': float(np.mean(contaminated_cluster_ham)),
                'std_hamming': float(np.std(contaminated_cluster_ham)),
                'mean_levenshtein': float(np.mean(contaminated_cluster_lev)),
                'std_levenshtein': float(np.std(contaminated_cluster_lev)),
                'success_rate': contaminated_cluster_success_rate,
                'failure_rate': contaminated_cluster_failure_rate
            }
        else:
            contaminated_cluster_metrics = {
                'num_examples': 0,
                'mean_hamming': 0.0,
                'std_hamming': 0.0,
                'mean_levenshtein': 0.0,
                'std_levenshtein': 0.0,
                'success_rate': 0.0,
                'failure_rate': 0.0
            }

        condition_metrics = {
            'mean_hamming': np.mean(hamming_distances),
            'std_hamming': np.std(hamming_distances),
            'mean_levenshtein': np.mean(levenshtein_distances),
            'std_levenshtein': np.std(levenshtein_distances),
            'num_examples': len(condition_results),
            'success_rate': success_rate,
            'failure_rate': failure_rate,
            'contamination_stats': {k: np.mean(v) for k, v in contamination_stats.items()},
            'clusters_with_contamination': contaminated_cluster_metrics
        }

        # Aggregate metrics by cluster size (all examples vs contaminated only)
        by_cluster_size = {}
        cluster_sizes = set([result[0] for result in condition_results])

        for N in sorted(cluster_sizes):
            # Get all results for this cluster size
            all_indices = [i for i, result in enumerate(condition_results) if result[0] == N]
            all_ham = [condition_results[i][8] for i in all_indices]
            all_lev = [condition_results[i][9] for i in all_indices]

            # Get contaminated-only results for this cluster size
            contaminated_indices = [i for i in all_indices
                                   if condition_results_indices[i] in contaminated_indices_this_rate]
            cont_ham = [condition_results[i][8] for i in contaminated_indices]
            cont_lev = [condition_results[i][9] for i in contaminated_indices]

            # Compute metrics for all examples
            all_success_count = sum(1 for h in all_ham if h == 0)
            all_metrics = {
                'count': len(all_ham),
                'mean_hamming': float(np.mean(all_ham)),
                'std_hamming': float(np.std(all_ham)),
                'mean_levenshtein': float(np.mean(all_lev)),
                'std_levenshtein': float(np.std(all_lev)),
                'success_rate': all_success_count / len(all_ham) if len(all_ham) > 0 else 0.0,
                'failure_rate': 1 - (all_success_count / len(all_ham)) if len(all_ham) > 0 else 0.0
            }

            # Compute metrics for contaminated-only examples
            if len(cont_ham) > 0:
                cont_success_count = sum(1 for h in cont_ham if h == 0)
                cont_metrics = {
                    'count': len(cont_ham),
                    'mean_hamming': float(np.mean(cont_ham)),
                    'std_hamming': float(np.std(cont_ham)),
                    'mean_levenshtein': float(np.mean(cont_lev)),
                    'std_levenshtein': float(np.std(cont_lev)),
                    'success_rate': cont_success_count / len(cont_ham),
                    'failure_rate': 1 - (cont_success_count / len(cont_ham))
                }
            else:
                # No contaminated examples for this cluster size at this rate
                cont_metrics = {
                    'count': 0,
                    'mean_hamming': 0.0,
                    'std_hamming': 0.0,
                    'mean_levenshtein': 0.0,
                    'std_levenshtein': 0.0,
                    'success_rate': 0.0,
                    'failure_rate': 0.0
                }

            by_cluster_size[N] = {
                'all_examples': all_metrics,
                'contaminated_only': cont_metrics
            }

        # Store with new 2D structure for heatmap
        experiment_results['results_by_condition'][condition_name] = {
            'overall': condition_metrics,
            'by_cluster_size': by_cluster_size,  # Add cluster size breakdown
            'by_multiplier_bin': {}  # Will be filled in post-processing
        }

        # Store contaminated indices for this rate
        experiment_results['contaminated_example_indices_per_rate'][condition_name] = sorted(list(contaminated_indices_this_rate))

        # Store per-cluster results with example indices for contamination mapping
        for idx, result in enumerate(condition_results):
            cluster_size = result[0]
            experiment_results['per_cluster_results'][cluster_size].append({
                'condition': condition_name,
                'contamination_rate': contamination_rate,
                'hamming_distance': result[8],
                'levenshtein_distance': result[9],
                'example_index': idx  # Add example index for contamination mapping
            })

        print(f"  Results: Hamming {condition_metrics['mean_hamming']:.3f}±{condition_metrics['std_hamming']:.3f}, "
                  f"Levenshtein {condition_metrics['mean_levenshtein']:.3f}±{condition_metrics['std_levenshtein']:.3f}, "
                  f"Success={condition_metrics['success_rate']:.2%}, Failure={condition_metrics['failure_rate']:.2%}")

        # Print cluster size breakdown
        print(f"  Breakdown by cluster size:")
        for N in sorted(by_cluster_size.keys()):
            all_metrics = by_cluster_size[N]['all_examples']
            cont_metrics = by_cluster_size[N]['contaminated_only']
            print(f"    N={N}: All ({all_metrics['count']} examples): "
                  f"Failure={all_metrics['failure_rate']:.2%}, "
                  f"HAM={all_metrics['mean_hamming']:.2f}, LEV={all_metrics['mean_levenshtein']:.4f} | "
                  f"Contaminated ({cont_metrics['count']} examples): "
                  f"Failure={cont_metrics['failure_rate']:.2%}, "
                  f"HAM={cont_metrics['mean_hamming']:.2f}, LEV={cont_metrics['mean_levenshtein']:.4f}")

    print(f"\nMisclustering robustness experiment completed!")
    print(f"   Total conditions tested: {total_conditions}")
    print(f"   Examples per condition: {len(experiment_data)}")
    print(f"   Total contamination events recorded: {len(experiment_results['contamination_details'])}")

    # Print contaminated examples per rate
    for condition_name, indices in experiment_results['contaminated_example_indices_per_rate'].items():
        print(f"   {condition_name}: {len(indices)} examples with contamination")

    # Run baseline inference on contaminated examples if requested
    run_baseline = misc_cfg.get('run_baseline_on_subset', False)
    if run_baseline:
        print_section_header("BASELINE INFERENCE ON CONTAMINATED EXAMPLES (for attention matching)", width=80)

        # Collect all unique contaminated indices across all rates
        all_contaminated_indices = set()
        for indices in experiment_results['contaminated_example_indices_per_rate'].values():
            all_contaminated_indices.update(indices)

        contaminated_indices_sorted = sorted(all_contaminated_indices)
        print(f"  Found {len(contaminated_indices_sorted)} unique contaminated examples across all rates")

        # Build mapping from index to data
        index_to_data = {idx: (x, gt, cs) for idx, x, gt, cs in experiment_data}

        # Get baseline data for contaminated examples only
        baseline_data = [index_to_data[idx] for idx in contaminated_indices_sorted if idx in index_to_data]

        print(f"  Running baseline on {len(baseline_data)} contaminated examples")
        print(f"  Index range: {min(contaminated_indices_sorted)}-{max(contaminated_indices_sorted)}")

        # Run batched inference
        baseline_cfg = OmegaConf.to_container(cfg.model.sampling, resolve=True)
        baseline_cfg.update({
            'block_size': cfg.data.block_size,
            'target_type': cfg.data.target_type,
            'ground_truth_length': cfg.data.ground_truth_length,
            'greedy': cfg.model.sampling.strategy == 'greedy',
            'model_type': cfg.model.get('model_type', 'gpt'),
            'cross_mode': cfg.data.get('cross', None),
            'constrained_generation': cfg.model.sampling.get('constrained_generation', False),
            'track_attention': cfg.model.sampling.get('track_attention', False),
        })

        batch_size = max(cfg.data.batch_size, 32)
        num_batches = math.ceil(len(baseline_data) / batch_size)

        print(f"  Attention tracking: {baseline_cfg.get('track_attention', False)}")
        print(f"  Processing in {num_batches} batches...")

        for batch_idx in range(num_batches):
            start_pos = batch_idx * batch_size
            end_pos = min(start_pos + batch_size, len(baseline_data))
            batch_chunk = baseline_data[start_pos:end_pos]
            batch_indices = contaminated_indices_sorted[start_pos:end_pos]

            # Run inference with matching indices
            run_one_batch(batch_chunk, 0, len(batch_chunk),
                         baseline_cfg, model, meta, device, ctx,
                         example_indices=batch_indices)

        print(f"   Baseline inference completed!")
        print(f"   Saved attention files for contaminated examples with matching indices")
        print(f"   Files: attention_sample_{{idx}}.pt for idx in {contaminated_indices_sorted[:5]}...{contaminated_indices_sorted[-3:]}")
        print(f"   These match contaminated_ex{{idx}}_*.npz files!\n")

    # Save contaminated example indices per rate for fair comparison (only if path is specified)
    indices_save_path = misc_cfg.get('contaminated_indices_save_path', None)
    if indices_save_path and experiment_results['contaminated_example_indices_per_rate']:
        os.makedirs(os.path.dirname(indices_save_path), exist_ok=True)

        save_data = {
            'contaminated_indices_per_rate': experiment_results['contaminated_example_indices_per_rate'],
            'total_examples': len(experiment_data),
            'experiment_date': datetime.now().isoformat(),
            'contamination_rates': list(contamination_rates)  # Convert ListConfig to list
        }

        with open(indices_save_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"\n   Saved contaminated example indices (per rate) to: {indices_save_path}")
    elif not indices_save_path:
        print(f"   Note: contaminated_indices_save_path not set - indices will not be saved")

    # Post-process: Create bins based on realized edit distance multipliers
    print(f"\nPost-processing: Creating edit distance multiplier bins...")
    experiment_results = create_multiplier_bins(experiment_results)

    # Post-process: Create cluster size aggregations
    print(f"\nPost-processing: Creating cluster size aggregations...")
    experiment_results = create_cluster_size_bins(experiment_results)

    return experiment_results


def run_one_batch(chunk, start, batch_size,
                  cfg_dict,
                  model, meta, device, ctx, pbar=None,
                  contamination_lookup=None, condition_name=None, example_indices=None):
    """
    chunk : list[(x_tensor, ground_truth_str, true_cluster_size)] or list[(x_tensor, ground_truth_str)]
    """
    # Define write_output function at the top to avoid scoping issues
    def write_output(msg):
        if pbar:
            pbar.write(msg)
        else:
            print(msg)

    # Log which device is being used
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        if start == 0:  # Only log once per rank
            print(f"Rank {rank}: Running batch on device {device}")

    stoi, itos   = meta['stoi'], meta['itos']
    decode_tokens = lambda t: ''.join(itos[i] for i in t)
    encode_str    = lambda s: [stoi.get(ch, stoi.get('<unk>', 0)) for ch in s]

    # take a slice of data
    batch    = chunk[start:start + batch_size]

    # Debug: Print batch length statistics if enabled
    debug_batch_lengths = cfg_dict.get('debug_batch_lengths', False)
    if debug_batch_lengths and len(batch) > 0:
        batch_lengths = [len(item[0]) for item in batch]
        min_len = min(batch_lengths)
        max_len = max(batch_lengths)
        avg_len = sum(batch_lengths) / len(batch_lengths)
        write_output(f"  Batch {start//batch_size}: size={len(batch)}, lengths: min={min_len}, max={max_len}, avg={avg_len:.1f}, range={max_len-min_len}")

    # Check if chunk contains true cluster sizes (3-tuple) or not (2-tuple)
    has_true_sizes = len(batch) > 0 and len(batch[0]) == 3

    if has_true_sizes:
        examples = [decode_tokens(x.tolist()) for x, _, _ in batch]
        gts      = [gt for _, gt, _ in batch]
        true_cluster_sizes = [cs for _, _, cs in batch]
    else:
        examples = [decode_tokens(x.tolist()) for x, _ in batch]
        gts      = [gt for _, gt in batch]
        true_cluster_sizes = None

    # keep only items that contain ':'
    if has_true_sizes:
        valid = [(ex, gt, cs) for ex, gt, cs in zip(examples, gts, true_cluster_sizes) if ':' in ex]
    else:
        valid = [(ex, gt) for ex, gt in zip(examples, gts) if ':' in ex]

    if not valid:
        return [], 0, []  # Return tuple with empty results, 0 skipped, and empty position metrics

    if has_true_sizes:
        inputs, gts, true_cluster_sizes = zip(*valid)
        # Use the true cluster sizes instead of calculating from the string
        alignment_sizes = list(true_cluster_sizes)
    else:
        inputs, gts = zip(*valid)
        alignment_sizes = [len(ex.split(':')[0].split('|')) for ex in inputs]

    # Check for oversized inputs and filter them out
    block_size = cfg_dict.get('block_size', 1500)
    ground_truth_length = cfg_dict.get('ground_truth_length', 110)

    valid_inputs, valid_gts, valid_alignment_sizes = [], [], []
    skipped_count = 0

    for inp, gt, N in zip(inputs, gts, alignment_sizes):
        prefix_part = inp.split(':')[0]  # Get only the reads part before ':'
        prefix_length = len(prefix_part)
        total_required = prefix_length + 1 + ground_truth_length  # +1 for the ':' separator

        if total_required > block_size:
            skipped_count += 1
            print(f"\n SKIPPING oversized input:")
            print(f"   Cluster size: {N}")
            print(f"   Prefix (reads) length: {prefix_length}")
            print(f"   Required output length: {ground_truth_length}")
            print(f"   Total required: {total_required}")
            print(f"   Block size: {block_size}")
            print(f"   Excess: {total_required - block_size} tokens")
        else:
            valid_inputs.append(inp)
            valid_gts.append(gt)
            valid_alignment_sizes.append(N)

    if skipped_count > 0:
        if pbar and skipped_count > 0:
            pbar.write(f"\n Batch summary: {len(valid_inputs)} processed, {skipped_count} skipped due to size")
        elif not pbar and skipped_count > 0:
            print(f"\n Batch summary: {len(valid_inputs)} processed, {skipped_count} skipped due to size")

    # If no valid examples remain, return empty results with skipped count
    if not valid_inputs:
        return [], skipped_count, []

    # Update variables to use filtered data
    inputs, gts, alignment_sizes = valid_inputs, valid_gts, valid_alignment_sizes


    # params for GPT_Inference (gpt path only)
    inf_params = {
        'model':  model,
        'ctx':    ctx,
        **cfg_dict,
        'device': device,
        'stoi':   stoi,
        'itos':   itos,
        'encode': encode_str,
        'decode': decode_tokens,
        'track_entropy': cfg_dict.get('track_entropy', False),
        'track_attention': cfg_dict.get('track_attention', False),
        'analyze_vote_confidence': cfg_dict.get('analyze_vote_confidence', False),
    }

    model_type  = cfg_dict.get('model_type', 'gpt')
    greedy      = cfg_dict.get('greedy', True)
    max_new_tok = cfg_dict.get('ground_truth_length', 110)

    # inference
    # Using torch.inference_mode() instead of torch.no_grad() for better performance:
    # - Disables autograd completely (no gradient computation or tracking)
    # - Disables view tracking (saves memory and compute)
    # - Disables version counter updates (less overhead)
    # Expected gain: 5-10% faster than torch.no_grad()
    with torch.inference_mode(), ctx:

        # Log beam search start for distributed debugging
        if not greedy and model_type == 'gpt' and 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
            print(f"Rank {rank}: Starting beam search for batch of {len(inputs)} examples")

        # GPT
        if model_type == 'gpt':
            # Synchronize GPU before timing
            if device.type == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            out   = GPT_Inference(inf_params).inference(
                        list(inputs), alignment_size=alignment_sizes
                    )

            # Synchronize GPU after inference
            if device.type == 'cuda':
                torch.cuda.synchronize()

            dt = time.perf_counter() - t0

            cands = out['candidate_sequences']
            # Extract entropy data if available
            batch_entropies = out.get('token_entropies', None)  # Shape: [batch_size, max_new_tokens] or None
            # Extract attention data if available
            batch_attention_data = out.get('attention_data', None)
            # Extract logits for vote confidence analysis
            batch_logits = out.get('logits', None)
            # Extract pure model inference time (excludes preprocessing/tracking)
            model_time = out.get('model_inference_time', None)

        # LSTM
        elif model_type == 'lstm':
            pad_id  = stoi['#']
            enc     = [torch.tensor(encode_str(s), device=device) for s in inputs]
            # Calculate actual sequence lengths (excluding padding)
            lengths = torch.tensor([(t != pad_id).sum().item() for t in enc], device=device)

            from torch.nn.utils.rnn import pad_sequence
            idx = pad_sequence(enc, batch_first=True, padding_value=pad_id)

            # Synchronize GPU before timing
            if device.type == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            ys = model.generate(
                    idx, lengths=lengths,
                    max_new_tokens=max_new_tok,
                    temperature=1.0 if greedy else cfg_dict.get('temperature', 1.0),
                    top_k=None if greedy else cfg_dict.get('top_k', 50),
                    eos_token_id=stoi.get('<eos>'),
                )

            # Synchronize GPU after inference
            if device.type == 'cuda':
                torch.cuda.synchronize()

            dt = time.perf_counter() - t0

            cands = [decode_tokens(row.tolist()) for row in ys]
            batch_entropies = None  # No entropy tracking for LSTM
            batch_attention_data = None  # No attention tracking for LSTM
            batch_logits = None  # No logits tracking for LSTM
            model_time = None  # LSTM doesn't separate preprocessing from model time


        # Mamba
        elif model_type == 'mamba':
            # Synchronize GPU before timing
            if device.type == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            cands = []
            for s in inputs:
                idx = torch.tensor(encode_str(s), device=device)[None, :]
                # use the lightweight sampler provided by the model
                # for now only supports single inference not batched
                ys  = model.my_generate(
                        idx,
                        max_new_tokens=max_new_tok,
                        temperature=1.0 if greedy else cfg_dict.get('temperature', 1.0),
                        top_k=None if greedy else cfg_dict.get('top_k', 50),
                    )
                cands.append(decode_tokens(ys[0].tolist()))

            # Synchronize GPU after inference
            if device.type == 'cuda':
                torch.cuda.synchronize()

            dt = time.perf_counter() - t0

            batch_entropies = None  # No entropy tracking for Mamba
            batch_attention_data = None  # No attention tracking for Mamba
            batch_logits = None  # No logits tracking for Mamba
            model_time = None  # Mamba doesn't separate preprocessing from model time

        else:
            raise ValueError(f"Unsupported model_type '{model_type}'")

    # metrics
    results = []
    all_position_metrics = []  # For vote confidence analysis
    # Get cross mode from config if available
    cross_mode = cfg_dict.get('cross_mode', None)
    analyze_vote_confidence = cfg_dict.get('analyze_vote_confidence', False)

    for i, (ex, gt, cand, N) in enumerate(zip(inputs, gts, cands, alignment_sizes)):
        pred_full = filter_string(cand)[:len(gt)]
        reads = ex.split(':', 1)[0]
        # N is now passed as alignment_size, which contains the true cluster size

        # Get entropy data for this example
        entropy_data = batch_entropies[i] if batch_entropies is not None and i < len(batch_entropies) else None

        # Get attention data for this example
        attention_data = batch_attention_data[i] if batch_attention_data is not None and i < len(batch_attention_data) else None

        # Save attention data if available
        if attention_data is not None and isinstance(attention_data, dict):
            attention_sequence = attention_data.get('attention')
            read_boundaries = attention_data.get('read_boundaries')

            # Debug logging
            if i == 0 and cfg_dict.get('track_attention', False):  # Only log for first example
                write_output(f"[Attention Debug] Example {i}:")
                write_output(f"  - attention_data keys: {attention_data.keys() if attention_data else 'None'}")
                write_output(f"  - attention_sequence: {'Present' if attention_sequence else 'None'} (len={len(attention_sequence) if attention_sequence else 0})")
                write_output(f"  - read_boundaries: {read_boundaries}")

            if attention_sequence and read_boundaries:
                # Calculate Levenshtein distance
                lev_dist = levenshtein_distance(pred_full, gt)

                # Check if this example is contaminated and save contaminated attention data
                if contamination_lookup and condition_name:
                    # Use provided example_indices if available, otherwise fall back to start + i
                    global_example_idx = example_indices[i] if example_indices is not None else start + i
                    if global_example_idx in contamination_lookup:
                        contamination_info = contamination_lookup[global_example_idx]
                        contaminated_positions = [pos['position'] for pos in contamination_info.get('contaminated_positions', [])]

                        if contaminated_positions:  # Only save if there are actual contaminated sequences
                            contaminated_attention_output_dir = (
                                cfg_dict.get('contaminated_attention_output_dir') or
                                '/workspaces/TReconLM/Contaminated_Attention_output'
                            )

                            save_contaminated_attention_data(
                                attention_sequence=attention_sequence,
                                read_boundaries=read_boundaries,
                                token_sequence=ex,
                                example_idx=global_example_idx,
                                cluster_size=N,
                                contaminated_positions=contaminated_positions,
                                contamination_info=contamination_info,
                                condition_name=condition_name,
                                output_dir=contaminated_attention_output_dir,
                                prediction=pred_full,
                                ground_truth=gt,
                                levenshtein_distance=lev_dist,
                                save_per_head_attention=cfg_dict.get('save_per_head_attention', False)
                            )
                else:
                    # Only save regular attention data when NOT in contamination mode
                    attention_output_dir = (
                        cfg_dict.get('attention_output_dir') or
                        cfg_dict.get('sampling', {}).get('attention_output_dir') or
                        '/workspaces/TReconLM/Attention_output'
                    )

                    # Use provided example_indices if available, otherwise fall back to start + i
                    example_idx = example_indices[i] if example_indices is not None else start + i
                    save_attention_data(
                        attention_sequence=attention_sequence,
                        read_boundaries=read_boundaries,
                        token_sequence=ex,
                        example_idx=example_idx,  # Use global index instead of batch-local index
                        cluster_size=N,
                        output_dir=attention_output_dir,
                        prediction=pred_full,
                        ground_truth=gt,
                        levenshtein_distance=lev_dist,
                        save_per_head_attention=cfg_dict.get('save_per_head_attention', False)
                    )

        # Collect vote confidence metrics if enabled
        if analyze_vote_confidence and batch_logits is not None:
            example_logits = batch_logits[i] if i < batch_logits.shape[0] else None
            if example_logits is not None:
                # For now, remove example_id from the call due to import issues
                position_metrics = collect_position_metrics(reads, gt, pred_full, example_logits, N)
                all_position_metrics.extend(position_metrics)

        # Compute metrics for both cropped and full sequences
        ham_full = hamming_distance_postprocessed(gt, pred_full)
        lev_full = levenshtein_distance(gt, pred_full) / len(gt)

        # Apply cross-evaluation truncation for cropped metrics
        if cross_mode == 'noisy':
            # For noisy: evaluate only first 110nt of 120nt concatenated sequences
            eval_length = min(110, len(gt))
            gt_cropped = gt[:eval_length]
            pred_cropped = pred_full[:eval_length]
        elif cross_mode == 'microsoft':
            # For microsoft: evaluate only first 60nt of 110nt sequences
            eval_length = min(60, len(gt))
            gt_cropped = gt[:eval_length]
            pred_cropped = pred_full[:eval_length]
        else:
            # Normal mode: no cropping needed
            gt_cropped = gt
            pred_cropped = pred_full
            eval_length = len(gt)

        ham_cropped = hamming_distance_postprocessed(gt_cropped, pred_cropped)
        lev_cropped = levenshtein_distance(gt_cropped, pred_cropped) / len(gt_cropped)

        # Calculate prefix length for display
        prefix_part = ex.split(':')[0]
        prefix_length = len(prefix_part)

        # Check if this is a subsampled example
        actual_reads_count = len(reads.split('|'))

        if N != actual_reads_count:
            write_output(f"Cluster size {N} (true), {actual_reads_count} reads (subsampled), Input length: {prefix_length}")
        else:
            write_output(f"Cluster size {N}, Input length: {prefix_length}")
        if cross_mode:
            write_output(f"[Cross-eval mode: {cross_mode}]")
            write_output(f"  Cropped evaluation (first {eval_length}nt): HAM={ham_cropped}, LEV={lev_cropped:.4f}")
            write_output(f"  Full sequence evaluation ({len(gt)}nt): HAM={ham_full}, LEV={lev_full:.4f}")
        else:
            # For normal mode, cropped and full are the same, but still show full metrics
            write_output(f"  Evaluation ({len(gt)}nt): HAM={ham_full}, LEV={lev_full:.4f}")
        # Show reads only for cluster sizes 9 and 10
        #if N in [9, 10]:
        #    reads_list = reads.split('|')
        #    write_output(f"[READS] {len(reads_list)} reads:")
        #    for i, read in enumerate(reads_list):
        #        # Truncate very long reads for display
        #        display_read = read[:60] + "..." if len(read) > 60 else read
        #        write_output(f"  R{i}: {display_read}")

        write_output(f"[GT]   {gt}")
        write_output(f"[PRED] {pred_full}")

        # Simple attention confirmation if tracking enabled
        if attention_data is not None and cfg_dict.get('track_attention', False):
            if isinstance(attention_data, dict) and attention_data.get('attention'):
                write_output(f"[ATTN] Attention data saved ({len(attention_data.get('attention', []))} tokens)")

        # High error detection for large clusters
        if lev_cropped > 0.3 and N >= 8:
            write_output("\n" + "="*80)
            write_output(f"  HIGH ERROR DETECTED - Cluster size {N}, Lev={lev_cropped:.4f}")

            prefix_part = ex.split(':')[0]
            prefix_length = len(prefix_part)
            required_space = prefix_length + 1 + len(gt)  # +1 for ':'

            write_output(f"Prefix (reads) length: {prefix_length} characters")
            write_output(f"Expected output length: {len(gt)}")
            write_output(f"Total required space: {required_space}")
            write_output(f"Actual prediction length: {len(pred_full)}")

            # Print ALL reads in the cluster
            reads_list = reads.split('|')
            write_output(f"\nAll {len(reads_list)} reads in cluster:")
            for i, read in enumerate(reads_list):
                write_output(f"  Read {i+1}: {read}")  # Show full read without truncation

            # Check if input might be truncated
            block_size = cfg_dict.get('block_size', 1500)
            if required_space > block_size - 10:
                write_output(f"WARNING: Required space ({required_space}) is close to or exceeds block_size ({block_size})")
                write_output("     This may cause truncation issues!")
            else:
                write_output(f"Space check: Required space ({required_space}) fits within block_size ({block_size})")

            write_output(f"\nFull ground truth: {gt}")
            write_output(f"Full prediction:   {pred_full}")
            write_output("="*80 + "\n")

        # Store both cropped and full metrics with entropy and attention data
        # Format: (N, reads, gt_cropped, pred_cropped, ham_cropped, lev_cropped,
        #          gt_full, pred_full, ham_full, lev_full, full_pipeline_time, model_only_time, token_entropies, attention_data)
        results.append((N, reads, gt_cropped, pred_cropped, ham_cropped, lev_cropped,
                       gt, pred_full, ham_full, lev_full, dt, model_time, entropy_data, attention_data))

    # Return both results and skipped count
    return results, skipped_count, all_position_metrics


def run_one_batch_with_majority_voting_batched(
    chunk, start, batch_size,
    cfg_dict, model, meta, device, ctx, pbar=None
):
    """
    BATCHED version of majority voting ensemble inference.

    Instead of processing each example individually with multiple sequential inference calls,
    this batches all permutations together for efficient GPU utilization.

    Strategy:
    - Take effective_batch_size = batch_size // max_permutations examples
    - Generate all permutations for all examples
    - Run ONE batched inference call with all permutations
    - Unpack results and apply majority voting per example

    Args:
        Same as run_one_batch_with_majority_voting

    Returns:
        Same as run_one_batch_with_majority_voting
    """
    from src.utils.permutation_utils import (
        generate_unique_permutations,
        permute_reads,
        positional_majority_vote,
        compute_pairwise_hamming_distances
    )

    # Extract majority voting parameters
    majority_cfg = cfg_dict.get('majority_voting', {})
    max_perms = majority_cfg.get('max_permutations', 10)
    seed = majority_cfg.get('seed', 42)
    tie_breaking_strategy = majority_cfg.get('tie_breaking_strategy', 'random')

    stoi, itos = meta['stoi'], meta['itos']
    decode_tokens = lambda t: ''.join(itos[i] for i in t)
    encode_str = lambda s: [stoi.get(ch, stoi.get('<unk>', 0)) for ch in s]

    # Get batch of examples (full batch_size)
    batch = chunk[start:start + batch_size]

    if not batch:
        return [], 0, []

    # Check if chunk contains true cluster sizes (3-tuple) or not (2-tuple)
    has_true_sizes = len(batch[0]) == 3

    # Step 1: Determine actual max cluster size in this batch
    max_cluster_size_in_batch = 0
    for example in batch:
        if has_true_sizes:
            _, _, cluster_size = example
        else:
            x_tensor, _ = example
            input_string = decode_tokens(x_tensor.tolist())
            if ':' not in input_string:
                continue
            reads_part = input_string.split(':', 1)[0]
            cluster_size = len(reads_part.split('|'))
        max_cluster_size_in_batch = max(max_cluster_size_in_batch, cluster_size)

    # Calculate actual max permutations for this batch
    import math
    actual_max_perms = min(max_perms, math.factorial(max_cluster_size_in_batch))

    # Calculate effective batch size based on ACTUAL max permutations in this batch
    # This optimizes GPU usage when cluster sizes are smaller
    effective_batch_size = max(1, batch_size // actual_max_perms)

    # Log optimization info
    test_mode = cfg_dict.get('majority_voting', {}).get('test_mode', False)
    if test_mode or (pbar and max_cluster_size_in_batch > 0):
        msg = (f"[BATCH] max_cluster_size={max_cluster_size_in_batch}, "
               f"actual_max_perms={actual_max_perms}, effective_batch_size={effective_batch_size}, "
               f"processing {min(len(batch), effective_batch_size)} examples, "
               f"approximately {actual_max_perms * min(len(batch), effective_batch_size)} total GPU permutations")
        if pbar:
            pbar.write(msg)
        elif test_mode:
            print(msg)

    # Get log file from config if available
    log_file = cfg_dict.get('log_file', None)

    def write_output(msg):
        if pbar:
            pbar.write(msg)
        else:
            print(msg)

        # Also write to log file if available
        if log_file is not None:
            try:
                log_file.write(msg + '\n')
            except Exception as e:
                if not hasattr(write_output, '_log_error_printed'):
                    print(f"Warning: Failed to write to log file: {e}")
                    write_output._log_error_printed = True

    # Process the full batch in mini-batches of effective_batch_size
    all_results = []
    total_skipped = 0
    all_position_metrics = []

    num_mini_batches = math.ceil(len(batch) / effective_batch_size)

    for mini_batch_idx in range(num_mini_batches):
        mini_batch_start = mini_batch_idx * effective_batch_size
        mini_batch_end = min(mini_batch_start + effective_batch_size, len(batch))
        mini_batch = batch[mini_batch_start:mini_batch_end]

        if test_mode:
            print(f"  [MINI-BATCH {mini_batch_idx+1}/{num_mini_batches}] Processing examples {mini_batch_start}-{mini_batch_end-1}")

        # Step 1: Generate all permutations for all examples in this mini-batch
        all_permuted_tensors = []
        example_metadata = []

        for example_idx, example in enumerate(mini_batch):
            if has_true_sizes:
                x_tensor, gt, cluster_size = example
            else:
                x_tensor, gt = example
                # Extract cluster size from input string
                input_string = decode_tokens(x_tensor.tolist())
                if ':' not in input_string:
                    continue
                reads_part = input_string.split(':', 1)[0]
                cluster_size = len(reads_part.split('|'))

            input_string = decode_tokens(x_tensor.tolist())

            # Skip if no ':' separator
            if ':' not in input_string:
                continue

            # Generate unique permutations for this cluster size
            perms = generate_unique_permutations(cluster_size, max_perms, seed + start + mini_batch_start + example_idx)
            num_perms = len(perms)

            # Create permuted inputs for all permutations of this example
            for perm_idx, perm in enumerate(perms):
                # Permute reads in input string
                try:
                    permuted_input = permute_reads(input_string, perm)
                except (ValueError, IndexError) as e:
                    write_output(f"Warning: Failed to permute reads for example {example_idx}: {e}")
                    continue

                # Encode permuted input
                encoded_input = encode_str(permuted_input)
                permuted_tensor = torch.tensor(encoded_input, dtype=torch.long)

                all_permuted_tensors.append(permuted_tensor)
                example_metadata.append({
                    'example_idx': example_idx,
                    'perm_idx': perm_idx,
                    'num_perms': num_perms,
                    'gt': gt,
                    'cluster_size': cluster_size,
                    'original_input': input_string
                })

        if not all_permuted_tensors:
            continue  # Skip this mini-batch if no valid examples

        # Step 2: Run batched inference on ALL permutations in this mini-batch at once
        # Create a temporary chunk with all permuted examples
        if has_true_sizes:
            temp_chunk = [(tensor, meta['gt'], meta['cluster_size'])
                          for tensor, meta in zip(all_permuted_tensors, example_metadata)]
        else:
            temp_chunk = [(tensor, meta['gt'])
                          for tensor, meta in zip(all_permuted_tensors, example_metadata)]

        # Run inference with the full batch of permutations
        # batch_size = len(temp_chunk) to process all permutations in one call
        if test_mode:
            print(f"    Running batched GPU inference on {len(temp_chunk)} permutations...")

        perm_results, perm_skipped, _ = run_one_batch(
            temp_chunk, 0, len(temp_chunk),
            cfg_dict, model, meta, device, ctx, pbar=None
        )

        if test_mode:
            print(f"    Received {len(perm_results)} results from GPU")

        if len(perm_results) != len(all_permuted_tensors):
            write_output(f"Warning: Expected {len(all_permuted_tensors)} results but got {len(perm_results)}")

        # Step 3: Unpack results and group by original example
        from collections import defaultdict
        results_by_example = defaultdict(lambda: {
            'predictions': [],
            'perm_results': [],
            'metadata': None
        })

        for result, metadata in zip(perm_results, example_metadata):
            example_idx = metadata['example_idx']
            # Extract prediction from result
            # Result format: (N, reads, gt_cropped, pred_cropped, ham_cropped, lev_cropped,
            #                 gt_full, pred_full, ham_full, lev_full, dt, model_time, entropy_data, attention_data)
            prediction = result[7]  # pred_full
            results_by_example[example_idx]['predictions'].append(prediction)
            results_by_example[example_idx]['perm_results'].append(result)

            # Store metadata (same for all permutations of an example)
            if results_by_example[example_idx]['metadata'] is None:
                results_by_example[example_idx]['metadata'] = metadata

        # Step 4: Apply majority voting for each example in this mini-batch
        total_skipped += perm_skipped

        for example_idx in sorted(results_by_example.keys()):
            example_data = results_by_example[example_idx]
            predictions = example_data['predictions']
            perm_results = example_data['perm_results']
            metadata = example_data['metadata']

            if not predictions:
                write_output(f"Warning: No valid predictions for example {example_idx}, skipping")
                total_skipped += 1
                continue

            # Get ground truth and other info from metadata
            gt = metadata['gt']
            cluster_size = metadata['cluster_size']
            num_perms = metadata['num_perms']

            # Perform majority voting
            voted_pred, vote_stats = positional_majority_vote(predictions, tie_breaking_strategy=tie_breaking_strategy)

            # Compute diversity metrics
            diversity = compute_pairwise_hamming_distances(predictions)

            # Compute metrics for voted prediction
            from Levenshtein import distance as levenshtein_distance
            from src.utils.hamming_distance import hamming_distance_postprocessed

            voted_hamming = hamming_distance_postprocessed(gt, voted_pred)
            voted_lev = levenshtein_distance(gt, voted_pred) / len(gt) if len(gt) > 0 else 0.0
            voted_failed = 1 if voted_lev > 0 else 0

            # Get first (baseline) prediction metrics
            first_result = perm_results[0]
            first_pred = predictions[0]
            first_lev = first_result[5]  # lev_cropped
            first_failed = 1 if first_lev > 0 else 0

            # Compute improvement
            lev_improvement = first_lev - voted_lev
            failure_rescued = 1 if (first_failed == 1 and voted_failed == 0) else 0

            # Log per-example statistics
            write_output(f"Cluster size {cluster_size} ({num_perms} permutations)")
            write_output(f"  First prediction:  LEV={first_lev:.4f}, {'FAILED' if first_failed else 'SUCCESS'}")
            write_output(f"  Voted prediction:  LEV={voted_lev:.4f}, {'FAILED' if voted_failed else 'SUCCESS'}")
            if lev_improvement != 0:
                rel_improvement = (lev_improvement / first_lev * 100) if first_lev > 0 else 0
                write_output(f"  Improvement: {lev_improvement:+.4f} ({rel_improvement:+.1f}%)")
            if failure_rescued:
                write_output(f"  RESCUED FAILURE")
            write_output(f"  Predictions differ by {diversity['mean_pairwise_hamming']:.2f} nucleotides on average (pairwise Hamming)")
            write_output(f"  Vote agreement: mean={vote_stats['mean_agreement']:.3f}, min={vote_stats['min_agreement']:.3f}")
            write_output(f"[GT]    {gt}")

            # Print all individual predictions from each permutation
            for perm_idx, pred in enumerate(predictions):
                write_output(f"[PERM{perm_idx}] {pred}")

            write_output(f"[VOTED] {voted_pred}")
            write_output("")

            # Create result tuple with extended information
            reads_part = metadata['original_input'].split(':', 1)[0]
            extended_result = (
                cluster_size,                           # 0: N
                reads_part,                             # 1: reads
                gt,                                     # 2: gt_cropped (same as gt_full in normal mode)
                voted_pred,                             # 3: pred_cropped (voted)
                voted_hamming,                          # 4: ham_cropped (voted)
                voted_lev,                              # 5: lev_cropped (voted)
                gt,                                     # 6: gt_full
                voted_pred,                             # 7: pred_full (voted)
                voted_hamming,                          # 8: ham_full (voted)
                voted_lev,                              # 9: lev_full (voted)
                first_result[10],                       # 10: dt (inference time)
                first_result[11],                       # 11: model_time
                first_result[12],                       # 12: entropy_data
                first_result[13],                       # 13: attention_data
                # Majority voting specific stats:
                num_perms,                              # 14: num_permutations
                first_lev,                              # 15: first_levenshtein
                lev_improvement,                        # 16: levenshtein_improvement
                first_failed,                           # 17: first_failed
                voted_failed,                           # 18: voted_failed
                failure_rescued,                        # 19: failure_rescued
                diversity['mean_pairwise_hamming'],     # 20: diversity_mean_hamming
                diversity['std_pairwise_hamming'],      # 21: diversity_std_hamming
                vote_stats['mean_agreement'],           # 22: vote_mean_agreement
                vote_stats['min_agreement'],            # 23: vote_min_agreement
            )

            all_results.append(extended_result)

    # Print summary in test mode
    if test_mode and len(all_results) > 0:
        print(f"\n[SUMMARY] Processed {len(all_results)} examples total across {num_mini_batches} mini-batches")
        print(f"[SUMMARY] Skipped {total_skipped} examples")
        # Calculate average cluster size and permutations
        cluster_sizes = [r[0] for r in all_results]
        num_perms_list = [r[14] for r in all_results]
        print(f"[SUMMARY] Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.1f}")
        print(f"[SUMMARY] Permutations per example: min={min(num_perms_list)}, max={max(num_perms_list)}, avg={np.mean(num_perms_list):.1f}")

    return all_results, total_skipped, all_position_metrics


def run_one_batch_with_majority_voting(
    chunk, start, batch_size,
    cfg_dict, model, meta, device, ctx, pbar=None
):
    """
    Wrapper for run_one_batch that implements majority voting ensemble inference.

    Uses batched inference for all permutations to maximize GPU utilization.

    For each example:
    1. Generate N unique permutations of read order (N = min(max_perms, cluster_size!))
    2. Run batched inference on all permutations together
    3. Perform positional majority voting across predictions
    4. Compute diversity metrics and improvement vs single inference

    Args:
        Same as run_one_batch

    Returns:
        Same as run_one_batch, but results include majority voting statistics
    """
    # Check if majority voting is enabled
    majority_cfg = cfg_dict.get('majority_voting', {})
    if not majority_cfg.get('enabled', False):
        # Fall back to standard single inference
        return run_one_batch(chunk, start, batch_size, cfg_dict, model, meta, device, ctx, pbar)

    # Always use batched inference
    return run_one_batch_with_majority_voting_batched(
        chunk, start, batch_size, cfg_dict, model, meta, device, ctx, pbar
    )


def aggregate_entropy_by_cluster_size(results_list):
    """
    Aggregate entropy statistics by cluster size from inference results.
    """
    entropy_by_size = defaultdict(list)

    for result in results_list:
        if isinstance(result, tuple) and len(result) >= 11:
            # Extended result format with entropy data
            # Format: (N, reads, gt_cropped, pred_cropped, ham_cropped, lev_cropped,
            #          gt_full, pred_full, ham_full, lev_full, time, token_entropies)
            cluster_size = result[0]
            token_entropies = result[11] if len(result) > 11 else None

            if token_entropies is not None:
                # token_entropies is a single example's entropy: shape [max_new_tokens]
                mean_entropy = float(np.mean(token_entropies))
                entropy_by_size[cluster_size].append(mean_entropy)

    # Compute statistics
    entropy_stats = {}
    for size in sorted(entropy_by_size.keys()):
        entropies = entropy_by_size[size]
        if entropies:
            entropy_stats[size] = {
                'mean': float(np.mean(entropies)),
                'std': float(np.std(entropies)),
                'median': float(np.median(entropies)),
                'min': float(np.min(entropies)),
                'max': float(np.max(entropies)),
                'count': len(entropies)
            }

    return entropy_stats


def analyze_detailed_attention_for_token(attention_weights, read_boundaries, token_position, threshold=0.9):
    """
    Analyze attention for a single generated token position.

    Note: The attention weights show what the model attended to when PROCESSING
    this token position (not when generating its logits). This reveals which
    reads the model focuses on when encoding each generated position.

    Args:
        attention_weights: [B, num_heads, 1, seq_len] attention matrix for this token
        read_boundaries: [(start, end), ...] boundaries of each read
        token_position: which generated token we're analyzing (for reporting)
        threshold: attention threshold (0.9 = 90%)

    Returns:
        dict with detailed attention breakdown
    """
    # Convert to numpy if it's a tensor and squeeze batch dimension
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.cpu().numpy()

    # Handle batch dimension - take first example
    if attention_weights.ndim == 4:  # [B, num_heads, 1, seq_len] for autoregressive generation
        attention_weights = attention_weights[0]  # [num_heads, 1, seq_len]

    # Average attention across heads for this token (squeeze the 1-dim)
    if attention_weights.ndim == 3:  # [num_heads, 1, seq_len]
        token_attention = attention_weights[:, 0, :].mean(axis=0)  # [seq_len]
    else:  # Fallback for other shapes
        token_attention = attention_weights.mean(axis=0)  # [seq_len]

    # Calculate attention to each read
    read_attentions = []
    read_position_details = []

    for read_idx, (start, end) in enumerate(read_boundaries):
        # Total attention to this read
        read_attention = token_attention[start:end+1].sum().item()
        read_attentions.append(read_attention)

        # Which positions within this read got most attention
        read_positions = token_attention[start:end+1]
        if read_attention > 0:
            # Normalize positions within this read
            normalized_positions = read_positions / read_attention
            # Find positions that get significant attention within this read
            significant_positions = []
            for pos_idx, pos_attention in enumerate(normalized_positions):
                if pos_attention > 0.1:  # 10% threshold within read
                    significant_positions.append({
                        'relative_position': pos_idx,
                        'absolute_position': start + pos_idx,
                        'attention_weight': pos_attention.item()
                    })
            read_position_details.append(significant_positions)
        else:
            read_position_details.append([])

    # Normalize read attentions to percentages
    total_attention = sum(read_attentions)
    if total_attention > 0:
        read_percentages = [att / total_attention for att in read_attentions]
    else:
        read_percentages = [0.0] * len(read_boundaries)

    # Find reads that get 90% of attention
    sorted_reads = sorted(enumerate(read_percentages), key=lambda x: x[1], reverse=True)
    cumulative_attention = 0
    top_reads = []

    for read_idx, percentage in sorted_reads:
        cumulative_attention += percentage
        top_reads.append({
            'read_index': read_idx,
            'attention_percentage': percentage,
            'position_details': read_position_details[read_idx]
        })
        if cumulative_attention >= threshold:
            break

    return {
        'token_position': token_position,
        'total_reads': len(read_boundaries),
        'top_reads_90pct': top_reads,
        'num_reads_for_90pct': len(top_reads),
        'attention_concentration': max(read_percentages),
        'attention_entropy': -sum(p * np.log(p + 1e-10) for p in read_percentages if p > 0),
        'all_read_percentages': read_percentages
    }


def analyze_sequence_attention_detailed(attention_sequence, read_boundaries, ground_truth):
    """
    Analyze attention for entire generated sequence.

    Args:
        attention_sequence: List of attention weights for each generated token
        read_boundaries: Read boundary information
        ground_truth: Ground truth sequence for length reference

    Returns:
        Detailed attention analysis for the sequence
    """
    position_analyses = []

    for token_pos, attention_weights in enumerate(attention_sequence):
        if attention_weights is not None and token_pos < len(ground_truth):
            analysis = analyze_detailed_attention_for_token(
                attention_weights, read_boundaries, token_pos
            )
            position_analyses.append(analysis)

    # Calculate aggregate statistics
    if position_analyses:
        avg_reads_for_90pct = np.mean([a['num_reads_for_90pct'] for a in position_analyses])
        normalized_reads_90pct = avg_reads_for_90pct / len(read_boundaries)

        avg_concentration = np.mean([a['attention_concentration'] for a in position_analyses])
        avg_entropy = np.mean([a['attention_entropy'] for a in position_analyses])

        # Read usage consistency
        all_read_usage = np.array([a['all_read_percentages'] for a in position_analyses])
        read_usage_std = np.std(all_read_usage, axis=0)  # Std across positions for each read
        read_consistency = 1.0 - np.mean(read_usage_std)  # Higher = more consistent

        aggregate_stats = {
            'avg_reads_for_90pct_attention': avg_reads_for_90pct,
            'normalized_reads_90pct': normalized_reads_90pct,
            'avg_attention_concentration': avg_concentration,
            'avg_attention_entropy': avg_entropy,
            'read_usage_consistency': read_consistency,
            'total_positions_analyzed': len(position_analyses)
        }
    else:
        aggregate_stats = None

    return {
        'position_by_position': position_analyses,
        'aggregate_stats': aggregate_stats,
        'cluster_size': len(read_boundaries)
    }


def aggregate_detailed_attention_by_cluster_size(results_list):
    """
    Aggregate detailed attention analyses by cluster size.
    """
    attention_by_size = defaultdict(list)

    for result in results_list:
        if isinstance(result, tuple) and len(result) >= 12:
            cluster_size = result[0]
            attention_data = result[12] if len(result) > 12 else None  # 13th element

            if attention_data is not None:
                # attention_data contains the raw attention sequence and read boundaries
                # We need to process it to get detailed analysis
                ground_truth = result[6]  # Full ground truth

                # Process attention data for this example
                # attention_data is a single dictionary, not a list
                if isinstance(attention_data, dict):
                    attention_sequence = attention_data.get('attention')
                    read_boundaries = attention_data.get('read_boundaries')

                    if attention_sequence and read_boundaries:
                        detailed_analysis = analyze_sequence_attention_detailed(
                            attention_sequence, read_boundaries, ground_truth
                        )
                        attention_by_size[cluster_size].append(detailed_analysis)

    return attention_by_size


def save_attention_data(attention_sequence, read_boundaries, token_sequence, example_idx, cluster_size,
                       output_dir="Attention_output", prediction=None, ground_truth=None, levenshtein_distance=None,
                       save_per_head_attention=False):
    """
    Save attention data to file for later visualization.

    Args:
        attention_sequence: List of attention tensors [num_heads, 1, seq_len] for each generated token
        read_boundaries: List of (start, end) positions for each read
        token_sequence: The input token sequence (reads:ground_truth format)
        example_idx: Index of the example
        cluster_size: Number of reads in the cluster
        output_dir: Directory to save attention files
        prediction: The model's prediction
        ground_truth: The ground truth sequence
        levenshtein_distance: The Levenshtein distance between prediction and ground truth
        save_per_head_attention: If True, save per-head attention [num_tokens, num_heads, seq_len];
                                 If False, save averaged attention [num_tokens, seq_len] (default)
    """
    if not attention_sequence or not read_boundaries:
        return

    # Create cluster-specific subdirectory
    cluster_dir = os.path.join(output_dir, f"cluster_size_{cluster_size}")
    os.makedirs(cluster_dir, exist_ok=True)

    # Process attention data
    # Stack all attention tensors and average over heads
    try:
        # Check if attention_sequence is dict (all layers) or list (last layer only)
        if isinstance(attention_sequence, dict):
            # Multiple layers stored - process each layer
            num_layers = len(attention_sequence)
            print(f"Processing {num_layers} layers of attention data...")

            # Process each layer separately
            layer_attention_data = {}
            for layer_idx, attention_list in attention_sequence.items():
                attention_tensors = []
                max_seq_len = 0

                for attn in attention_list:
                    if hasattr(attn, 'cpu'):
                        tensor = attn.cpu()
                    else:
                        tensor = torch.tensor(attn)
                    attention_tensors.append(tensor)
                    max_seq_len = max(max_seq_len, tensor.shape[-1])

                if attention_tensors:
                    # Pad and stack for this layer
                    padded_tensors = []
                    for tensor in attention_tensors:
                        current_len = tensor.shape[-1]
                        if current_len < max_seq_len:
                            pad_size = max_seq_len - current_len
                            padded = torch.nn.functional.pad(tensor, (0, pad_size), mode='constant', value=0)
                            padded_tensors.append(padded)
                        else:
                            padded_tensors.append(tensor)

                    stacked_attention = torch.stack(padded_tensors, dim=0).squeeze(1).squeeze(2)
                    # stacked_attention shape: [num_tokens, num_heads, seq_len]

                    if save_per_head_attention:
                        # Keep per-head attention: [num_tokens, num_heads, seq_len]
                        layer_attention_data[f'layer_{layer_idx}'] = stacked_attention
                    else:
                        # Average over heads: [num_tokens, seq_len]
                        averaged_attention = stacked_attention.mean(dim=1)
                        layer_attention_data[f'layer_{layer_idx}'] = averaged_attention

            # Save all layers in structured format
            if layer_attention_data:
                colon_pos = token_sequence.index(':') if ':' in token_sequence else len(token_sequence)
                read_ends = [end for start, end in read_boundaries if end <= colon_pos]

                # Get num_heads from the first layer's data shape
                first_layer_data = next(iter(layer_attention_data.values()))
                num_heads = first_layer_data.shape[1] if save_per_head_attention and len(first_layer_data.shape) == 3 else None

                save_data = {
                    'attention_by_layer': layer_attention_data,  # Dict of {layer_0: tensor, layer_1: tensor, ...}
                    'token_sequence': token_sequence,
                    'read_ends': read_ends,
                    'cluster_size': cluster_size,
                    'example_idx': example_idx,
                    'colon_pos': colon_pos,
                    'read_boundaries': read_boundaries,
                    'prediction': prediction,
                    'ground_truth': ground_truth,
                    'levenshtein_distance': levenshtein_distance,
                    'num_layers': num_layers,
                    'per_head': save_per_head_attention,  # Flag indicating data format
                    'num_heads': num_heads  # Number of attention heads (only if per_head=True)
                }

                filename = f'attention_sample_{example_idx}.pt'
                filepath = os.path.join(cluster_dir, filename)
                torch.save(save_data, filepath)
                print(f"Saved {num_layers}-layer attention data to: {filepath}")
                return

        # Single layer case (original code path)
        attention_list = attention_sequence

        # Convert tensors to CPU and handle variable sequence lengths
        attention_tensors = []
        max_seq_len = 0

        for attn in attention_list:
            if hasattr(attn, 'cpu'):
                tensor = attn.cpu()
            else:
                tensor = torch.tensor(attn)
            attention_tensors.append(tensor)
            # Track maximum sequence length: tensor shape is [1, num_heads, 1, seq_len]
            max_seq_len = max(max_seq_len, tensor.shape[-1])

        if attention_tensors:
            # Pad tensors to same length and stack
            padded_tensors = []
            for tensor in attention_tensors:
                current_len = tensor.shape[-1]
                if current_len < max_seq_len:
                    # Pad the last dimension to max_seq_len
                    pad_size = max_seq_len - current_len
                    # Pad with zeros: (pad_left, pad_right) for last dim
                    padded = torch.nn.functional.pad(tensor, (0, pad_size), mode='constant', value=0)
                    padded_tensors.append(padded)
                else:
                    padded_tensors.append(tensor)

            # Stack: [num_generated_tokens, 1, num_heads, 1, seq_len] -> squeeze -> [num_generated_tokens, num_heads, seq_len]
            stacked_attention = torch.stack(padded_tensors, dim=0).squeeze(1).squeeze(2)

            if save_per_head_attention:
                # Keep per-head attention: [num_generated_tokens, num_heads, seq_len]
                attention_to_save = stacked_attention
                num_heads = stacked_attention.shape[1]
            else:
                # Average over heads: [num_generated_tokens, seq_len]
                attention_to_save = stacked_attention.mean(dim=1)
                num_heads = None

            # Find colon position to separate reads from ground truth
            colon_pos = token_sequence.index(':') if ':' in token_sequence else len(token_sequence)

            # Extract read end positions (convert boundaries to end positions)
            read_ends = [end for start, end in read_boundaries if end <= colon_pos]

            # Prepare data for saving
            save_data = {
                'normalized_attention': attention_to_save,
                'token_sequence': token_sequence,
                'read_ends': read_ends,
                'cluster_size': cluster_size,
                'example_idx': example_idx,
                'colon_pos': colon_pos,
                'read_boundaries': read_boundaries,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'levenshtein_distance': levenshtein_distance,
                'per_head': save_per_head_attention,  # Flag indicating data format
                'num_heads': num_heads  # Number of attention heads (only if per_head=True)
            }

            # Save to cluster-specific directory
            filename = f'attention_sample_{example_idx}.pt'
            filepath = os.path.join(cluster_dir, filename)
            torch.save(save_data, filepath)
            print(f"Saved attention data to: {filepath}")

    except Exception as e:
        print(f"Warning: Could not save attention data for example {example_idx}: {e}")



def save_sequences_to_file(all_results, output_file):
    """
    Save ground truth and prediction sequences to a TSV file for failure mode analysis.

    Args:
        all_results: List of result tuples from inference
        output_file: Path to output TSV file
    """
    import csv

    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')

            # Write header
            writer.writerow([
                'cluster_size', 'ground_truth', 'prediction',
                'ham_distance', 'lev_distance', 'sequence_length', 'noisy_traces'
            ])

            # Write data
            for result in all_results:
                # Format: (N, reads, gt_cropped, pred_cropped, ham_cropped, lev_cropped,
                #          gt_full, pred_full, ham_full, lev_full, time, token_entropies, attention_data)
                cluster_size = result[0]
                reads = result[1]  # Noisy traces
                gt_full = result[6]  # Use full ground truth
                pred_full = result[7]  # Use full prediction
                ham_full = result[8]
                lev_full = result[9]

                # Convert reads list to pipe-separated string
                noisy_traces_str = '|'.join(reads) if reads else ''

                writer.writerow([
                    cluster_size,
                    gt_full,
                    pred_full,
                    ham_full,
                    lev_full,
                    len(gt_full),
                    noisy_traces_str
                ])

        print(f"Successfully saved {len(all_results)} sequences to {output_file}")

    except Exception as e:
        print(f"Warning: Could not save sequences to {output_file}: {e}")


def print_detailed_attention_for_example(attention_analysis, ground_truth, prediction, example_idx):
    """
    Print detailed attention analysis for a single example.
    """
    print(f"\n" + "="*100)
    print(f"DETAILED ATTENTION ANALYSIS - Example {example_idx}")
    print(f"Cluster Size: {attention_analysis['cluster_size']}")
    print(f"="*100)

    # Print aggregate stats first
    if attention_analysis['aggregate_stats']:
        stats = attention_analysis['aggregate_stats']
        print(f"\nAGGREGATE STATISTICS:")
        print(f"  Average reads for 90% attention: {stats['avg_reads_for_90pct_attention']:.2f}")
        print(f"  Normalized by cluster size: {stats['normalized_reads_90pct']:.3f}")
        print(f"  Average attention concentration: {stats['avg_attention_concentration']:.3f}")
        print(f"  Average attention entropy: {stats['avg_attention_entropy']:.3f}")
        print(f"  Read usage consistency: {stats['read_usage_consistency']:.3f}")

    # Print position-by-position analysis
    print(f"\nPOSITION-BY-POSITION ATTENTION:")
    print(f"{'Pos':<4} {'GT':<4} {'Pred':<4} {'#Reads':<7} {'Top Reads (90% attention)':<50}")
    print("-"*100)

    position_analyses = attention_analysis['position_by_position']

    for i, analysis in enumerate(position_analyses):
        if i >= len(ground_truth):
            break

        gt_char = ground_truth[i] if i < len(ground_truth) else '?'
        pred_char = prediction[i] if i < len(prediction) else '?'
        num_reads = analysis['num_reads_for_90pct']

        # Format top reads info
        top_reads_info = []
        for read_info in analysis['top_reads_90pct']:
            read_idx = read_info['read_index']
            percentage = read_info['attention_percentage']

            # Get top positions within this read
            top_positions = sorted(read_info['position_details'],
                                 key=lambda x: x['attention_weight'], reverse=True)[:3]

            if top_positions:
                pos_str = ",".join([f"{p['relative_position']}" for p in top_positions[:2]])
                read_str = f"R{read_idx}({percentage:.2f}%@{pos_str})"
            else:
                read_str = f"R{read_idx}({percentage:.2f}%)"

            top_reads_info.append(read_str)

        top_reads_display = " ".join(top_reads_info[:4])  # Show top 4 reads max

        print(f"{i:<4} {gt_char:<4} {pred_char:<4} {num_reads:<7} {top_reads_display:<50}")

    print(f"\nLegend: R{{read_idx}}({{percentage}}%@{{positions}}) = Read index (attention %) @ top positions within read")


def aggregate_attention_stats_by_cluster_size(all_attention_analyses):
    """
    Aggregate detailed attention statistics by cluster size.
    """
    stats_by_size = defaultdict(list)

    for analysis in all_attention_analyses:
        if analysis and analysis['aggregate_stats']:
            cluster_size = analysis['cluster_size']
            stats = analysis['aggregate_stats']
            stats_by_size[cluster_size].append(stats)

    # Calculate means for each cluster size
    summary_stats = {}
    for size, stats_list in stats_by_size.items():
        if stats_list:
            summary_stats[size] = {
                'count': len(stats_list),
                'avg_reads_for_90pct': np.mean([s['avg_reads_for_90pct_attention'] for s in stats_list]),
                'std_reads_for_90pct': np.std([s['avg_reads_for_90pct_attention'] for s in stats_list]),
                'avg_normalized_reads': np.mean([s['normalized_reads_90pct'] for s in stats_list]),
                'avg_concentration': np.mean([s['avg_attention_concentration'] for s in stats_list]),
                'avg_entropy': np.mean([s['avg_attention_entropy'] for s in stats_list]),
                'avg_consistency': np.mean([s['read_usage_consistency'] for s in stats_list])
            }

    return summary_stats


def print_attention_comparison_summary(summary_stats):
    """
    Print summary comparison across cluster sizes.
    """
    print(f"\n" + "="*90)
    print("ATTENTION SUMMARY BY CLUSTER SIZE")
    print("="*90)

    print(f"{'Size':<6} {'Count':<8} {'Avg Reads':<12} {'Normalized':<12} {'Entropy':<10} {'Consistency':<12}")
    print("-"*90)

    sizes = sorted(summary_stats.keys())
    for size in sizes:
        stats = summary_stats[size]
        print(f"{size:<6} {stats['count']:<8} "
              f"{stats['avg_reads_for_90pct']:.2f}±{stats['std_reads_for_90pct']:.2f}  "
              f"{stats['avg_normalized_reads']:<12.3f} "
              f"{stats['avg_entropy']:<10.3f} "
              f"{stats['avg_consistency']:<12.3f}")

    # Analysis
    print(f"\n" + "="*90)
    print("INTERPRETATION:")
    print("="*90)

    for size in sizes:
        stats = summary_stats[size]
        normalized_reads = stats['avg_normalized_reads']
        consistency = stats['avg_consistency']

        print(f"\nCluster Size {size}:")

        if normalized_reads > 0.8:
            print(f"  Uses most reads ({normalized_reads:.1%}), good consensus seeking")
        elif normalized_reads > 0.5:
            print(f"  Uses majority of reads ({normalized_reads:.1%}), moderate focus")
        else:
            print(f"  Uses few reads ({normalized_reads:.1%}), potentially problematic bias")

        if consistency > 0.7:
            print(f"  High consistency ({consistency:.3f}), stable attention patterns")
        elif consistency > 0.5:
            print(f"  Moderate consistency ({consistency:.3f}), some variation")
        else:
            print(f"  Low consistency ({consistency:.3f}), erratic attention patterns")


def log_entropy_analysis(entropy_stats, use_wandb=True):
    """
    Log and visualize entropy analysis results.
    """
    if not entropy_stats:
        print("No entropy data to analyze")
        return

    # Print to console
    print("\n" + "=" * 70)
    print("ENTROPY ANALYSIS BY CLUSTER SIZE")
    print("=" * 70)
    print(f"{'Size':<8} {'Count':<8} {'Mean±Std':<20} {'Median':<10} {'Min-Max':<15}")
    print("-" * 70)

    sizes = sorted(entropy_stats.keys())
    for size in sizes:
        stats = entropy_stats[size]
        print(f"{size:<8} {stats['count']:<8} "
              f"{stats['mean']:.3f}±{stats['std']:.3f}  "
              f"{stats['median']:<10.3f} "
              f"{stats['min']:.3f}-{stats['max']:.3f}")

    # Calculate correlation if enough data
    if len(sizes) > 2:
        means = [entropy_stats[s]['mean'] for s in sizes]
        correlation = np.corrcoef(sizes, means)[0, 1]
        print(f"\nCorrelation (cluster size vs mean entropy): {correlation:.3f}")

        if correlation > 0.5:
            print("Strong positive correlation: Entropy increases with cluster size")
            print("  This supports the hypothesis of confusion/blending on large clusters")
        elif correlation > 0.3:
            print("Moderate positive correlation detected")
        else:
            print("Weak or no correlation detected")

        # Log to W&B if available
        if use_wandb and wandb.run is not None:
            # Create data for plotting
            table_data = []
            for size in sizes:
                stats = entropy_stats[size]
                table_data.append([
                    size,
                    stats['mean'],
                    stats['std'],
                    stats['median'],
                    stats['count']
                ])

            # Log table and correlation
            table = wandb.Table(
                columns=["cluster_size", "mean_entropy", "std_entropy", "median_entropy", "count"],
                data=table_data
            )

            wandb.log({
                "entropy_by_cluster_size_table": table,
                "entropy_stats_dict": entropy_stats,
                "entropy_correlation": correlation
            })

            print("Entropy analysis logged to Weights & Biases")

    return correlation if len(sizes) > 2 else None





def run_timing_measurement(cfg, model, meta, device, ctx, all_data):
    """
    Run throughput measurement by cycling through dataset for fixed time windows.

    This function measures pure model inference throughput without the overhead of:
    - Distance metrics computation (Hamming, Levenshtein)
    - Contamination analysis
    - Position-wise tracking
    - File I/O and logging

    The data is already sorted by input length (from line 2051) for efficient batching.
    When cycling, we maintain this sorted order within each cycle iteration.

    Args:
        cfg: Hydra config
        model, meta, device, ctx: Model and inference setup
        all_data: List of (x_tensor, ground_truth, cluster_size) tuples (sorted by length)

    Note: Currently only supports GPT model type.
    """
    from itertools import cycle

    timing_cfg = cfg.get('timing', {})
    run_duration = timing_cfg.get('run_duration', 900)  # 15 min default
    num_runs = timing_cfg.get('num_runs', 5)
    warmup_runs = timing_cfg.get('warmup_runs', 1)
    batch_size = cfg.data.batch_size  # Use data config batch size
    log_interval = timing_cfg.get('log_interval', 100)

    model_type = cfg.model.model_type

    # Validate model type
    if model_type != 'gpt':
        print(f"\nError: Timing mode only supports GPT model (got: {model_type})")
        print("Exiting without timing measurement.")
        return

    print_section_header("TIMING MODE: THROUGHPUT MEASUREMENT", width=80)
    print(f"Configuration:")
    print(f"  Model type: {model_type}")
    print(f"  Run duration: {run_duration / 60:.1f} minutes ({run_duration}s)")
    print(f"  Number of runs: {num_runs} ({warmup_runs} warmup + {num_runs - warmup_runs} measured)")
    print(f"  Batch size: {batch_size}")
    print(f"  Dataset size: {len(all_data)} examples (sorted by input length)")
    print(f"  Device: {device}")
    print(f"")

    # Build inference config dict
    stoi, itos = meta['stoi'], meta['itos']
    decode_tokens = lambda t: ''.join(itos[i] for i in t)
    encode_str = lambda s: [stoi.get(ch, stoi.get('<unk>', 0)) for ch in s]

    # Set up sampling parameters
    sampling_dict = extract_sampling_config(cfg)
    sampling_dict['track_attention'] = cfg.model.sampling.get('track_attention', False)
    sampling_dict['track_all_layers'] = cfg.model.sampling.get('track_all_layers', False)
    sampling_dict['save_per_head_attention'] = cfg.model.sampling.get('save_per_head_attention', False)

    # Check actual unpadded input lengths (after removing # padding and before : separator)
    def get_unpadded_input_length(item):
        """Get length of actual input (after removing padding and extracting input part)"""
        decoded = decode_tokens(item[0].tolist())
        unpadded = decoded.split('#', 1)[0]  # Remove padding
        input_only = unpadded.split(':', 1)[0]  # Extract input part (before ground truth)
        return len(input_only)

    all_lengths = [get_unpadded_input_length(item) for item in all_data]
    min_length = min(all_lengths)
    max_length = max(all_lengths)
    avg_length = sum(all_lengths) / len(all_lengths)

    print(f"Dataset: {len(all_data)} examples")
    print(f"Unpadded input lengths: min={min_length}, max={max_length}, avg={avg_length:.1f}")

    # Create representative subset for timing
    # Sample evenly across observation_size (cluster size) for realistic throughput
    print(f"\nCreating representative timing subset:")

    import random
    timing_seed = cfg.get('timing', {}).get('seed', 42)
    rng = random.Random(timing_seed)

    # Group by observation_size (cluster size = number of reads)
    from collections import defaultdict
    by_obs_size = defaultdict(list)
    for item in all_data:
        obs_size = item[2]  # cluster_size = observation_size
        by_obs_size[obs_size].append(item)

    print(f"  Found observation sizes: {sorted(by_obs_size.keys())}")
    for obs in sorted(by_obs_size.keys()):
        print(f"    Obs size {obs}: {len(by_obs_size[obs])} examples")

    # Sample equal number from each observation_size
    samples_per_obs = timing_cfg.get('samples_per_observation_size', batch_size)  # Default to batch_size for clean division
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
        lengths = [get_unpadded_input_length(item) for item in sampled]
        print(f"  Sampled {len(sampled)} from obs_size={obs}, lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

    print(f"  Total timing subset: {len(timing_subset)} examples")

    # Sort subset by length for efficient batching (minimize padding)
    timing_subset.sort(key=get_unpadded_input_length)
    print(f"  Sorted by length for efficient batching")

    # Group into length buckets for minimal padding within batches
    # Bucket size determines the max length range within a batch
    bucket_size = cfg.get('timing', {}).get('bucket_length_range', 15)  # Default: max 15 char range per bucket
    print(f"  Grouping into length buckets (max range per bucket: {bucket_size} chars)")

    # Calculate lengths for all items
    lengths_with_items = [(get_unpadded_input_length(item), item) for item in timing_subset]

    # Create buckets: group by length ranges
    length_buckets = defaultdict(list)
    for length, item in lengths_with_items:
        bucket_id = length // bucket_size  # e.g., lengths 150-164 -> bucket 10 (if bucket_size=15)
        length_buckets[bucket_id].append(item)

    print(f"  Created {len(length_buckets)} length buckets:")
    for bucket_id in sorted(length_buckets.keys()):
        bucket = length_buckets[bucket_id]
        bucket_lengths = [get_unpadded_input_length(item) for item in bucket]
        min_len = min(bucket_lengths)
        max_len = max(bucket_lengths)
        avg_len = sum(bucket_lengths) / len(bucket_lengths)
        obs_sizes = [item[2] for item in bucket]
        obs_dist = {obs: obs_sizes.count(obs) for obs in sorted(set(obs_sizes))}
        print(f"    Bucket {bucket_id} (len {min_len}-{max_len}): {len(bucket)} examples, "
              f"avg={avg_len:.1f}, range={max_len-min_len}, obs_sizes={obs_dist}")

    # Flatten buckets into a list where each batch will come from a single bucket
    # This ensures minimal padding within each batch
    # We'll cycle through buckets to maintain diversity
    bucketed_data = []
    bucket_ids = sorted(length_buckets.keys())

    # For each bucket, create full batches
    for bucket_id in bucket_ids:
        bucket = length_buckets[bucket_id]
        # Shuffle within bucket to mix observation sizes
        rng.shuffle(bucket)
        bucketed_data.extend(bucket)

    print(f"  Reorganized {len(bucketed_data)} examples into length-bucketed order")
    print(f"  This ensures each batch has minimal length variation (improving GPU efficiency)")

    # Enable length debugging to see the mixed distribution
    enable_length_debug = True

    # Storage for results
    all_throughputs = []
    all_example_counts = []
    all_durations = []

    # Create initial cycle (will be reset for each measured run)
    # Use bucketed_data instead of timing_subset for better batching
    data_cycle = cycle(enumerate(bucketed_data))

    # Run timing windows
    for run_idx in range(num_runs):
        is_warmup = run_idx < warmup_runs
        run_label = "WARMUP" if is_warmup else f"RUN {run_idx - warmup_runs + 1}"

        # Reset cycle at start of each measured run (not warmup)
        # This ensures all measured runs see the same data for consistent results
        if not is_warmup:
            data_cycle = cycle(enumerate(bucketed_data))

        print(f"\n{run_label}: Starting {run_duration / 60:.1f} minute timing window...")

        run_start = time.perf_counter()
        examples_this_run = 0

        # Process examples until time limit
        while True:
            # Check if we've exceeded time limit
            elapsed = time.perf_counter() - run_start
            if elapsed >= run_duration:
                break

            # Collect batch of consecutive examples (sorted by length if variable)
            batch = [next(data_cycle)[1] for _ in range(batch_size)]

            # Debug: Print batch length statistics (only if lengths vary)
            if enable_length_debug and examples_this_run % (log_interval * 10) == 0:
                batch_lengths = [get_unpadded_input_length(item) for item in batch]
                min_len = min(batch_lengths)
                max_len = max(batch_lengths)
                avg_len = sum(batch_lengths) / len(batch_lengths)
                print(f"  [{run_label}] Batch unpadded input lengths: min={min_len}, max={max_len}, avg={avg_len:.1f}, range={max_len-min_len}")

            # Extract inputs from batch (all_data is 3-tuple format)
            inputs = [decode_tokens(ex[0].tolist()) for ex in batch]
            gts = [ex[1] for ex in batch]
            alignment_sizes = [ex[2] for ex in batch]

            # Remove padding and extract prefix
            inputs = [s.split('#', 1)[0] for s in inputs]

            # Run inference (model timing only, no metrics collection)
            # NOTE: We don't use run_one_batch() here because it includes extra overhead:
            # - Hamming/Levenshtein distance computation
            # - Contamination metrics tracking
            # - Position-wise analysis
            # - File I/O for logging
            # For pure throughput measurement, we only time the model inference itself.
            try:
                # Build inference parameters
                inf_params = {
                    'model': model,
                    'ctx': ctx,
                    **sampling_dict,
                    'device': device,
                    'stoi': stoi,
                    'itos': itos,
                    'encode': encode_str,
                    'decode': decode_tokens,
                }

                # Synchronize GPU before inference
                if device.type == 'cuda':
                    torch.cuda.synchronize()

                # Run model inference
                _ = GPT_Inference(inf_params).inference(list(inputs), alignment_size=alignment_sizes)

                # Synchronize GPU after inference
                if device.type == 'cuda':
                    torch.cuda.synchronize()

                examples_this_run += len(batch)

                # Log progress
                if examples_this_run % log_interval == 0:
                    elapsed_now = time.perf_counter() - run_start
                    current_rate = (examples_this_run / elapsed_now) * 3600 if elapsed_now > 0 else 0
                    print(f"  [{run_label}] Processed {examples_this_run} examples in {elapsed_now:.1f}s (current rate: {current_rate:.0f} ex/hr)")

            except Exception as e:
                print(f"  Warning: Skipped example due to error: {e}")
                continue

        # Calculate metrics for this run
        run_end = time.perf_counter()
        actual_duration = run_end - run_start
        throughput = (examples_this_run / actual_duration) * 3600

        print(f"  [{run_label}] Completed: {examples_this_run} examples in {actual_duration:.1f}s, {throughput:.0f} ex/hr")

        # Store if not warmup
        if not is_warmup:
            all_throughputs.append(throughput)
            all_example_counts.append(examples_this_run)
            all_durations.append(actual_duration)

    # Calculate statistics (excluding warmup)
    mean_throughput = np.mean(all_throughputs)
    std_throughput = np.std(all_throughputs)
    cv_throughput = (std_throughput / mean_throughput) * 100 if mean_throughput > 0 else 0

    total_examples = sum(all_example_counts)
    total_time_min = sum(all_durations) / 60

    print_section_header("TIMING RESULTS", width=80)
    print(f"Measured runs: {num_runs - warmup_runs}")
    print(f"Individual throughputs: {[f'{t:.0f}' for t in all_throughputs]} ex/hr")
    print(f"")
    print(f"Mean throughput: {mean_throughput:.0f} ex/hr")
    print(f"Std throughput: {std_throughput:.0f} ex/hr")
    print(f"Coefficient of Variation (CV): {cv_throughput:.2f}%")
    print(f"")
    print(f"Total examples processed: {total_examples}")
    print(f"Total time (measured runs): {total_time_min:.1f} minutes")
    print(f"")
    print(f"Extrapolated to 60 minutes: ~{mean_throughput:.0f} examples/hour")
    print_separator(width=80, newline_before=False, newline_after=True)

    # Log to WandB
    wandb_log = {
        'timing_mean_throughput_per_hour': mean_throughput,
        'timing_std_throughput_per_hour': std_throughput,
        'timing_cv_throughput_percent': cv_throughput,
        'timing_total_examples': total_examples,
        'timing_total_time_minutes': total_time_min,
        'timing_num_measured_runs': num_runs - warmup_runs,
    }

    # Log individual run throughputs
    for i, throughput in enumerate(all_throughputs):
        wandb_log[f'timing_run_{i+1}_throughput'] = throughput

    wandb.log(wandb_log)
    print(f"Timing results logged to WandB")


@hydra.main(config_path='hydra/inference_config',
            config_name='inference_config.yaml',
            version_base=None)
def main(cfg: DictConfig):
    try:
        _main_impl(cfg)
    finally:
        # Cleanup distributed and wandb
        if dist.is_initialized():
            dist.destroy_process_group()
        if wandb.run is not None:
            wandb.finish()

def _main_impl(cfg: DictConfig):
    """Main implementation."""
    # Check if running in distributed mode
    distributed = is_distributed()

    if distributed:
        # Initialize distributed training
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Use rank as local device index (CUDA_VISIBLE_DEVICES remaps GPUs to 0,1,2,...)
        local_device_id = rank
        torch.cuda.set_device(local_device_id)
        device = torch.device(f'cuda:{local_device_id}')
        print(f"Distributed mode: rank {rank}/{world_size}, using device {device} (local_id: {local_device_id})")
    else:
        # Single process mode 
        rank = 0
        if torch.cuda.is_available():
            # Option to use multiple GPUs in single process mode
            num_gpus = torch.cuda.device_count()
            # For now, use single GPU in single-process mode for simplicity
            # Multi-GPU single-process would require more complex batching logic
            world_size = 1
            device = torch.device('cuda:0')
            if num_gpus > 1:
                print(f"Single process mode: using device {device} (GPU 0 of {num_gpus} available)")
                print(f"  Note: Use 'torchrun --nproc_per_node={num_gpus}' for multi-GPU inference")
            else:
                print(f"Single process mode: using device {device}")
            torch.cuda.set_device(0)
        else:
            world_size = 1
            device = torch.device('cpu')
            print(f"Single process mode: using device {device}")

    # Load model on each GPU 
    ckpt       = torch.load(cfg.model.checkpoint_path, map_location='cpu')
    wandb_run_id = ckpt.get("wandb_run_id", None)  

    # Only rank 0 (or single process) initializes W&B
    if rank == 0:
        run_cfg  = wandb_kwargs_via_cfg(cfg)
        now      = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix   = "_sweep" if cfg.data.get("sweep", False) else ""
        cross_suffix = f"_cross_{cfg.data.get('cross', '')}" if cfg.data.get('cross') else ""
        base     = f"TReconLM_inference_{now}{suffix}{cross_suffix}"
        run_name = f"{base}_{cfg.experiment}" if cfg.experiment else base
        out_dir = os.path.join(cfg.general.results_path,
                               'model_evaluation',
                               cfg.wandb.wandb_project,
                               run_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"wandb_run_id from checkpoint: {wandb_run_id}")
        
        print("cfg says entity/project:", cfg.wandb.wandb_entity, cfg.wandb.wandb_project)
                
        # Try to resume the run from checkpoint, fallback to new run if fails
        try:
            if wandb_run_id:
                # First try to resume the existing run
                wandb.init(
                    project=cfg.wandb.wandb_project,
                    entity=cfg.wandb.wandb_entity,
                    name=run_name,
                    config=run_cfg,
                    dir=out_dir,
                    id=wandb_run_id,
                    resume="must"
                )
                print(f"Successfully resumed wandb run: {wandb_run_id}")
            else:
                # No run ID in checkpoint, start new run
                wandb.init(
                    project=cfg.wandb.wandb_project,
                    entity=cfg.wandb.wandb_entity,
                    name=run_name,
                    config=run_cfg,
                    dir=out_dir,
                    resume=None
                )
                print("Started new wandb run (no run ID in checkpoint)")
        except Exception as e:
            print(f"Failed to resume wandb run {wandb_run_id}: {e}")
            print("Starting new wandb run...")
            # Resume failed, start a new run
            wandb.init(
                project=cfg.wandb.wandb_project,
                entity=cfg.wandb.wandb_entity,
                name=run_name,
                config=run_cfg,
                dir=out_dir,
                resume=None
            )
            print("Started new wandb run")
            sys.stdout.flush()

    # Don't need barrier after wandb init - only rank 0 uses it
    # First real synchronization will be after model loading

    model_args = ckpt['model_args']
    state_dict = ckpt['model']
    for k in list(state_dict):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

    # Load vocab metadata 
    script_dir   = os.path.dirname(__file__)
    data_pkg_dir = os.path.join(script_dir, 'data_pkg')
    meta_path    = os.path.join(data_pkg_dir,
                                f"meta_{cfg.data.sequence_type}.pkl")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos   = meta['stoi'], meta['itos']

        
    model_type = model_args.get('model_type', 'gpt')

    if model_type == 'gpt':
        model_args.pop("model_type", None)
        model = GPT(GPTConfig(**model_args)).half().to(device).eval()
    elif model_type == 'lstm':
        pad_id   = stoi['#']
        lstm_cfg = LSTMConfig(
            vocab_size = model_args['vocab_size'],
            n_layer    = model_args['n_layer'],
            n_embd     = model_args['n_embd'],
            dropout    = model_args['dropout'],
        )
        model = (LSTMConsensus(lstm_cfg, pad_id).half().to(device).eval())         

    elif model_type == 'mamba':
        from src.mamba_pkg.my_config_mamba import MambaConfig
        from src.mamba_pkg.my_mamba_model  import MambaLMHeadModel    
        
        clean_args = {k: v for k, v in model_args.items() if k != "model_type"}

        mcfg = MambaConfig(**clean_args)
        model = (MambaLMHeadModel(mcfg).half().to(device).eval())

    else:
        raise ValueError(f"Unsupported model_type {model_type}")

    model.load_state_dict(state_dict, strict=False)
    print(f"Rank {rank}: Model loaded successfully on {device}")

    # Apply INT8 quantization if enabled
    use_int8 = cfg.model.get('use_int8_quantization', False)
    if use_int8:
        if rank == 0:
            print_section_header("APPLYING INT8 QUANTIZATION", width=60)
        try:
            import bitsandbytes as bnb
            import torch.nn as nn

            # Convert model to INT8 by replacing Linear layers
            # This replaces torch.nn.Linear layers with 8-bit quantized versions
            def replace_linear_with_int8(module, parent_device):
                """Recursively replace Linear layers with INT8 versions."""
                for name, child in module.named_children():
                    if isinstance(child, nn.Linear):
                        # Create INT8 Linear layer on same device as original
                        int8_layer = bnb.nn.Linear8bitLt(
                            child.in_features,
                            child.out_features,
                            bias=child.bias is not None,
                            has_fp16_weights=False,
                            threshold=6.0
                        )

                        # Move INT8 layer to same device before copying weights
                        int8_layer = int8_layer.to(parent_device)

                        # Copy weight data
                        with torch.no_grad():
                            int8_layer.weight.data.copy_(child.weight.data)
                            if child.bias is not None:
                                int8_layer.bias.data.copy_(child.bias.data)

                        setattr(module, name, int8_layer)
                    else:
                        # Recursively apply to submodules
                        replace_linear_with_int8(child, parent_device)

            replace_linear_with_int8(model, device)

            # Run a warmup forward pass to initialize INT8 quantization state
            # This builds the codebook (CB) and other quantization metadata
            if rank == 0:
                print("INT8 quantization applied, initializing with warmup pass...")
            try:
                with torch.no_grad():
                    # Create dummy input matching model's expected input
                    dummy_input = torch.randint(0, meta['vocab_size'], (1, cfg.data.block_size), device=device)
                    _ = model(dummy_input)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                if rank == 0:
                    print("  Warmup pass completed")
            except Exception as e:
                if rank == 0:
                    print(f"  Warning: Warmup pass failed: {e}")

            if rank == 0:
                print("INT8 quantization initialized successfully")
                print_separator(width=60, newline_before=False, newline_after=True)
        except ImportError:
            if rank == 0:
                print(f"\n{'='*60}")
                print("WARNING: INT8 quantization requested but bitsandbytes not installed")
                print("Install with: pip install bitsandbytes")
                print("Continuing without quantization...")
                print_separator(width=60, newline_before=False, newline_after=True)
        except Exception as e:
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"WARNING: Failed to apply INT8 quantization: {e}")
                print("Continuing without quantization...")
                print_separator(width=60, newline_before=False, newline_after=True)

    # Exchange positional encoding if specified in config
    pe_checkpoint_path = cfg.model.sampling.get('exchange_positional_encoding', None)
    if pe_checkpoint_path:
        if rank == 0:  # Only print on main process to avoid spam
            print_section_header("POSITIONAL ENCODING EXCHANGE", width=60)
        exchange_positional_encoding(model, pe_checkpoint_path, model_type)
        if rank == 0:
            print_separator(width=60, newline_before=False, newline_after=True)

    # Compile model for faster inference (PyTorch 2.0+)
    # Expected speedup: 30-200% depending on model and batch size
    compile_model = cfg.model.get('compile', False)
    if compile_model:
        try:
            # Check PyTorch version
            pytorch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
            if pytorch_version >= (2, 0):
                if rank == 0:
                    print_section_header("COMPILING MODEL WITH TORCH.COMPILE", width=60)
                    print(f"PyTorch version: {torch.__version__}")
                    print(f"Mode: reduce-overhead (optimized for fixed shapes)")
                    print("This may take 1-2 minutes on first run...")

                # Use 'reduce-overhead' mode for best performance with fixed shapes
                # 'max-autotune' gives even better performance but takes longer to compile
                compile_mode = cfg.model.get('compile_mode', 'reduce-overhead')
                model = torch.compile(model, mode=compile_mode)

                if rank == 0:
                    print("Model compiled successfully!")
                    print_separator(width=60, newline_before=False, newline_after=True)
            else:
                if rank == 0:
                    print(f"Warning: torch.compile requires PyTorch 2.0+, found {torch.__version__}")
                    print("Skipping compilation, using eager mode.")
        except Exception as e:
            if rank == 0:
                print(f"Warning: Failed to compile model: {e}")
                print("Continuing with eager mode (non-compiled).")

    ctx = torch.amp.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    # Skip early barriers - they cause NCCL deadlocks
    # We'll synchronize only when necessary (before data sharding and inference)
    if rank == 0:
        print("Model loading complete on all ranks")

    # Print cross-evaluation mode if set
    cross_mode = cfg.data.get('cross', None)
    if rank == 0 and cross_mode:
        print(f"\n{'='*60}")
        print(f"CROSS-EVALUATION MODE: {cross_mode}")
        if cross_mode == 'noisy':
            print("Evaluating Microsoft model (110nt) on Noisy DNA data (60nt)")
            print("Strategy: Concatenating pairs of 60nt sequences -> 120nt")
            print("Evaluation: First 110nt of predictions")
        elif cross_mode == 'microsoft':
            print("Evaluating Noisy DNA model (60nt) on Microsoft data (110nt)")
            print("Strategy: Using 110nt sequences as-is")
            print("Evaluation: First 60nt of predictions")
        print_separator(width=60, newline_before=False, newline_after=True)

    # Loop over sweep or single artifact
    ks = list(range(11)) if cfg.data.get("sweep", False) else [None]
    for k in ks:
        if k is not None:
            seed     = cfg.data.test_seed + k
            art_name = (f"sweep{k}_seed{seed}_gl"
                        f"{cfg.data.ground_truth_length}_bs"
                        f"{cfg.data.block_size}_ds"
                        f"{cfg.data.test_dataset_size}_"
                        f"fixedN{cfg.data.observation_size}")
        else:
            art_name = cfg.data.artifact_name

        # Check if local_data_dir is provided (skip artifact download)
        local_data_dir = cfg.data.get('local_data_dir', None)

        if local_data_dir:
            # Use local data directory
            art_dir = local_data_dir
            if rank == 0:
                print(f"Using local data directory: {art_dir}")
                sys.stdout.flush()

                # Verify required files exist
                test_x_path = os.path.join(art_dir, 'test_x.pt')
                gt_file = 'ground_truth_cleaned.txt' if cfg.data.get("cleaned", False) else 'ground_truth.txt'
                gt_path = os.path.join(art_dir, gt_file)

                if not os.path.exists(test_x_path):
                    raise FileNotFoundError(f"test_x.pt not found in {art_dir}")
                if not os.path.exists(gt_path):
                    raise FileNotFoundError(f"{gt_file} not found in {art_dir}")

                print(f"Found test_x.pt and {gt_file} in local directory")
                sys.stdout.flush()
        else:
            # Download artifact from wandb
            # Rank 0 downloads artifact and writes path to a shared file
            # Other ranks read the path from the file
            # This avoids NCCL broadcast issues and wandb.init() requirements on systems with broken GPU communication
            if rank == 0:
                print(f"Rank 0: Downloading artifact {art_name}...")
                sys.stdout.flush()
                art_dir = safe_download_artifact(
                    cfg.wandb.wandb_entity,
                    cfg.data.data_project,
                    art_name
                )
                print(f"Rank 0: Download complete for {art_name}")
                sys.stdout.flush()

                # Write artifact path to shared file for other ranks
                if distributed:
                    # Use a unique filename based on master port to avoid conflicts
                    master_port = os.environ.get('MASTER_PORT', '29500')
                    art_path_file = f"/tmp/artifact_path_port{master_port}.txt"
                    with open(art_path_file, 'w') as f:
                        f.write(art_dir)
                    print(f"Rank 0: Wrote artifact path to {art_path_file}")
                    sys.stdout.flush()
            else:
                # Wait for rank 0 to write the artifact path
                master_port = os.environ.get('MASTER_PORT', '29500')
                art_path_file = f"/tmp/artifact_path_port{master_port}.txt"
                max_wait = 300  # 5 minutes
                waited = 0
                while not os.path.exists(art_path_file) and waited < max_wait:
                    time.sleep(1)
                    waited += 1

                if os.path.exists(art_path_file):
                    with open(art_path_file, 'r') as f:
                        art_dir = f.read().strip()
                    print(f"Rank {rank}: Read artifact path from file: {art_dir}")
                    sys.stdout.flush()
                else:
                    raise RuntimeError(f"Rank {rank}: Timed out waiting for artifact path file")

            # Clean up the artifact path file at exit (rank 0 does this)
            if rank == 0 and distributed:
                import atexit
                atexit.register(lambda: os.remove(art_path_file) if os.path.exists(art_path_file) else None)

        # Load test examples
        # Stagger file reads to avoid I/O contention in distributed mode
        if distributed and rank != 0:
            time.sleep(0.1 * rank)  # Small delay for non-rank-0 processes

        x_test = torch.load(os.path.join(art_dir, 'test_x.pt'),
                            map_location='cpu')
        gt_file = ('ground_truth_cleaned.txt'
                   if cfg.data.get("cleaned", False)
                   else 'ground_truth.txt')
        with open(os.path.join(art_dir, gt_file)) as f:
            gts = [l.strip() for l in f]
        assert len(x_test) == len(gts)

        # Handle cross-evaluation data preparation (cross_mode already defined outside loop)
        if cross_mode == 'noisy':
            # For noisy mode: concatenate pairs of 60nt sequences to create 120nt
            print(f"Cross-evaluation mode: {cross_mode}: concatenating pairs of noisy DNA sequences")

            # Import random for shuffling/subsampling experiments
            import random
            random.seed(42)  # Reproducible results

            # Get subsample flag from config (not used in noisy mode anymore)
            subsample_large = False  # Disabled for noisy mode

            # Load raw reads for easier concatenation
            reads_file = os.path.join(art_dir, 'reads.txt')
            if not os.path.exists(reads_file):
                raise FileNotFoundError(f"reads.txt not found in {art_dir}. Required for noisy cross-evaluation.")

            with open(reads_file, 'r') as f:
                reads_content = f.read()

            # Parse reads grouped by separator
            read_groups = [group.strip().split('\n') for group in reads_content.split('===============================')
                          if group.strip()]

            # Ensure we have the right number of read groups
            if len(read_groups) != len(gts):
                raise ValueError(f"Mismatch: {len(read_groups)} read groups vs {len(gts)} ground truths")

            # Group examples by cluster size first
            groups_by_size = defaultdict(list)
            for i, reads in enumerate(read_groups):
                cluster_size = len(reads)
                groups_by_size[cluster_size].append((i, reads, gts[i]))

            new_x_test = []
            new_gts = []
            total_skipped = 0

            # Process pairs within each cluster size group
            for cluster_size, examples in groups_by_size.items():
                print(f"Processing cluster size {cluster_size}: {len(examples)} examples")

                # Process pairs within this cluster size
                for j in range(0, len(examples) - 1, 2):
                    idx1, reads1, gt1 = examples[j]
                    idx2, reads2, gt2 = examples[j + 1]

                    # Sanity check (should always be true now)
                    assert len(reads1) == len(reads2) == cluster_size

                    # Randomly shuffle reads for cluster size 10 only
                    if cluster_size == 10:
                        print(f"Shuffling reads for cluster size 10 (pair {j//2 + 1})")
                        reads1_shuffled = reads1.copy()
                        reads2_shuffled = reads2.copy()
                        random.shuffle(reads1_shuffled)
                        random.shuffle(reads2_shuffled)
                        reads1, reads2 = reads1_shuffled, reads2_shuffled

                    # Concatenate corresponding reads
                    concatenated_reads = [r1 + r2 for r1, r2 in zip(reads1, reads2)]

                    # Concatenate ground truths
                    combined_gt = gt1 + gt2

                    # Create the input sequence in the expected format: reads|reads|...:ground_truth
                    combined_seq = '|'.join(concatenated_reads) + ':' + combined_gt

                    # Encode the concatenated sequence
                    encoded_seq = [stoi.get(ch, stoi.get('<unk>', 0)) for ch in combined_seq]
                    new_x = torch.tensor(encoded_seq, dtype=torch.long)

                    new_x_test.append(new_x)
                    new_gts.append(combined_gt)

                # Count skipped examples (odd ones in each group)
                if len(examples) % 2 == 1:
                    total_skipped += 1

            if total_skipped > 0:
                print(f"Skipped {total_skipped} unpaired examples (odd counts per cluster size)")

            x_test = new_x_test
            gts = new_gts
            print(f"Created {len(x_test)} concatenated examples from original data")
            print(f"Example concatenated ground truth length: {len(gts[0]) if gts else 0}")

        elif cross_mode == 'microsoft':
            # For microsoft mode: subsample large clusters if configured
            print(f"Cross-evaluation mode: {cross_mode} - using Microsoft data with potential subsampling")

            # Import random for subsampling experiments
            import random
            random.seed(42)  # Reproducible results

            # Get subsample flag from config
            subsample_large = cfg.data.get('subsample_large_clusters', False)

            if subsample_large:
                print("Subsampling enabled for large clusters (9, 10)")

                # Load raw reads for easier manipulation
                reads_file = os.path.join(art_dir, 'reads.txt')
                if not os.path.exists(reads_file):
                    raise FileNotFoundError(f"reads.txt not found in {art_dir}. Required for microsoft cross-evaluation with subsampling.")

                with open(reads_file, 'r') as f:
                    reads_content = f.read()

                # Parse reads grouped by separator
                read_groups = [group.strip().split('\n') for group in reads_content.split('===============================')
                              if group.strip()]

                # Ensure we have the right number of read groups
                if len(read_groups) != len(gts):
                    raise ValueError(f"Mismatch: {len(read_groups)} read groups vs {len(gts)} ground truths")

                new_x_test = []
                new_gts = []
                original_cluster_sizes = []  # Store the true cluster sizes

                for i, reads in enumerate(read_groups):
                    cluster_size = len(reads)
                    gt = gts[i]

                    # Store the original cluster size
                    original_cluster_sizes.append(cluster_size)

                    # Subsample large clusters
                    if cluster_size in [9, 10]:
                        subsample_size = random.randint(2, 5)  # Random size between 2-5
                        print(f"Subsampling cluster size {cluster_size} to {subsample_size} reads (example {i+1})")
                        reads = random.sample(reads, subsample_size)

                    # Create the input sequence in the expected format: reads|reads|...:ground_truth
                    seq = '|'.join(reads) + ':' + gt

                    # Encode the sequence
                    encoded_seq = [stoi.get(ch, stoi.get('<unk>', 0)) for ch in seq]
                    new_x = torch.tensor(encoded_seq, dtype=torch.long)

                    new_x_test.append(new_x)
                    new_gts.append(gt)

                x_test = new_x_test
                gts = new_gts
                print(f"Processed {len(x_test)} examples with subsampling for large clusters")
            else:
                print("No subsampling - using Microsoft data as-is")

        # Sync after data loading
        if distributed:
            dist.barrier()

        itos = meta['itos']
        decode_tokens = lambda x: ''.join(itos[i] for i in x)

        # Check if we have original cluster sizes (from microsoft cross-eval with subsampling)
        has_original_sizes = cross_mode == 'microsoft' and subsample_large and 'original_cluster_sizes' in locals()

        # Build data tuples and group by length buckets for efficient batching
        # Groups similar lengths together while maintaining full batches
        BUCKET_SIZE = 20  # Group sequences within 20 tokens of each other

        # Find colon token ID for fast tensor-based prefix length calculation
        colon_id = meta['stoi'][':']
        pipe_id = meta['stoi']['|']

        by_length_bucket = defaultdict(list)
        for i, (x_tensor, gt) in enumerate(zip(x_test, gts)):
            # Fast: Find colon position directly in tensor (no string conversion!)
            tensor_list = x_tensor.tolist()
            colon_pos = tensor_list.index(colon_id) if colon_id in tensor_list else len(tensor_list)

            # Fast: Count cluster size by counting pipe tokens before colon
            if has_original_sizes:
                cs = original_cluster_sizes[i]
            else:
                cs = sum(1 for tok in tensor_list[:colon_pos] if tok == pipe_id) + 1

            prefix_len = colon_pos  # Length of prefix in tokens

            # Assign to bucket (e.g., 500-519, 520-539, etc.)
            bucket_key = (prefix_len // BUCKET_SIZE) * BUCKET_SIZE
            by_length_bucket[bucket_key].append((x_tensor, gt, cs, prefix_len))

        # Flatten buckets in sorted order for sequential batching
        # This ensures batches have similar-length sequences (minimal padding)
        # while maintaining full batch sizes (good GPU utilization)
        all_data = []
        for bucket_key in sorted(by_length_bucket.keys()):
            bucket_items = by_length_bucket[bucket_key]
            # Sort within bucket for even better grouping
            bucket_items.sort(key=lambda t: t[3])
            all_data.extend(bucket_items)

        all_data = [(x, gt, cs) for x, gt, cs, _ in all_data]

        if rank == 0:
            print(f"Grouped {len(all_data)} examples into {len(by_length_bucket)} length buckets (bucket_size={BUCKET_SIZE})")

        # Optional: Sample a subset of examples for faster testing
        max_samples = cfg.model.sampling.get('max_samples', None)
        sampling_seed = cfg.model.sampling.get('sampling_seed', 42)

        if max_samples is not None and max_samples < len(all_data):
            print(f"\nSampling {max_samples} of {len(all_data)} examples (seed={sampling_seed})")
            rng = np.random.RandomState(sampling_seed)
            sampled_indices = rng.choice(len(all_data), size=max_samples, replace=False)
            all_data = [all_data[i] for i in sampled_indices]

            # Re-bucket by length after sampling
            BUCKET_SIZE = 20
            by_length_bucket_sampled = defaultdict(list)
            colon_id = meta['stoi'][':']
            pipe_id = meta['stoi']['|']

            for item in all_data:
                # Fast: Find colon position directly in tensor
                tensor_list = item[0].tolist()
                colon_pos = tensor_list.index(colon_id) if colon_id in tensor_list else len(tensor_list)
                prefix_len = colon_pos

                bucket_key = (prefix_len // BUCKET_SIZE) * BUCKET_SIZE
                by_length_bucket_sampled[bucket_key].append((item[0], item[1], item[2], prefix_len))

            # Flatten in sorted order
            all_data = []
            for bucket_key in sorted(by_length_bucket_sampled.keys()):
                bucket_items = by_length_bucket_sampled[bucket_key]
                bucket_items.sort(key=lambda t: t[3])
                all_data.extend(bucket_items)

            all_data = [(x, gt, cs) for x, gt, cs, _ in all_data]

        # Permute traces if requested (for testing positional bias in attention)
        permute_traces = cfg.model.sampling.get('permute_traces', False)
        if permute_traces:
            print(f"\n{'='*80}")
            print(f"PERMUTING TRACES ORDER (testing for positional bias)")
            print(f"{'='*80}")
            permutation_seed = cfg.model.sampling.get('permutation_seed', sampling_seed)
            permutation_rng = np.random.RandomState(permutation_seed)

            permuted_data = []
            for x_tensor, gt, cs in all_data:
                # Permute the traces within this example
                permuted_x = permute_traces_in_tensor(x_tensor, meta['itos'], meta['stoi'], permutation_rng)
                permuted_data.append((permuted_x, gt, cs))

            all_data = permuted_data
            print(f"Successfully permuted traces for {len(all_data)} examples (seed={permutation_seed})")
            print(f"{'='*80}\n")

        # Check if timing mode is enabled
        if cfg.get('timing', {}).get('enabled', False):
            run_timing_measurement(cfg, model, meta, device, ctx, all_data)
            if log_file is not None:
                log_file.close()
            wandb.finish()
            return  # Exit early after timing measurement

        # Check if we should run baseline inference per contamination rate
        subset_indices_path = cfg.data.get('subset_indices_path', None)
        run_per_rate_baseline = False
        indices_per_rate = {}

        if subset_indices_path and os.path.exists(subset_indices_path):
            print(f"\nLoading subset indices from: {subset_indices_path}")
            with open(subset_indices_path, 'r') as f:
                indices_data = json.load(f)

            indices_per_rate = indices_data['contaminated_indices_per_rate']
            run_per_rate_baseline = True
            print(f"  Loaded per-rate indices for {len(indices_per_rate)} contamination rates")
            print(f"  Will run baseline inference separately for each rate")
            for rate_name, indices in indices_per_rate.items():
                print(f"    {rate_name}: {len(indices)} examples")

        # Data already sorted by length for efficient batching
        print(f"\nSplitting {len(all_data)} length-sorted examples across {world_size} GPU(s)")

        # Shard among GPUs
        chunks   = split_list(all_data, world_size)
        my_chunk = chunks[rank]

        if distributed:
            print(f"Rank {rank}: Received {len(my_chunk)} examples out of {len(all_data)} total")
            dist.barrier()
            print(f"Rank {rank}: All ranks synchronized after data sharding")

        sampling_dict = extract_sampling_config(cfg)
        sampling_dict['cross_mode'] = cross_mode
        sampling_dict['track_attention'] = cfg.model.sampling.get('track_attention', False)
        sampling_dict['track_all_layers'] = cfg.model.sampling.get('track_all_layers', False)
        sampling_dict['save_per_head_attention'] = cfg.model.sampling.get('save_per_head_attention', False)

        # Create log file for majority voting output (if enabled)
        majority_voting_enabled = cfg.model.sampling.get('majority_voting', {}).get('enabled', False)
        log_file = None
        if majority_voting_enabled and rank == 0:  # Only rank 0 creates log file in current directory
            # Create log file in current working directory
            log_filename = f"majority_voting_rank{rank}.log" if distributed else "majority_voting.log"
            log_filepath = os.path.join(os.getcwd(), log_filename)
            log_file = open(log_filepath, 'w', buffering=1)  # Line buffering for real-time updates
            sampling_dict['log_file'] = log_file
            print(f"Majority voting output will be logged to: {log_filepath}")
        elif majority_voting_enabled and distributed:
            # Non-rank-0 processes get their own log files in current directory
            log_filename = f"majority_voting_rank{rank}.log"
            log_filepath = os.path.join(os.getcwd(), log_filename)
            log_file = open(log_filepath, 'w', buffering=1)
            sampling_dict['log_file'] = log_file
        else:
            sampling_dict['log_file'] = None

        # Run misclustering robustness experiment if enabled (only on rank 0)
        misclustering_results = None
        if rank == 0:
            misclustering_results = run_misclustering_robustness_experiment(
                all_data, cfg, model, meta, device, ctx, rank
            )

        # If misclustering experiment was run, log results and exit
        if misclustering_results is not None:
            # Log misclustering robustness results to WandB
            if wandb.run is not None:
                print(f"\nLogging misclustering robustness results to WandB...")

                # Log overall experiment metadata (convert ListConfig to list for JSON serialization)
                misc_metadata = {
                    'misclustering_baseline_error_rate_per_nt': float(misclustering_results['baseline_error_rate_per_nt']),
                    'misclustering_ground_truth_length': int(misclustering_results['baseline_info']['ground_truth_length']),
                    'misclustering_expected_total_edit_distance': float(misclustering_results['baseline_info']['expected_total_edit_distance']),
                    'misclustering_contamination_rates': list(misclustering_results['contamination_rates']),  # Convert to list
                    'misclustering_total_conditions': int(len(misclustering_results['results_by_condition'])),
                    'misclustering_total_contamination_events': int(len(misclustering_results['contamination_details']))
                }
                wandb.log(misc_metadata)

                # Log results for each experimental condition (overall results)
                for condition_name, condition_data in misclustering_results['results_by_condition'].items():
                    metrics = condition_data['overall']
                    condition_log = {
                        f'misclustering_{condition_name}_mean_hamming': float(metrics['mean_hamming']),
                        f'misclustering_{condition_name}_std_hamming': float(metrics['std_hamming']),
                        f'misclustering_{condition_name}_mean_levenshtein': float(metrics['mean_levenshtein']),
                        f'misclustering_{condition_name}_std_levenshtein': float(metrics['std_levenshtein']),
                        f'misclustering_{condition_name}_num_examples': int(metrics['num_examples']),
                        f'misclustering_{condition_name}_success_rate_all': float(metrics['success_rate']),
                        f'misclustering_{condition_name}_failure_rate_all': float(metrics['failure_rate']),
                    }
                    wandb.log(condition_log)

                # Log detailed bin metrics for heatmap (contamination_rate × multiplier_bin)
                total_bin_entries = 0
                for condition_name, condition_data in misclustering_results['results_by_condition'].items():
                    if 'by_multiplier_bin' in condition_data:
                        for bin_name, bin_metrics in condition_data['by_multiplier_bin'].items():
                            bin_log = {
                                f'misclustering_{condition_name}_{bin_name}_mean_hamming': float(bin_metrics['mean_hamming']),
                                f'misclustering_{condition_name}_{bin_name}_std_hamming': float(bin_metrics['std_hamming']),
                                f'misclustering_{condition_name}_{bin_name}_mean_levenshtein': float(bin_metrics['mean_levenshtein']),
                                f'misclustering_{condition_name}_{bin_name}_std_levenshtein': float(bin_metrics['std_levenshtein']),
                                f'misclustering_{condition_name}_{bin_name}_num_examples': int(bin_metrics['num_examples'])
                            }
                            wandb.log(bin_log)
                            total_bin_entries += 1
                print(f"  Logged {total_bin_entries} detailed bin metrics for heatmap")

                # Log detailed cluster size metrics
                total_cluster_entries = 0
                for condition_name, condition_data in misclustering_results['results_by_condition'].items():
                    if 'by_cluster_size' in condition_data:
                        for N, cluster_data in condition_data['by_cluster_size'].items():
                            all_metrics = cluster_data['all_examples']
                            cont_metrics = cluster_data['contaminated_only']

                            cluster_log = {
                                # All examples metrics
                                f'misclustering_{condition_name}_N{N}_all_count': int(all_metrics['count']),
                                f'misclustering_{condition_name}_N{N}_all_mean_hamming': float(all_metrics['mean_hamming']),
                                f'misclustering_{condition_name}_N{N}_all_std_hamming': float(all_metrics['std_hamming']),
                                f'misclustering_{condition_name}_N{N}_all_mean_levenshtein': float(all_metrics['mean_levenshtein']),
                                f'misclustering_{condition_name}_N{N}_all_std_levenshtein': float(all_metrics['std_levenshtein']),
                                f'misclustering_{condition_name}_N{N}_all_success_rate': float(all_metrics['success_rate']),
                                f'misclustering_{condition_name}_N{N}_all_failure_rate': float(all_metrics['failure_rate']),

                                # Contaminated-only metrics
                                f'misclustering_{condition_name}_N{N}_contaminated_count': int(cont_metrics['count']),
                                f'misclustering_{condition_name}_N{N}_contaminated_mean_hamming': float(cont_metrics['mean_hamming']),
                                f'misclustering_{condition_name}_N{N}_contaminated_std_hamming': float(cont_metrics['std_hamming']),
                                f'misclustering_{condition_name}_N{N}_contaminated_mean_levenshtein': float(cont_metrics['mean_levenshtein']),
                                f'misclustering_{condition_name}_N{N}_contaminated_std_levenshtein': float(cont_metrics['std_levenshtein']),
                                f'misclustering_{condition_name}_N{N}_contaminated_success_rate': float(cont_metrics['success_rate']),
                                f'misclustering_{condition_name}_N{N}_contaminated_failure_rate': float(cont_metrics['failure_rate']),
                            }
                            wandb.log(cluster_log)
                            total_cluster_entries += 1
                print(f"  Logged {total_cluster_entries} detailed cluster size metrics")

                # Log 2D matrices if available (ensure lists are JSON serializable)
                if 'results_2d' in misclustering_results:
                    results_2d = misclustering_results['results_2d']
                    matrix_log = {
                        'misclustering_2d_hamming_matrix': results_2d['hamming_matrix'],  # Already converted to list in helper_functions.py
                        'misclustering_2d_levenshtein_matrix': results_2d['levenshtein_matrix'],
                        'misclustering_2d_counts_matrix': results_2d['counts_matrix'],
                        'misclustering_2d_contamination_rates': list(results_2d['contamination_rates']),  # Convert to list
                        'misclustering_2d_multiplier_bins': list(results_2d['multiplier_bins'])  # Convert to list
                    }
                    wandb.log(matrix_log)
                    print(f"  Logged 2D matrices for heatmap visualization")

                # Log cluster size 2D matrices if available
                if 'results_2d_cluster_size' in misclustering_results:
                    results_2d_cs = misclustering_results['results_2d_cluster_size']
                    cluster_matrix_log = {
                        'misclustering_cluster_size_hamming_matrix': results_2d_cs['hamming_matrix'],
                        'misclustering_cluster_size_levenshtein_matrix': results_2d_cs['levenshtein_matrix'],
                        'misclustering_cluster_size_counts_matrix': results_2d_cs['counts_matrix'],
                        'misclustering_cluster_size_contamination_rates': list(results_2d_cs['contamination_rates']),
                        'misclustering_cluster_size_cluster_sizes': list(results_2d_cs['cluster_sizes'])
                    }
                    wandb.log(cluster_matrix_log)
                    print(f"  Logged cluster size 2D matrices for heatmap visualization")

                print(f"Misclustering robustness results logged to WandB")

            print(f"\nMisclustering robustness experiment completed. Exiting without normal inference.")
            if log_file is not None:
                log_file.close()
            if distributed:
                dist.barrier()  # Sync before all ranks exit
            return  # Exit early, skip normal inference

        # Sync after potential misclustering experiment
        if distributed:
            dist.barrier()

        # Run logit margin profiling if enabled (only on rank 0)
        # Check both model.profile_logit_margin and top-level profile_logit_margin
        profile_cfg = None
        if hasattr(cfg, 'model') and hasattr(cfg.model, 'profile_logit_margin'):
            profile_cfg = cfg.model.profile_logit_margin
        elif hasattr(cfg, 'profile_logit_margin'):
            profile_cfg = cfg.profile_logit_margin

        if rank == 0 and profile_cfg and hasattr(profile_cfg, 'enabled') and profile_cfg.enabled:
            print(f"\n{'='*80}")
            print("MODEL CONFIDENCE PROFILING")
            print_separator(width=80, newline_before=False, newline_after=True)

            num_samples = profile_cfg.num_samples if hasattr(profile_cfg, 'num_samples') else 1000

            # Subsample data
            profiling_data = subsample_data_uniformly(all_data, num_samples, seed=42)

            # Process in batches
            batch_size = cfg.data.batch_size
            all_margin_results = []

            # Group by cluster size for efficient batching
            grouped_data = defaultdict(list)
            for x_tensor, gt, cluster_size in profiling_data:
                grouped_data[cluster_size].append((x_tensor, gt, cluster_size))

            # Create progress bar for all batches
            total_batches = sum(math.ceil(len(data) / batch_size) for data in grouped_data.values())
            pbar = tqdm(total=total_batches, desc="Profiling model confidence", unit="batch")

            for cluster_size, cluster_data in sorted(grouped_data.items()):
                num_batches = math.ceil(len(cluster_data) / batch_size)

                for batch_idx in range(num_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, len(cluster_data))
                    batch_data = cluster_data[start:end]

                    # Extract data
                    batch_x_tensors = [item[0] for item in batch_data]
                    batch_gt = [item[1] for item in batch_data]
                    batch_cs = [item[2] for item in batch_data]

                    # Decode tensors to strings (like normal inference does)
                    # test_x.pt contains full examples: "reads:groundtruth######"
                    decode_fn = lambda l: ''.join([meta['itos'][i] for i in l])
                    batch_strings = [decode_fn(x.tolist()) for x in batch_x_tensors]

                    # Split on ':' to get just the prefix (reads part)
                    prefixes = [s.split(':', 1)[0] for s in batch_strings]

                    # Encode prefixes
                    encode_fn = lambda s: [meta['stoi'][c] for c in s]
                    enc_prefixes = [encode_fn(p) for p in prefixes]

                    # Left-pad to max length in batch and add colon (like GPT_Inference does)
                    pad_id = meta['stoi']['#']
                    colon_id = meta['stoi'][':']
                    max_prefix_len = max(len(e) for e in enc_prefixes)

                    # Create left-padded tensor with colon
                    X = torch.tensor([
                        [pad_id] * (max_prefix_len - len(e)) + e + [colon_id]
                        for e in enc_prefixes
                    ], dtype=torch.long, device=device)

                    # Create attention mask (False for left padding, True for real tokens + colon)
                    attn_mask = torch.tensor([
                        [False] * (max_prefix_len - len(e)) + [True] * (len(e) + 1)
                        for e in enc_prefixes
                    ], dtype=torch.bool, device=device)

                    # Profile this batch
                    batch_results = profile_logit_margins_single_batch(
                        test_examples=X,
                        attn_mask=attn_mask,
                        model=model,
                        ctx=ctx,
                        max_new_tokens=cfg.data.ground_truth_length,
                        stoi=meta['stoi'],
                        itos=meta['itos'],
                        temperature=sampling_dict.get('temperature', 1.0),
                        constrained_generation=sampling_dict.get('constrained_generation', False),
                        cluster_sizes=batch_cs
                    )

                    all_margin_results.extend(batch_results)
                    pbar.update(1)

            pbar.close()
            print()  # Newline after progress bar

            # Aggregate statistics
            margin_stats = aggregate_margin_statistics(all_margin_results)
            print_margin_statistics(margin_stats)

            # Log to WandB if available
            if wandb.run is not None:
                # Build a single dictionary with all metrics to log in one call
                log_dict = {
                    'confidence_overall_mean_top1': float(margin_stats['overall']['mean_top1_prob']),
                    'confidence_overall_mean_top2': float(margin_stats['overall']['mean_top2_prob']),
                    'confidence_num_examples': int(margin_stats['overall']['num_examples'])
                }

                # Add per-cluster-size statistics
                for cluster_size, stats in margin_stats['by_cluster_size'].items():
                    log_dict[f'confidence_N{cluster_size}_mean_top1'] = float(stats['mean_top1_prob'])
                    log_dict[f'confidence_N{cluster_size}_mean_top2'] = float(stats['mean_top2_prob'])
                    log_dict[f'confidence_N{cluster_size}_count'] = int(stats['count'])

                # Add highest/lowest
                if margin_stats['highest_confidence_cluster']['cluster_size'] is not None:
                    log_dict['confidence_highest_cluster_size'] = int(margin_stats['highest_confidence_cluster']['cluster_size'])
                    log_dict['confidence_highest_mean_top1'] = float(margin_stats['highest_confidence_cluster']['mean_top1_prob'])

                if margin_stats['lowest_confidence_cluster']['cluster_size'] is not None:
                    log_dict['confidence_lowest_cluster_size'] = int(margin_stats['lowest_confidence_cluster']['cluster_size'])
                    log_dict['confidence_lowest_mean_top1'] = float(margin_stats['lowest_confidence_cluster']['mean_top1_prob'])

                # Log everything at once
                wandb.log(log_dict)
                print(f"\nLogged {len(log_dict)} confidence metrics to WandB")
                print(f"Search for 'confidence' in your WandB run to view results")

            print(f"\n{'='*80}")
            print("Model confidence profiling completed")
            print_separator(width=80, newline_before=False, newline_after=True)

            if distributed:
                dist.barrier()

            if log_file is not None:
                log_file.close()

            if wandb.run is not None:
                wandb.finish()

            return

        # Sync after potential profiling
        if distributed:
            dist.barrier()

        # If we have per-rate indices, run baseline inference separately for each rate
        if run_per_rate_baseline and rank == 0:
            print(f"\n{'='*80}")
            print("BASELINE INFERENCE PER CONTAMINATION RATE")
            print_separator(width=80, newline_before=False, newline_after=True)

            per_rate_results = {}

            for rate_name, rate_indices in indices_per_rate.items():
                print(f"\nProcessing baseline for {rate_name}")
                print(f"  Number of examples: {len(rate_indices)}")

                # Filter data to this rate's indices
                rate_data = [all_data[i] for i in rate_indices if i < len(all_data)]

                if not rate_data:
                    print(f"  No valid data for {rate_name}, skipping")
                    continue

                # Run inference on this subset
                batch_size = cfg.data.batch_size
                num_batches = math.ceil(len(rate_data) / batch_size)
                rate_results = []

                pbar = tqdm(range(num_batches), desc=f"Baseline {rate_name}")

                for batch_idx in pbar:
                    start = batch_idx * batch_size
                    res, _, _ = run_one_batch(rate_data, start, batch_size,
                                            sampling_dict, model, meta,
                                            device, ctx, pbar)
                    rate_results.extend(res)

                # Compute overall metrics for this rate
                hamming_distances = [r[8] for r in rate_results]
                levenshtein_distances = [r[9] for r in rate_results]

                rate_metrics = {
                    'mean_hamming': np.mean(hamming_distances),
                    'std_hamming': np.std(hamming_distances),
                    'mean_levenshtein': np.mean(levenshtein_distances),
                    'std_levenshtein': np.std(levenshtein_distances),
                    'num_examples': len(rate_results)
                }

                # Compute per-cluster-size metrics for this rate
                cluster_size_metrics = defaultdict(lambda: {'hamming': [], 'levenshtein': []})
                for result in rate_results:
                    cluster_size = result[0]  # First element is cluster size
                    hamming = result[8]
                    levenshtein = result[9]
                    cluster_size_metrics[cluster_size]['hamming'].append(hamming)
                    cluster_size_metrics[cluster_size]['levenshtein'].append(levenshtein)

                # Aggregate per cluster size
                cluster_size_aggregates = {}
                for cs, metrics in cluster_size_metrics.items():
                    cluster_size_aggregates[cs] = {
                        'mean_hamming': np.mean(metrics['hamming']),
                        'std_hamming': np.std(metrics['hamming']),
                        'mean_levenshtein': np.mean(metrics['levenshtein']),
                        'std_levenshtein': np.std(metrics['levenshtein']),
                        'num_examples': len(metrics['hamming'])
                    }

                per_rate_results[rate_name] = {
                    'overall': rate_metrics,
                    'by_cluster_size': cluster_size_aggregates
                }

                print(f"  Overall: Hamming {rate_metrics['mean_hamming']:.3f}±{rate_metrics['std_hamming']:.3f}, "
                      f"Levenshtein {rate_metrics['mean_levenshtein']:.3f}±{rate_metrics['std_levenshtein']:.3f}")
                print(f"  Cluster sizes: {sorted(cluster_size_aggregates.keys())}")

                # Log to WandB - overall metrics
                if wandb.run is not None:
                    wandb.log({
                        f'baseline_{rate_name}_overall_mean_hamming': rate_metrics['mean_hamming'],
                        f'baseline_{rate_name}_overall_std_hamming': rate_metrics['std_hamming'],
                        f'baseline_{rate_name}_overall_mean_levenshtein': rate_metrics['mean_levenshtein'],
                        f'baseline_{rate_name}_overall_std_levenshtein': rate_metrics['std_levenshtein'],
                        f'baseline_{rate_name}_overall_num_examples': rate_metrics['num_examples']
                    })

                    # Log per-cluster-size metrics
                    for cs, cs_metrics in cluster_size_aggregates.items():
                        wandb.log({
                            f'baseline_{rate_name}_cs{cs}_mean_hamming': cs_metrics['mean_hamming'],
                            f'baseline_{rate_name}_cs{cs}_std_hamming': cs_metrics['std_hamming'],
                            f'baseline_{rate_name}_cs{cs}_mean_levenshtein': cs_metrics['mean_levenshtein'],
                            f'baseline_{rate_name}_cs{cs}_std_levenshtein': cs_metrics['std_levenshtein'],
                            f'baseline_{rate_name}_cs{cs}_num_examples': cs_metrics['num_examples']
                        })

            print_section_header("Baseline inference per rate completed!", width=80)

            # Exit after per-rate baseline (don't run normal inference)
            if log_file is not None:
                log_file.close()
            return

        # Inference on each rank
        batch_size  = cfg.data.batch_size
        num_batches = math.ceil(len(my_chunk) / batch_size)

        # Check if test mode is enabled for majority voting
        majority_cfg = cfg.model.sampling.get('majority_voting', {})
        test_mode = majority_cfg.get('test_mode', False)
        test_num_batches = majority_cfg.get('test_num_batches', 4)

        if test_mode and majority_cfg.get('enabled', False):
            original_num_batches = num_batches
            num_batches = min(num_batches, test_num_batches)
            if rank == 0:
                print(f"\n{'='*80}")
                print(f"TEST MODE ENABLED")
                print(f"{'='*80}")
                print(f"Processing only {num_batches} batches (out of {original_num_batches} total)")
                print(f"Total examples to process per rank: approximately {num_batches * batch_size}")
                print(f"Max permutations: {majority_cfg.get('max_permutations', 10)}")
                print(f"{'='*80}\n")

        local_results = []
        total_skipped = 0

        if distributed:
            print(f"Rank {rank}: Starting inference with {num_batches} batches, batch_size={batch_size}")
            # Ensure all ranks are ready to start inference
            dist.barrier()
            if rank == 0:
                print("All ranks synchronized and ready for inference")


        # Only show progress bar on rank 0 to avoid interference in distributed mode
        if distributed:
            pbar = tqdm(range(num_batches), desc=f"Rank {rank}", disable=(rank != 0))
        else:
            pbar = tqdm(range(num_batches), desc=f"Rank {rank}")

        all_position_metrics = []  # Collect all position metrics for vote confidence analysis

        for batch_idx in pbar:
            start = batch_idx * batch_size
            if distributed and batch_idx % 10 == 0:  # Log progress every 10 batches
                pbar.write(f"Rank {rank}: Processing batch {batch_idx}/{num_batches}")

            res, skipped, position_metrics = run_one_batch_with_majority_voting(my_chunk, start, batch_size,
                                          sampling_dict, model, meta,
                                          device, ctx, pbar)
            local_results.extend(res)
            total_skipped += skipped
            all_position_metrics.extend(position_metrics)  # Collect position metrics

            # Periodic sync to prevent divergence
            if distributed and batch_idx % 50 == 0 and batch_idx > 0:
                dist.barrier()
                if rank == 0:
                    print(f"Checkpoint sync at batch {batch_idx}")

        # Gather all local_results on rank 0
        if distributed:
            print(f"Rank {rank}: Completed local inference, gathered {len(local_results)} results, skipped {total_skipped}")
            dist.barrier()
            print(f"Rank {rank}: Starting all_gather_object")

            gathered_results = [None for _ in range(world_size)]
            gathered_skipped = [None for _ in range(world_size)]
            gathered_position_metrics = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_results, local_results)
            dist.all_gather_object(gathered_skipped, total_skipped)
            dist.all_gather_object(gathered_position_metrics, all_position_metrics)

            if rank == 0:
                all_results = [r for shard in gathered_results for r in shard]
                total_skipped_all = sum(gathered_skipped)
                all_position_metrics = [m for shard in gathered_position_metrics for m in shard]
                print(f"Rank 0: Gathered total {len(all_results)} results, {total_skipped_all} skipped from all ranks")
        else:
            # In single-process mode, local_results is all_results
            all_results = local_results
            total_skipped_all = total_skipped
            # all_position_metrics stays as is

        if rank == 0:
            n_ex = len(all_results)

            # Extract cropped and full metrics from new result format
            # Format: (N, reads, gt_cropped, pred_cropped, ham_cropped, lev_cropped,
            #          gt_full, pred_full, ham_full, lev_full, full_pipeline_time, model_only_time, token_entropies, attention_data)
            h_vals_cropped = np.array([r[4] for r in all_results])
            l_vals_cropped = np.array([r[5] for r in all_results])
            h_vals_full = np.array([r[8] for r in all_results])
            l_vals_full = np.array([r[9] for r in all_results])
            batch_times_full = np.array([r[10] for r in all_results])  # Full pipeline time
            batch_times_model = np.array([r[11] if r[11] is not None else r[10] for r in all_results])  # Model-only time (fallback to full if not available)

            # Perform entropy analysis if entropy tracking is enabled
            track_entropy = cfg.model.sampling.get('track_entropy', False)
            if track_entropy:
                print("\nPerforming entropy analysis...")
                entropy_stats = aggregate_entropy_by_cluster_size(all_results)
                correlation = log_entropy_analysis(entropy_stats, use_wandb=wandb.run is not None)

            # Save sequences to TSV file if enabled
            save_sequences = cfg.model.sampling.get('save_sequences', False)
            if save_sequences:
                sequence_output_file = cfg.model.sampling.get('sequence_output_file', 'predictions_output.tsv')
                print(f"\nSaving sequences to {sequence_output_file}...")
                save_sequences_to_file(all_results, sequence_output_file)

            # Perform vote confidence analysis if enabled
            analyze_vote_confidence = cfg.model.sampling.get('analyze_vote_confidence', False)
            if analyze_vote_confidence and all_position_metrics:
                print("\nPerforming comprehensive vote confidence analysis...")

                # Create output directory for analysis results
                # Use custom output directory from config if specified
                custom_vote_dir = cfg.model.sampling.get('vote_confidence_output_dir', None)
                if custom_vote_dir:
                    vote_analysis_dir = custom_vote_dir
                else:
                    vote_analysis_dir = os.path.join(out_dir, "vote_confidence_analysis")

                # Run comprehensive analysis with excess loss computation and W&B logging
                correlations, stratified_results, loss_analysis = analyze_and_log_vote_confidence(
                    all_position_metrics,
                    vote_analysis_dir,
                    log_to_wandb=(wandb.run is not None),
                    wandb_prefix="vote_analysis"
                )

                print(f"Analyzed {len(all_position_metrics)} positions from {loss_analysis.get('num_examples', 0)} examples")
                print(f"Estimated p_s: {loss_analysis.get('estimated_p_s_mean', 0):.4f} ± {loss_analysis.get('estimated_p_s_std', 0):.4f}")
                print(f"p_s range: [{loss_analysis.get('estimated_p_s_min', 0):.4f}, {loss_analysis.get('estimated_p_s_max', 0):.4f}]")
                print(f"Average excess loss: {loss_analysis['avg_excess_loss']:.4f}")
                print(f"Model performs {'better' if loss_analysis['avg_excess_loss'] < 0 else 'worse'} than optimal predictor")
                print(f"Analysis results saved to: {vote_analysis_dir}")

                # Save detailed position metrics to file for later analysis/plotting
                save_position_data = cfg.model.sampling.get('save_position_data', True)
                if save_position_data and all_position_metrics:
                    import pandas as pd
                    position_df = pd.DataFrame([{
                        'vote_margin': metric['vote_margin'],
                        'prob_margin': metric['prob_margin'],
                        'top1_prob': metric['top1_prob'],
                        'entropy': metric['entropy'],
                        'cluster_size': metric['cluster_size'],
                        'correct': metric['correct'],
                        'excess_loss': metric.get('pos_excess_loss', 0),
                        'example_id': metric.get('example_id', 0),
                        'position': metric.get('position', 0)
                    } for metric in all_position_metrics])

                    position_data_path = os.path.join(vote_analysis_dir, "position_metrics.parquet")
                    position_df.to_parquet(position_data_path, index=False)
                    print(f"Saved {len(all_position_metrics)} position metrics to: {position_data_path}")

                    # Optional: Still log a subset to W&B for quick overview
                    detailed_position_logging = cfg.model.sampling.get('detailed_position_logging', False)
                    if detailed_position_logging and wandb.run is not None:
                        # Sample a subset for W&B (e.g., every 100th position)
                        sample_size = min(1000, len(all_position_metrics))
                        sample_indices = np.random.choice(len(all_position_metrics), sample_size, replace=False)
                        print(f"Logging {sample_size} sampled position metrics to W&B...")
                        for idx in sample_indices:
                            metric = all_position_metrics[idx]
                            wandb.log({
                                "position_vote_margin": metric['vote_margin'],
                                "position_prob_margin": metric['prob_margin'],
                                "position_top1_prob": metric['top1_prob'],
                                "position_entropy": metric['entropy'],
                                "position_cluster_size": metric['cluster_size'],
                                "position_correct": metric['correct'],
                                "position_excess_loss": metric.get('pos_excess_loss', 0)
                            })

            # Perform aggregate attention analysis if attention tracking is enabled
            track_attention = cfg.model.sampling.get('track_attention', False)
            if track_attention:
                print("\nPerforming aggregate attention analysis...")
                attention_by_size = aggregate_detailed_attention_by_cluster_size(all_results)

                # Compute and print aggregate statistics by cluster size
                if attention_by_size:
                    all_analyses = []
                    for cluster_size, analyses in attention_by_size.items():
                        all_analyses.extend(analyses)

                    if all_analyses:
                        summary_stats = aggregate_attention_stats_by_cluster_size(all_analyses)
                        if summary_stats:
                            print_attention_comparison_summary(summary_stats)

            if k is not None:
                success_count_cropped = sum(1 for r in all_results if r[4] == 0)
                success_count_full = sum(1 for r in all_results if r[8] == 0)

                log_dict = {
                    f"avg_hamming_cropped_k={k}":     float(h_vals_cropped.mean()),
                    f"avg_levenshtein_cropped_k={k}": float(l_vals_cropped.mean()),
                    f"success_rate_cropped_k={k}":    success_count_cropped / n_ex,
                    f"failure_rate_cropped_k={k}":    1 - (success_count_cropped / n_ex),
                }

                # Add full sequence metrics if in cross mode
                if cross_mode:
                    log_dict.update({
                        f"avg_hamming_full_k={k}":     float(h_vals_full.mean()),
                        f"avg_levenshtein_full_k={k}": float(l_vals_full.mean()),
                        f"success_rate_full_k={k}":    success_count_full / n_ex,
                        f"failure_rate_full_k={k}":    1 - (success_count_full / n_ex),
                        f"cross_mode_k={k}": cross_mode
                    })
                wandb.log(log_dict)
            else:
                # breakdown by N for both cropped and full metrics
                count = defaultdict(int)
                success_cropped, success_full = defaultdict(int), defaultdict(int)
                h_per_N_cropped, l_per_N_cropped = defaultdict(list), defaultdict(list)
                h_per_N_full, l_per_N_full = defaultdict(list), defaultdict(list)

                for r in all_results:
                    # Extract values from tuple
                    N = r[0]
                    ham_cropped, lev_cropped = r[4], r[5]
                    ham_full, lev_full = r[8], r[9]

                    count[N] += 1
                    success_cropped[N] += (ham_cropped == 0)
                    success_full[N] += (ham_full == 0)
                    h_per_N_cropped[N].append(ham_cropped)
                    l_per_N_cropped[N].append(lev_cropped)
                    h_per_N_full[N].append(ham_full)
                    l_per_N_full[N].append(lev_full)

                for N in sorted(count):
                    h_arr_cropped = np.array(h_per_N_cropped[N])
                    l_arr_cropped = np.array(l_per_N_cropped[N])

                    if cross_mode:
                        # Cross evaluation - use cropped/full distinction
                        metrics_dict = {
                            f"count_N={N}":                      count[N],
                            f"success_rate_cropped_N={N}":       success_cropped[N] / count[N],
                            f"avg_hamming_cropped_N={N}":        float(h_arr_cropped.mean()),
                            f"std_hamming_cropped_N={N}":        float(h_arr_cropped.std()),
                            f"avg_levenshtein_cropped_N={N}":    float(l_arr_cropped.mean()),
                            f"std_levenshtein_cropped_N={N}":    float(l_arr_cropped.std()),
                        }
                    else:
                        # Normal evaluation - clean naming
                        metrics_dict = {
                            f"count_N={N}":                  count[N],
                            f"success_rate_N={N}":           success_cropped[N] / count[N],
                            f"avg_hamming_N={N}":            float(h_arr_cropped.mean()),
                            f"std_hamming_N={N}":            float(h_arr_cropped.std()),
                            f"avg_levenshtein_N={N}":        float(l_arr_cropped.mean()),
                            f"std_levenshtein_N={N}":        float(l_arr_cropped.std()),
                        }

                    # Add full metrics if in cross mode
                    if cross_mode:
                        h_arr_full = np.array(h_per_N_full[N])
                        l_arr_full = np.array(l_per_N_full[N])
                        metrics_dict.update({
                            f"success_rate_full_N={N}":      success_full[N] / count[N],
                            f"avg_hamming_full_N={N}":       float(h_arr_full.mean()),
                            f"std_hamming_full_N={N}":       float(h_arr_full.std()),
                            f"avg_levenshtein_full_N={N}":   float(l_arr_full.mean()),
                            f"std_levenshtein_full_N={N}":   float(l_arr_full.std()),
                        })

                    wandb.log(metrics_dict)

                # Log majority voting metrics if enabled
                majority_cfg = cfg.model.sampling.get('majority_voting', {})
                if majority_cfg.get('enabled', False) and all_results:
                    # Check if results contain majority voting stats (extended tuple with 24 elements)
                    if len(all_results[0]) >= 24:
                        print("\n" + "="*80)
                        print("MAJORITY VOTING RESULTS")
                        print("="*80)

                        # Aggregate majority voting metrics by cluster size
                        mv_metrics_by_N = defaultdict(lambda: {
                            'num_perms': [],
                            'first_lev': [],
                            'voted_lev': [],
                            'lev_improvement': [],
                            'first_failed': [],
                            'voted_failed': [],
                            'failure_rescued': [],
                            'diversity': [],
                            'vote_agreement': [],
                        })

                        for r in all_results:
                            N = r[0]
                            mv_metrics_by_N[N]['num_perms'].append(r[14])
                            mv_metrics_by_N[N]['first_lev'].append(r[15])
                            mv_metrics_by_N[N]['voted_lev'].append(r[9])  # voted_lev is at index 9
                            mv_metrics_by_N[N]['lev_improvement'].append(r[16])
                            mv_metrics_by_N[N]['first_failed'].append(r[17])
                            mv_metrics_by_N[N]['voted_failed'].append(r[18])
                            mv_metrics_by_N[N]['failure_rescued'].append(r[19])
                            mv_metrics_by_N[N]['diversity'].append(r[20])
                            mv_metrics_by_N[N]['vote_agreement'].append(r[22])

                        # Log per-cluster-size majority voting metrics
                        for N in sorted(mv_metrics_by_N.keys()):
                            metrics = mv_metrics_by_N[N]

                            first_lev_arr = np.array(metrics['first_lev'])
                            voted_lev_arr = np.array(metrics['voted_lev'])
                            lev_improvement_arr = np.array(metrics['lev_improvement'])

                            first_failed_arr = np.array(metrics['first_failed'])
                            voted_failed_arr = np.array(metrics['voted_failed'])
                            failure_rescued_arr = np.array(metrics['failure_rescued'])

                            first_failure_rate = first_failed_arr.mean()
                            voted_failure_rate = voted_failed_arr.mean()
                            rescued_failures_frac = failure_rescued_arr.mean()

                            diversity_arr = np.array(metrics['diversity'])
                            vote_agreement_arr = np.array(metrics['vote_agreement'])
                            num_perms = int(np.mean(metrics['num_perms']))  # Should be consistent per cluster size

                            # Print summary for this cluster size
                            print(f"\nCluster size N={N} ({count[N]} examples, {num_perms} permutations):")
                            print(f"  First inference:  LEV={first_lev_arr.mean():.4f}, Failure rate={first_failure_rate:.2%}")
                            print(f"  Voted inference:  LEV={voted_lev_arr.mean():.4f}, Failure rate={voted_failure_rate:.2%}")
                            print(f"  Improvement:      LEV={lev_improvement_arr.mean():+.4f}, Rescued={rescued_failures_frac:.2%}")
                            print(f"  Diversity:        Hamming={diversity_arr.mean():.2f}")

                            mv_log = {
                                # Raw performance
                                f'majority_voting_first_levenshtein_N={N}': float(first_lev_arr.mean()),
                                f'majority_voting_voted_levenshtein_N={N}': float(voted_lev_arr.mean()),
                                f'majority_voting_levenshtein_improvement_N={N}': float(lev_improvement_arr.mean()),

                                # Failure rates
                                f'majority_voting_first_failure_rate_N={N}': float(first_failure_rate),
                                f'majority_voting_voted_failure_rate_N={N}': float(voted_failure_rate),
                                f'majority_voting_rescued_failures_frac_N={N}': float(rescued_failures_frac),

                                # Diversity
                                f'majority_voting_diversity_N={N}': float(diversity_arr.mean()),
                                f'majority_voting_vote_agreement_N={N}': float(vote_agreement_arr.mean()),
                                f'majority_voting_num_perms_N={N}': num_perms,
                            }
                            wandb.log(mv_log)

                        # Overall majority voting metrics
                        all_first_lev = np.array([r[15] for r in all_results])
                        all_voted_lev = np.array([r[9] for r in all_results])
                        all_lev_improvement = np.array([r[16] for r in all_results])
                        all_first_failed = np.array([r[17] for r in all_results])
                        all_voted_failed = np.array([r[18] for r in all_results])
                        all_failure_rescued = np.array([r[19] for r in all_results])
                        all_diversity = np.array([r[20] for r in all_results])
                        all_vote_agreement = np.array([r[22] for r in all_results])

                        overall_first_failure_rate = all_first_failed.mean()
                        overall_voted_failure_rate = all_voted_failed.mean()
                        overall_rescued_frac = all_failure_rescued.mean()

                        print(f"\n{'='*80}")
                        print("OVERALL MAJORITY VOTING SUMMARY")
                        print(f"{'='*80}")
                        print(f"First inference:  LEV={all_first_lev.mean():.4f}, Failure rate={overall_first_failure_rate:.2%}")
                        print(f"Voted inference:  LEV={all_voted_lev.mean():.4f}, Failure rate={overall_voted_failure_rate:.2%}")
                        print(f"Improvement:      LEV={all_lev_improvement.mean():+.4f} ({all_lev_improvement.mean()/all_first_lev.mean()*100:+.1f}%)")
                        print(f"Failure rate improvement: {overall_first_failure_rate - overall_voted_failure_rate:+.2%}")
                        print(f"Rescued failures: {overall_rescued_frac:.2%}")
                        print(f"Mean diversity:   {all_diversity.mean():.2f} Hamming distance between predictions")
                        print(f"Mean vote agreement: {all_vote_agreement.mean():.3f}")
                        print(f"Voting helped in: {(all_lev_improvement > 0).mean():.1%} of cases")
                        print(f"{'='*80}\n")

                        mv_overall_log = {
                            'majority_voting_first_levenshtein_all': float(all_first_lev.mean()),
                            'majority_voting_voted_levenshtein_all': float(all_voted_lev.mean()),
                            'majority_voting_levenshtein_improvement_all': float(all_lev_improvement.mean()),
                            'majority_voting_first_failure_rate_all': float(overall_first_failure_rate),
                            'majority_voting_voted_failure_rate_all': float(overall_voted_failure_rate),
                            'majority_voting_failure_rate_improvement_all': float(overall_first_failure_rate - overall_voted_failure_rate),
                            'majority_voting_rescued_failures_frac_all': float(overall_rescued_frac),
                            'majority_voting_diversity_all': float(all_diversity.mean()),
                            'majority_voting_vote_agreement_all': float(all_vote_agreement.mean()),
                            'majority_voting_helped_rate_all': float((all_lev_improvement > 0).mean()),
                        }
                        wandb.log(mv_overall_log)

                        # Test mode summary
                        if test_mode:
                            print(f"\n{'='*80}")
                            print(f"TEST MODE: WANDB LOGGING SUMMARY")
                            print(f"{'='*80}")
                            print(f"Logged metrics for {len(mv_metrics_by_N)} cluster sizes")
                            print(f"Logged overall majority voting metrics")
                            print(f"Total metrics logged to W&B: approximately {len(mv_metrics_by_N) * 9 + 10} items")
                            print(f"\nKey metrics logged:")
                            for N in sorted(mv_metrics_by_N.keys()):
                                print(f"  N={N}: first_lev, voted_lev, improvement, failure rates, diversity")
                            print(f"  Overall: all aggregated metrics")
                            print(f"{'='*80}\n")

                # log global metrics
                if cross_mode:
                    # Use "cropped" naming for cross evaluation
                    log_dict = {
                        'count_all':                      n_ex,
                        'count_skipped':                  total_skipped_all,
                        'count_total_attempted':          n_ex + total_skipped_all,
                        'success_rate_cropped_all':       sum(success_cropped.values()) / n_ex if n_ex > 0 else 0,
                        'avg_hamming_cropped_all':        float(h_vals_cropped.mean()) if n_ex > 0 else 0,
                        'avg_levenshtein_cropped_all':    float(l_vals_cropped.mean()) if n_ex > 0 else 0,
                        'avg_time_per_example_full':      batch_times_full.mean() if n_ex > 0 else 0,
                        'std_time_per_example_full':      batch_times_full.std() if n_ex > 0 else 0,
                        'avg_time_per_example_model':     batch_times_model.mean() if n_ex > 0 else 0,
                        'std_time_per_example_model':     batch_times_model.std() if n_ex > 0 else 0,
                        'preprocessing_overhead_ms':      (batch_times_full.mean() - batch_times_model.mean()) * 1000 if n_ex > 0 else 0,
                    }
                else:
                    # Use normal naming when no cross evaluation
                    log_dict = {
                        'count_all':                      n_ex,
                        'count_skipped':                  total_skipped_all,
                        'count_total_attempted':          n_ex + total_skipped_all,
                        'success_rate_all':               sum(success_cropped.values()) / n_ex if n_ex > 0 else 0,
                        'avg_hamming_all':                float(h_vals_cropped.mean()) if n_ex > 0 else 0,
                        'avg_levenshtein_all':            float(l_vals_cropped.mean()) if n_ex > 0 else 0,
                        'avg_time_per_example_full':      batch_times_full.mean() if n_ex > 0 else 0,
                        'std_time_per_example_full':      batch_times_full.std() if n_ex > 0 else 0,
                        'avg_time_per_example_model':     batch_times_model.mean() if n_ex > 0 else 0,
                        'std_time_per_example_model':     batch_times_model.std() if n_ex > 0 else 0,
                        'preprocessing_overhead_ms':      (batch_times_full.mean() - batch_times_model.mean()) * 1000 if n_ex > 0 else 0,
                    }

                # Add full sequence metrics if in cross mode
                if cross_mode:
                    log_dict.update({
                        'success_rate_full_all':      sum(success_full.values()) / n_ex,
                        'avg_hamming_full_all':       float(h_vals_full.mean()),
                        'avg_levenshtein_full_all':   float(l_vals_full.mean()),
                        'cross_evaluation_mode':      cross_mode,
                    })

                    if cross_mode == 'noisy':
                        log_dict['evaluation_cropped_length'] = 110  # Cropped evaluation length
                        log_dict['evaluation_full_length'] = 120     # Full concatenated length
                        log_dict['original_seq_length'] = 60         # Original noisy DNA sequences
                    elif cross_mode == 'microsoft':
                        log_dict['evaluation_cropped_length'] = 60   # Cropped evaluation length
                        log_dict['evaluation_full_length'] = 110     # Full sequence length
                        log_dict['original_seq_length'] = 110        # Original Microsoft sequences

                wandb.log(log_dict)

                # Log misclustering robustness results if available
                if misclustering_results and wandb.run is not None:
                    print(f"\nLogging misclustering robustness results to WandB...")

                    # Log overall experiment metadata
                    misc_metadata = {
                        'misclustering_baseline_error_rate_per_nt': misclustering_results['baseline_error_rate_per_nt'],
                        'misclustering_ground_truth_length': misclustering_results['baseline_info']['ground_truth_length'],
                        'misclustering_expected_total_edit_distance': misclustering_results['baseline_info']['expected_total_edit_distance'],
                        'misclustering_contamination_rates': misclustering_results['contamination_rates'],
                        'misclustering_total_conditions': len(misclustering_results['results_by_condition']),
                        'misclustering_total_contamination_events': len(misclustering_results['contamination_details'])
                    }
                    wandb.log(misc_metadata)

                    # Analyze contamination details and group by realized edit distance multipliers
                    contamination_details = misclustering_results['contamination_details']
                    if contamination_details:
                        print(f"  Analyzing {len(contamination_details)} contamination events...")

                        # Collect all realized multipliers for grouping
                        all_multipliers = []
                        for detail in contamination_details:
                            for contam_info in detail.get('contaminated_positions', []):
                                realized_mult = contam_info.get('realized_edit_distance_multiplier', 0)
                                if realized_mult > 0:  # Only include valid multipliers
                                    all_multipliers.append(realized_mult)

                        if all_multipliers:
                            multiplier_stats = {
                                'misclustering_min_realized_multiplier': min(all_multipliers),
                                'misclustering_max_realized_multiplier': max(all_multipliers),
                                'misclustering_mean_realized_multiplier': np.mean(all_multipliers),
                                'misclustering_std_realized_multiplier': np.std(all_multipliers),
                                'misclustering_num_contamination_events': len(all_multipliers)
                            }
                            wandb.log(multiplier_stats)

                            print(f"    Realized edit distance multipliers: {multiplier_stats['misclustering_min_realized_multiplier']:.2f} - {multiplier_stats['misclustering_max_realized_multiplier']:.2f} (mean: {multiplier_stats['misclustering_mean_realized_multiplier']:.2f})")

                    # Log multiplier bin configuration if available
                    if 'multiplier_bin_config' in misclustering_results:
                        bin_config = misclustering_results['multiplier_bin_config']

                        # Log bin configuration
                        bin_config_log = {
                            'misclustering_multiplier_bins_count': bin_config['num_bins'],
                            'misclustering_multiplier_range_min': bin_config.get('min_multiplier', 0),
                            'misclustering_multiplier_range_max': bin_config.get('max_multiplier', 0),
                            'misclustering_bin_width': bin_config.get('bin_width', 0)
                        }
                        wandb.log(bin_config_log)

                        # Log 2D results: contamination_rate × multiplier_bin
                        total_2d_entries = 0
                        for condition_name, condition_data in misclustering_results['results_by_condition'].items():
                            if 'by_multiplier_bin' in condition_data:
                                for bin_name, bin_metrics in condition_data['by_multiplier_bin'].items():
                                    bin_log = {
                                        f'misclustering_{condition_name}_{bin_name}_mean_hamming': bin_metrics['mean_hamming'],
                                        f'misclustering_{condition_name}_{bin_name}_std_hamming': bin_metrics['std_hamming'],
                                        f'misclustering_{condition_name}_{bin_name}_mean_levenshtein': bin_metrics['mean_levenshtein'],
                                        f'misclustering_{condition_name}_{bin_name}_std_levenshtein': bin_metrics['std_levenshtein'],
                                        f'misclustering_{condition_name}_{bin_name}_num_examples': bin_metrics['num_examples'],
                                        f'misclustering_{condition_name}_{bin_name}_mean_cluster_size': bin_metrics['mean_cluster_size']
                                    }
                                    wandb.log(bin_log)
                                    total_2d_entries += 1

                        print(f"    Logged {total_2d_entries} 2D heatmap cells to WandB")

                    # Log results for each experimental condition (overall results)
                    for condition_name, condition_data in misclustering_results['results_by_condition'].items():
                        # Log overall metrics for this condition
                        metrics = condition_data['overall']
                        condition_log = {
                            f'misclustering_{condition_name}_mean_hamming': metrics['mean_hamming'],
                            f'misclustering_{condition_name}_std_hamming': metrics['std_hamming'],
                            f'misclustering_{condition_name}_mean_levenshtein': metrics['mean_levenshtein'],
                            f'misclustering_{condition_name}_std_levenshtein': metrics['std_levenshtein'],
                            f'misclustering_{condition_name}_num_examples': metrics['num_examples'],
                            f'misclustering_{condition_name}_success_rate_all': metrics['success_rate'],
                            f'misclustering_{condition_name}_failure_rate_all': metrics['failure_rate'],
                        }
                        wandb.log(condition_log)

                    # Log per-cluster size analysis
                    cluster_summary = {}
                    for cluster_size, results_list in misclustering_results['per_cluster_results'].items():
                        if results_list:
                            # Calculate average performance by condition for this cluster size
                            by_condition = defaultdict(list)
                            for result in results_list:
                                condition = result['condition']
                                by_condition[condition].append(result['hamming_distance'])

                            for condition, hamming_vals in by_condition.items():
                                if hamming_vals:
                                    cluster_summary[f'misclustering_cluster_{cluster_size}_{condition}_mean_hamming'] = np.mean(hamming_vals)
                                    cluster_summary[f'misclustering_cluster_{cluster_size}_{condition}_std_hamming'] = np.std(hamming_vals)

                    if cluster_summary:
                        wandb.log(cluster_summary)

                    print(f"Misclustering robustness results logged to WandB")

            save_worst_best = False
            if n_ex and save_worst_best:                                             
                worst_pct = cfg.data.get("worst_pct", 0.05)      # 5 % default
                by_N = defaultdict(list)
                for tup in all_results:              # Updated tuple format
                    by_N[tup[0]].append(tup)

                worst_examples = []
                for N, lst in by_N.items():
                    kN = max(1, int(math.ceil(len(lst) * worst_pct)))
                    worst_examples.extend(
                        sorted(lst, key=lambda t: t[5], reverse=True)[:kN]  # t[5] = norm. Lev
                    )

                # make one unified TSV that still records the cluster size in column 1
                fname = f"worst_perN_{int(worst_pct*100)}pct.tsv"
                worst_path = os.path.join(out_dir, fname)

                with open(worst_path, "w", newline="") as f:
                    writer = csv.writer(f, delimiter="\t")
                    if cross_mode:
                        writer.writerow(["cluster_size", "reads", "ground_truth_cropped",
                                        "prediction_cropped", "hamming_cropped",
                                        "levenshtein_cropped", "ground_truth_full",
                                        "prediction_full", "hamming_full", "levenshtein_full"])
                    else:
                        writer.writerow(["cluster_size", "reads", "ground_truth",
                                        "prediction", "hamming_dist",
                                        "levenshtein_dist"])
                    for tup in worst_examples:
                        # Handle new tuple format with both cropped and full metrics
                        N, reads, gt_crop, pred_crop, ham_crop, lev_crop = tup[0:6]
                        if cross_mode and len(tup) > 10:
                            # Include full metrics if available
                            gt_full, pred_full, ham_full, lev_full = tup[6:10]
                            writer.writerow([N, reads, gt_crop, pred_crop, ham_crop, lev_crop,
                                           gt_full, pred_full, ham_full, lev_full])
                        else:
                            writer.writerow([N, reads, gt_crop, pred_crop, ham_crop, lev_crop])

                print(f"Wrote worst {worst_pct:.0%} per‑N examples to {worst_path}")

                # save to W&B 
                art = wandb.Artifact("worst_perN_examples", type="analysis")
                art.add_file(worst_path)
                wandb.log_artifact(art)

                # BEST 5% per N
                best_pct = cfg.data.get("best_pct", 0.05)
                best_examples = []
                for N, lst in by_N.items():
                    kN = max(1, int(math.ceil(len(lst) * best_pct)))
                    best_examples.extend(
                        sorted(lst, key=lambda t: t[5])[:kN]     # smallest norm Lev
                    )

                fname_best = f"best_perN_{int(best_pct*100)}pct.tsv"
                best_path  = os.path.join(out_dir, fname_best)

                with open(best_path, "w", newline="") as f:
                    writer = csv.writer(f, delimiter="\t")
                    if cross_mode:
                        writer.writerow(["cluster_size", "reads", "ground_truth_cropped",
                                        "prediction_cropped", "hamming_cropped",
                                        "levenshtein_cropped", "ground_truth_full",
                                        "prediction_full", "hamming_full", "levenshtein_full"])
                    else:
                        writer.writerow(["cluster_size","reads","ground_truth",
                                        "prediction","hamming_dist","levenshtein_dist"])
                    for tup in best_examples:
                        # Handle new tuple format with both cropped and full metrics
                        N, reads, gt_crop, pred_crop, ham_crop, lev_crop = tup[0:6]
                        if cross_mode and len(tup) > 10:
                            # Include full metrics if available
                            gt_full, pred_full, ham_full, lev_full = tup[6:10]
                            writer.writerow([N, reads, gt_crop, pred_crop, ham_crop, lev_crop,
                                           gt_full, pred_full, ham_full, lev_full])
                        else:
                            writer.writerow([N, reads, gt_crop, pred_crop, ham_crop, lev_crop])

                art_best = wandb.Artifact("best_perN_examples", type="analysis")
                art_best.add_file(best_path)
                wandb.log_artifact(art_best)

        # Close log file if it was opened
        if log_file is not None:
            log_file.close()
            print(f"Closed majority voting log file")


if __name__ == "__main__":
    main()
