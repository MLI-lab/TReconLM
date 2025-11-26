import os
import sys
import time
import random
from datetime import datetime
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import pytz
import gzip
import wandb

from src.utils.hamming_distance import hamming_distance_postprocessed


def extract_elements(pairs, indicator):
    """
    Extracts the second elements from a list of key-value pairs where the key matches the specified indicator.

    Parameters:
        pairs (List[Tuple[Any, Any]]): A list of 2-element pairs, typically structured as (label, value).
        indicator (Any): The key to match (e.g., "CPRED", "ground_truth").

    Returns:
        List[Any]: A list of all values (second elements) where the key matches the given indicator.

    Example:
         pairs = [
             ("ground_truth", "ACGT"),
             ("CPRED", "T--T"),
             ("CPRED", "T--G"),
             ("ground_truth", "TGCA"),
         ]
         extract_elements(pairs, "CPRED") gives ['T--T', 'T--G']
    """
    return [pair[1] for pair in pairs if pair[0] == indicator]


def safe_download_artifact(entity, project, artifact_name, max_retries=3):
    """
    Download wandb artifact with retries.

    Parameters:
        entity (str): WandB entity name
        project (str): WandB project name
        artifact_name (str): Artifact name to download
        max_retries (int): Maximum number of retry attempts

    Returns:
        str: Path to downloaded artifact directory

    Raises:
        RuntimeError: If download fails after all retries
    """
    for attempt in range(1, max_retries + 1):
        try:
            art = wandb.use_artifact(f'{entity}/{project}/{artifact_name}:latest', type='dataset')
            return art.download()
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to download {artifact_name} after {max_retries} attempts")


def split_list(lst, n):
    """
    Split a list into n roughly equal chunks.

    Parameters:
        lst (list): The list to split
        n (int): Number of chunks

    Returns:
        list: List of n sublists
    """
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i,m):(i+1)*k + min(i+1,m)] for i in range(n)]


def is_distributed():
    """Check if running in distributed mode (multi-GPU with torchrun)."""
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ


def calculate_baseline_error_rate(cfg):
    """
    Calculate baseline error rate from IDS channel config.

    Parameters:
        cfg: Hydra config object

    Returns:
        dict: Dictionary with 'error_rate_per_nt' and other baseline info
    """
    # Get IDS channel parameters from config
    sub_lb = cfg.data.get('substitution_probability_lb', 0.01)
    ins_lb = cfg.data.get('insertion_probability_lb', 0.01)
    del_lb = cfg.data.get('deletion_probability_lb', 0.01)

    sub_ub = cfg.data.get('substitution_probability_ub', 0.1)
    ins_ub = cfg.data.get('insertion_probability_ub', 0.1)
    del_ub = cfg.data.get('deletion_probability_ub', 0.1)

    # Use midpoint as baseline
    avg_sub = (sub_lb + sub_ub) / 2
    avg_ins = (ins_lb + ins_ub) / 2
    avg_del = (del_lb + del_ub) / 2

    # Total error rate per nucleotide
    error_rate_per_nt = avg_sub + avg_ins + avg_del

    # Get ground truth length from config
    ground_truth_length = cfg.data.get('ground_truth_length', 110)

    # Calculate expected total edit distance
    expected_total_edit_distance = error_rate_per_nt * ground_truth_length

    return {
        'error_rate_per_nt': error_rate_per_nt,
        'substitution_rate': avg_sub,
        'insertion_rate': avg_ins,
        'deletion_rate': avg_del,
        'ground_truth_length': ground_truth_length,
        'expected_total_edit_distance': expected_total_edit_distance
    }


def contaminate_trace_cluster(traces, ground_truth, contamination_rate, baseline_error_rate, cfg, rng):
    """
    Contaminate a trace cluster by replacing some traces with noisy versions from wrong ground truth.

    Parameters:
        traces (list): List of trace strings
        ground_truth (str): Actual ground truth sequence
        contamination_rate (float): Probability that each trace gets contaminated (0.0-1.0)
        baseline_error_rate (float): Base error rate per nucleotide
        cfg: Hydra config object
        rng: numpy RandomState for reproducibility

    Returns:
        tuple: (contaminated_traces, contamination_info dict)
    """
    num_traces = len(traces)

    # Probabilistic per-trace contamination: each trace independently sampled
    contaminate_indices = []
    for i in range(num_traces):
        if rng.random() < contamination_rate:
            contaminate_indices.append(i)

    num_to_contaminate = len(contaminate_indices)

    if num_to_contaminate == 0:
        return traces, {
            'num_contaminants': 0,
            'contamination_rate': 0.0,
            'avg_edit_distance': 0.0,
            'contaminated_positions': []
        }

    contaminated_traces = list(traces)
    edit_distances = []
    contaminated_positions = []

    for idx in contaminate_indices:
        # Generate a random wrong ground truth (different from actual)
        gt_len = len(ground_truth)
        wrong_gt = ''.join(rng.choice(['A', 'C', 'G', 'T'], size=gt_len))

        # Generate noisy trace from wrong ground truth
        # Simple IDS channel simulation
        noisy_trace = []
        for nt in wrong_gt:
            # Substitution
            if rng.random() < baseline_error_rate / 3:
                noisy_trace.append(rng.choice(['A', 'C', 'G', 'T']))
            else:
                noisy_trace.append(nt)

            # Insertion
            if rng.random() < baseline_error_rate / 3:
                noisy_trace.append(rng.choice(['A', 'C', 'G', 'T']))

        # Deletion (skip nucleotides)
        final_trace = []
        for nt in noisy_trace:
            if rng.random() > baseline_error_rate / 3:
                final_trace.append(nt)

        contaminated_trace = ''.join(final_trace) if final_trace else 'A'  # Ensure non-empty
        contaminated_traces[idx] = contaminated_trace

        # Calculate edit distance
        try:
            from Levenshtein import distance as levenshtein_distance
            edit_dist = levenshtein_distance(traces[idx], contaminated_trace)
            edit_distances.append(edit_dist)
        except:
            edit_dist = 0
            edit_distances.append(0)

        # Track contaminated position info
        # Calculate multiplier as ratio of edit distance to expected baseline
        expected_edit_dist = len(traces[idx]) * baseline_error_rate
        multiplier = edit_dist / expected_edit_dist if expected_edit_dist > 0 else 0.0

        contaminated_positions.append({
            'trace_index': int(idx),
            'edit_distance': edit_dist,
            'realized_edit_distance_multiplier': multiplier
        })

    return contaminated_traces, {
        'num_contaminants': num_to_contaminate,
        'contamination_rate': num_to_contaminate / num_traces,
        'avg_edit_distance': sum(edit_distances) / len(edit_distances) if edit_distances else 0.0,
        'contaminated_positions': contaminated_positions
    }


def create_multiplier_bins(experiment_results):
    """
    Add multiplier bins to experiment results for analysis.

    Parameters:
        experiment_results (dict): Experiment results dictionary

    Returns:
        dict: Updated results with multiplier bins
    """
    # This function groups contamination results by "error multiplier"
    # (how much the error increased compared to baseline)
    num_bins = experiment_results.get('multiplier_bin_config', {}).get('num_bins', 10)
    experiment_results['multiplier_bins'] = {
        'num_bins': num_bins,
        'bin_edges': [i * 0.5 for i in range(num_bins + 1)],  # 0.0, 0.5, 1.0, ...
        'results_by_bin': {}
    }
    return experiment_results


def create_cluster_size_bins(experiment_results):
    """
    Add cluster size bins to experiment results for analysis.

    Parameters:
        experiment_results (dict): Experiment results dictionary

    Returns:
        dict: Updated results with cluster size bins
    """
    # Group results by cluster size for analysis
    experiment_results['cluster_size_bins'] = {}
    return experiment_results


def exchange_positional_encoding(model, checkpoint_path, model_type):
    """
    Exchange positional encoding from another checkpoint.

    Parameters:
        model: The model to modify
        checkpoint_path (str): Path to checkpoint with different positional encoding
        model_type (str): Type of model (gpt, lstm, mamba)
    """
    import torch
    print(f"Loading positional encoding from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Extract positional encoding weights
        if model_type == 'gpt':
            if 'transformer.wpe.weight' in state_dict:
                model.transformer.wpe.weight.data = state_dict['transformer.wpe.weight'].to(model.transformer.wpe.weight.device)
                print("Successfully exchanged positional encoding")
            else:
                print("Warning: No positional encoding found in checkpoint")
        else:
            print(f"Positional encoding exchange not implemented for {model_type}")

    except Exception as e:
        print(f"Error exchanging positional encoding: {e}")


def save_contaminated_attention_data(attention_sequence, read_boundaries, token_sequence,
                                      example_idx, cluster_size, prediction, ground_truth,
                                      levenshtein_distance, contamination_info, output_dir):
    """
    Save attention data for contaminated examples.

    Parameters:
        attention_sequence: Attention weights
        read_boundaries: Boundaries of reads in sequence
        token_sequence: Token indices
        example_idx (int): Example index
        cluster_size (int): Number of traces in cluster
        prediction (str): Model prediction
        ground_truth (str): Ground truth sequence
        levenshtein_distance (int): Edit distance
        contamination_info (dict): Info about contamination applied
        output_dir (str): Output directory path
    """
    import json
    os.makedirs(output_dir, exist_ok=True)

    filename = f"contaminated_ex{example_idx}_cs{cluster_size}_lev{levenshtein_distance}.json"
    filepath = os.path.join(output_dir, filename)

    data = {
        'example_idx': example_idx,
        'cluster_size': cluster_size,
        'prediction': prediction,
        'ground_truth': ground_truth,
        'levenshtein_distance': levenshtein_distance,
        'contamination_info': contamination_info,
        'read_boundaries': read_boundaries if isinstance(read_boundaries, list) else read_boundaries.tolist(),
        'token_sequence': token_sequence if isinstance(token_sequence, list) else token_sequence.tolist()
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def create_folder(folder_path):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def convert_to_fasta(sequences: List[str], name: str) -> List[str]:

    """
    Converts a list of sequences to a list in .fasta format.
    
    Args:
    sequences (list): The list of sequences.

    Returns:
    list: The list of sequences in FASTA format.

    EXAMPLE:
    INPUT:  ['ATT', 'ACC', 'AGG']
    OUTPUT: ['>sequence0', 'ATT', '>sequence1', 'ACC', '>sequence2', 'AGG']
    """

    fasta_format = []
    for i, sequence in enumerate(sequences):
        fasta_format.append(f'>{name}_seq_{i+1}')
        fasta_format.append(sequence)
        
    return fasta_format

def write_fasta(sequences_fasta, filepath):
    
    """
    Writes a list of sequences in .fasta format to a file.

    Args:
    sequences_fasta (list): The list of sequences in .fasta format.
    filepath (str): The path to the file to write the sequences to.

    Returns:
    """

    with open(filepath, "w") as fasta_file:
        for line in sequences_fasta:
            fasta_file.write(line + "\n")

    
def read_fasta(filepath):

    """
    Reads a .fasta file and returns the sequences in a list.

    Args:
    filepath (str): The path to the .fasta file.

    Returns:
    list: The list of sequences in the .fasta file.
    
    EXAMPLE:
    INPUT:  'test.fasta'
    ...
    """

    sequences = []
    current_sequence = ""

    with open(filepath, "r") as file:
        for line in file:
            if line.startswith('>'):
                if current_sequence:
                    sequences.append(current_sequence)
                current_sequence = ""
            else:
                current_sequence += line.strip()

    # Add the last sequence
    if current_sequence:
        sequences.append(current_sequence)

    return sequences


def create_fasta_file(sequences, name, file_path):
    """
    This file creates a .fasta file out of one training example of the following form obs1|obs2:algn1|algn2.
    Based on the indicator, the function extracts the observed sequences or the alignment sequences.

    Args:
    test_data_example (str): The training example of the form obs1|obs2:algn1|algn2.
    data_path (str): The path including the filename to the file to write the .fasta file to.
    indicator (int): The indicator to extract the observed sequences or the alignment sequences.

    Returns:
    -
    """
    list_fasta = convert_to_fasta(sequences = sequences, name = name)
    write_fasta(sequences_fasta = list_fasta, filepath = file_path)


def remove_dash_tokens(lst: List[str]) -> List[str]:

    """
    Removes all '-' tokens from a list of strings.
    
    Args:
    lst (List[str]): The list of strings to remove the '-' tokens from.
    
    Returns: 
    List[str]: The list of strings with the '-' tokens removed.
    """
    
    return [''.join(ch for ch in string if ch != '-') for string in lst]

def filter_string(s: str) -> str:

    """
    Filters a string to only contain the characters 'A', 'C', 'T', and 'G'.

    Args:
    s (str): The string to filter.

    Returns:
    str: The filtered string.
    """

    return ''.join(c for c in s if c in 'ACTG')

def append_fasta_seqs(sequences: List[str], index: int, data_path: str) -> None: 

    """
    Appends the sequences to a .fasta file.

    Args:
    sequences (list): The list of sequences to append to the .fasta file.
    index (int): The index of the sequences.
    data_path (str): The path to the .fasta file to append the sequences to.

    Returns:
    -
    """

    list_fasta = convert_to_fasta(sequences = sequences, name = f'index_{index}')
    with open(data_path, 'a') as file:
        file.write('\n')  # Write a blank line
        for line in list_fasta:
            file.write(line + '\n')  # Write each sequence line by line


def get_now_str():
    timezone = pytz.timezone('Europe/Berlin')
    now = datetime.now(timezone)
    now_str = now.strftime('%Y%m%d_%H%M%S')
    return now_str

def get_repo_path(script_dir, n):
    dir_n_levels_up = script_dir
    for _ in range(n):
        dir_n_levels_up = os.path.dirname(dir_n_levels_up)

    return dir_n_levels_up

def tensor_to_str(x_tensor, itos):
    return "".join(itos[int(i)] for i in x_tensor.tolist())

def permute_traces_in_tensor(x_tensor, itos, stoi, rng):
    """
    Permute the order of traces/reads within a single tensor example.

    The tensor format is: read1|read2|read3|...:ground_truth####
    This function randomly shuffles the reads while keeping the ground truth intact.

    Args:
        x_tensor: Input tensor containing encoded sequence
        itos: Index to string mapping (for decoding)
        stoi: String to index mapping (for encoding)
        rng: Random number generator (e.g., np.random.RandomState)

    Returns:
        Permuted tensor with the same shape
    """
    import torch

    # Decode tensor to string
    decoded = tensor_to_str(x_tensor, itos)

    # Split on ':' to separate reads from ground truth and padding
    parts = decoded.split(':', 1)
    if len(parts) != 2:
        # No colon found, return original tensor
        return x_tensor

    reads_part, rest = parts

    # Split reads by '|'
    reads = reads_part.split('|')

    # Permute reads
    permuted_reads = list(reads)  # Make a copy
    rng.shuffle(permuted_reads)

    # Reconstruct the sequence
    permuted_sequence = '|'.join(permuted_reads) + ':' + rest

    # Re-encode to tensor
    encoded = [stoi.get(ch, stoi.get('<unk>', 0)) for ch in permuted_sequence]
    permuted_tensor = torch.tensor(encoded, dtype=x_tensor.dtype)

    return permuted_tensor

def create_subclusters(
    clusters: List[List[str]],
    ground_truths: List[str],
    max_reads: Optional[int] = 10,
    truncate: bool = False,
) -> Tuple[List[str], List[int]]:
    """
    Build examples from read clusters.

    Rules:
        If max_reads is None → never subsample; use full cluster (if ≥2 reads).
        If truncate=True:
            use random.sample(max_reads) only if cluster > max_reads; else full cluster.
        If truncate=False:
            if cluster <= max_reads, use full cluster;
            if cluster > max_reads, split into random subclusters of size between 2 and max_reads
              until <2 remain (discard singletons).

    Returns:
        examples: ["read1|read2|...|readN:ground_truth", ...]
        new_cluster_sizes: [N1, N2, ...]
    """
    examples: List[str] = []
    new_cluster_sizes: List[int] = []

    for reads, gt in zip(clusters, ground_truths):
        if not reads:
            continue

        reads_copy = reads.copy()
        random.shuffle(reads_copy)
        n = len(reads_copy)

        # Case: never subsample
        if max_reads is None:
            if n >= 2:
                examples.append("|".join(reads_copy) + ":" + gt)
                new_cluster_sizes.append(n)
            continue

        # max_reads is an int -------------------------------------------------
        if truncate:
            sampled = random.sample(reads_copy, max_reads) if n > max_reads else reads_copy
            if len(sampled) >= 2:
                examples.append("|".join(sampled) + ":" + gt)
                new_cluster_sizes.append(len(sampled))
        else:
            if n <= max_reads:
                # small enough: keep whole cluster
                if n >= 2:
                    examples.append("|".join(reads_copy) + ":" + gt)
                    new_cluster_sizes.append(n)
            else:
                # oversized: split into random-sized subclusters
                reads_remaining = reads_copy
                while len(reads_remaining) >= 2:
                    sub_size = random.randint(2, min(max_reads, len(reads_remaining)))
                    sub_reads = reads_remaining[:sub_size]
                    reads_remaining = reads_remaining[sub_size:]
                    examples.append("|".join(sub_reads) + ":" + gt)
                    new_cluster_sizes.append(len(sub_reads))
                # discard leftover 0/1
    return examples, new_cluster_sizes


def file_to_list(path):
    """Reads a plain .txt file containing one sequence per line."""
    with open(path, "r") as f:
        lines = f.read().splitlines()
    return lines


def fastq_to_list(path):
    """Reads a .fastq or .fastq.gz file and extracts the sequences."""
    open_fn = gzip.open if path.endswith(".gz") else open
    sequences = []
    with open_fn(path, "rt") as f:
        line_idx = 0
        for line in f:
            if line_idx % 4 == 1:  # sequence line in FASTQ
                sequences.append(line.strip())
            line_idx += 1
    return sequences
        
def decode_dna_index(index_dna):
    assert index_dna.startswith("ACAAC"), f"Invalid prefix in index: {index_dna}"
    payload = index_dna[5:]  # last 7 nt = 14 bits
    binary_str = dna_to_binary(payload)
    int_val = int(binary_str, 2)
    return binary_str, int_val

def dna_to_binary(dna):
    mapping = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}
    return ''.join(mapping[nt] for nt in dna)


def cluster_by_index_region(reads, decoded_ints, threshold=0, start_window=42, end_window=54, print_flag=False):
    """
    Cluster noisy reads by extracting embedded indices via prefix-hamming-distance logic.

    Args:
        reads (list): List of noisy DNA reads.
        decoded_ints (iterable): Set of decoded index integers from ground truth sequences.
        threshold (int): Max allowed Hamming distance to prefix 'ACAAC'.
        start_window (int): Sliding window start position.
        end_window (int): Sliding window end position.
        print_flag (bool): Whether to print progress messages.

    Returns:
        clusters (dict): Mapping from decoded integer index to list of read **sequences**.
        failed (int): Number of reads that could not be clustered.
    """
    valid_ints = set(decoded_ints)
    clusters = defaultdict(list)
    failed = 0
    start_time = time.time()

    for i, read in enumerate(reads):
        if i % 10**6 == 0 and print_flag:
            print(f"Processing read {i:.2e}")

        found = False
        for j in range(start_window, end_window - 12 + 1):
            candidate = read[j:j+12]
            if len(candidate) != 12:
                continue

            prefix = candidate[:5]
            payload = candidate[5:]

            if hamming_distance_postprocessed(prefix, "ACAAC") <= threshold and all(nt in "ACGT" for nt in candidate):
                try:
                    binary_str = dna_to_binary(payload)
                    int_val = int(binary_str, 2)
                    if int_val in valid_ints:
                        clusters[int_val].append(read)  # store the actual read sequence
                        found = True
                        break
                except Exception as e:
                    if print_flag:
                        print(f"Failed to decode {candidate} at read {i}, pos {j}: {e}")
        if not found:
            failed += 1

    elapsed = round(time.time() - start_time, 2)
    print(f"Number of reads that could not be clustered is {failed}/{len(reads)} ({100 * failed / len(reads):.2f}%)")
    print(f"Clustering time taken: {elapsed} seconds")

    return clusters, failed

if __name__ == "__main__":
    
    print("Running helper_functions.py")


    