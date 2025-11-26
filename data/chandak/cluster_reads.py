#!/usr/bin/env python3
"""
Cluster noisy reads to ground truth sequences using unique prefix indexing.

For each experiment:
1. Find shortest unique prefix for each GT sequence
2. Search for these prefixes in noisy reads
3. Assign reads to GT based on prefix matches
4. Analyze cluster sizes and unmapped reads
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from functools import partial
import multiprocessing as mp
from Bio import SeqIO
import pandas as pd
import numpy as np
import editdistance
from tqdm import tqdm


def find_unique_prefixes(gt_sequences, min_len=10, max_len=50):
    """
    Find the shortest prefix length where all GT sequences have unique prefixes.

    Args:
        gt_sequences: list of SeqRecord objects
        min_len: minimum prefix length to try
        max_len: maximum prefix length to try

    Returns:
        dict: {prefix_string: (gt_index, gt_id)}
        int: prefix length used
    """
    print(f"  Finding unique prefixes (min_len={min_len}, max_len={max_len})...")

    for prefix_len in range(min_len, max_len + 1):
        prefixes = {}
        all_unique = True

        for idx, record in enumerate(gt_sequences):
            seq_str = str(record.seq)

            # Check if sequence is long enough
            if len(seq_str) < prefix_len:
                print(f"    Warning: Sequence {record.id} is only {len(seq_str)}bp, shorter than prefix_len={prefix_len}")
                all_unique = False
                break

            prefix = seq_str[:prefix_len]

            # Check if prefix already exists
            if prefix in prefixes:
                all_unique = False
                break

            prefixes[prefix] = (idx, record.id)

        if all_unique:
            print(f"  Found unique prefixes with length {prefix_len}bp")
            print(f"  Total unique prefixes: {len(prefixes)}")
            return prefixes, prefix_len

    raise ValueError(f"Could not find unique prefixes with length up to {max_len}bp. Sequences may be too similar.")


def search_prefix_in_read(read_seq, prefix, edit_dist_threshold=0):
    """
    Search for a prefix in a read sequence.

    Args:
        read_seq: string of read sequence
        prefix: string of prefix to search for
        edit_dist_threshold: maximum edit distance allowed

    Returns:
        tuple: (found, position, edit_distance)
    """
    prefix_len = len(prefix)

    if len(read_seq) < prefix_len:
        return False, -1, float('inf')

    # For exact match, use fast string search
    if edit_dist_threshold == 0:
        pos = read_seq.find(prefix)
        if pos != -1:
            return True, pos, 0
        else:
            return False, -1, float('inf')

    # For edit distance > 0, use sliding window
    best_dist = float('inf')
    best_pos = -1

    for i in range(len(read_seq) - prefix_len + 1):
        window = read_seq[i:i + prefix_len]
        dist = editdistance.eval(window, prefix)

        if dist < best_dist:
            best_dist = dist
            best_pos = i

        # Early exit if we found a match within threshold
        if dist <= edit_dist_threshold:
            return True, best_pos, dist

    # Check if best match is within threshold
    if best_dist <= edit_dist_threshold:
        return True, best_pos, best_dist
    else:
        return False, -1, best_dist


def cluster_reads_to_gt(noisy_reads, prefix_dict, prefix_len, edit_dist_threshold=0):
    """
    Cluster noisy reads to GT sequences based on prefix matches.

    Args:
        noisy_reads: list of SeqRecord objects
        prefix_dict: dict mapping prefix -> (gt_idx, gt_id)
        prefix_len: length of prefixes
        edit_dist_threshold: maximum edit distance for matching

    Returns:
        list of dicts with clustering results
    """
    print(f"  Clustering {len(noisy_reads):,} reads to GT using prefixes...")
    print(f"  Edit distance threshold: {edit_dist_threshold}")

    results = []

    for read in tqdm(noisy_reads, desc="  Processing reads"):
        read_seq = str(read.seq)
        read_id = read.id

        # Search for all GT prefixes in this read
        matches = []

        for prefix, (gt_idx, gt_id) in prefix_dict.items():
            found, pos, dist = search_prefix_in_read(read_seq, prefix, edit_dist_threshold)

            if found:
                matches.append({
                    'gt_idx': gt_idx,
                    'gt_id': gt_id,
                    'position': pos,
                    'edit_dist': dist
                })

        # Determine assignment
        if len(matches) == 0:
            # No match found
            results.append({
                'read_id': read_id,
                'gt_idx': -1,
                'gt_id': 'unmapped',
                'match_pos': -1,
                'edit_dist': -1,
                'status': 'unmapped',
                'num_matches': 0
            })
        elif len(matches) == 1:
            # Single match
            match = matches[0]
            results.append({
                'read_id': read_id,
                'gt_idx': match['gt_idx'],
                'gt_id': match['gt_id'],
                'match_pos': match['position'],
                'edit_dist': match['edit_dist'],
                'status': 'mapped',
                'num_matches': 1
            })
        else:
            # Multiple matches - pick best (lowest edit distance, then earliest position)
            best_match = min(matches, key=lambda x: (x['edit_dist'], x['position']))
            results.append({
                'read_id': read_id,
                'gt_idx': best_match['gt_idx'],
                'gt_id': best_match['gt_id'],
                'match_pos': best_match['position'],
                'edit_dist': best_match['edit_dist'],
                'status': 'mapped_ambiguous',
                'num_matches': len(matches)
            })

    return results


def analyze_clustering(results, num_gt_sequences):
    """
    Analyze clustering results and compute statistics.

    Args:
        results: list of clustering result dicts
        num_gt_sequences: total number of GT sequences

    Returns:
        dict of statistics
    """
    print("  Analyzing clustering results...")

    # Count reads per GT
    cluster_sizes = defaultdict(int)
    num_mapped = 0
    num_unmapped = 0
    num_ambiguous = 0

    for result in results:
        if result['status'] == 'unmapped':
            num_unmapped += 1
        else:
            num_mapped += 1
            cluster_sizes[result['gt_idx']] += 1
            if result['status'] == 'mapped_ambiguous':
                num_ambiguous += 1

    # Create array of cluster sizes (including 0s for uncovered GTs)
    cluster_size_array = [cluster_sizes.get(i, 0) for i in range(num_gt_sequences)]

    # Compute statistics
    uncovered_gts = sum(1 for size in cluster_size_array if size == 0)

    stats = {
        'total_reads': len(results),
        'num_mapped': num_mapped,
        'num_unmapped': num_unmapped,
        'num_ambiguous': num_ambiguous,
        'pct_mapped': 100 * num_mapped / len(results) if len(results) > 0 else 0,
        'pct_unmapped': 100 * num_unmapped / len(results) if len(results) > 0 else 0,
        'num_gt_sequences': num_gt_sequences,
        'uncovered_gts': uncovered_gts,
        'pct_uncovered_gts': 100 * uncovered_gts / num_gt_sequences if num_gt_sequences > 0 else 0,
        'cluster_sizes': cluster_size_array,
        'mean_cluster_size': np.mean(cluster_size_array),
        'median_cluster_size': np.median(cluster_size_array),
        'std_cluster_size': np.std(cluster_size_array),
        'min_cluster_size': np.min(cluster_size_array),
        'max_cluster_size': np.max(cluster_size_array),
    }

    # Print summary
    print(f"\n  Summary:")
    print(f"    Total reads: {stats['total_reads']:,}")
    print(f"    Mapped: {stats['num_mapped']:,} ({stats['pct_mapped']:.1f}%)")
    print(f"    Unmapped: {stats['num_unmapped']:,} ({stats['pct_unmapped']:.1f}%)")
    print(f"    Ambiguous: {stats['num_ambiguous']:,}")
    print(f"    Uncovered GTs: {stats['uncovered_gts']:,} / {stats['num_gt_sequences']:,} ({stats['pct_uncovered_gts']:.1f}%)")
    print(f"    Cluster size: {stats['mean_cluster_size']:.1f} ± {stats['std_cluster_size']:.1f}")
    print(f"    Cluster size range: {stats['min_cluster_size']} - {stats['max_cluster_size']}")

    return stats


def process_experiment(exp_id, processed_dir, output_dir, min_prefix_len, max_prefix_len, edit_dist_threshold):
    """
    Process a single experiment: find prefixes, cluster reads, analyze results.

    Args:
        exp_id: Experiment ID
        processed_dir: Input directory with gt.fa and reads.fastq
        output_dir: Output directory for clustering results
        ...
    """
    print("="*70)
    print(f"EXPERIMENT {exp_id}")
    print("="*70)

    # Input files
    input_exp_dir = Path(processed_dir) / f'experiment_{exp_id}'
    gt_file = input_exp_dir / 'gt.fa'
    reads_file = input_exp_dir / 'reads.fastq'

    # Output directory
    output_exp_dir = Path(output_dir) / f'experiment_{exp_id}'
    output_exp_dir.mkdir(parents=True, exist_ok=True)

    # Check if files exist
    if not gt_file.exists():
        print(f"  Warning: GT file not found: {gt_file}")
        return None

    if not reads_file.exists():
        print(f"  Warning: Reads file not found: {reads_file}")
        return None

    # Load GT sequences
    print(f"  Loading GT sequences from {gt_file.name}...")
    gt_sequences = list(SeqIO.parse(gt_file, 'fasta'))
    print(f"  Loaded {len(gt_sequences):,} GT sequences")

    # Load noisy reads
    print(f"  Loading noisy reads from {reads_file.name}...")
    noisy_reads = list(SeqIO.parse(reads_file, 'fastq'))
    print(f"  Loaded {len(noisy_reads):,} noisy reads")

    # Find unique prefixes
    prefix_dict, prefix_len = find_unique_prefixes(gt_sequences, min_prefix_len, max_prefix_len)

    # Save prefix information
    prefix_info = {
        'prefix_length': prefix_len,
        'num_prefixes': len(prefix_dict),
        'prefixes': {prefix: {'gt_idx': gt_idx, 'gt_id': gt_id}
                     for prefix, (gt_idx, gt_id) in prefix_dict.items()}
    }
    prefix_file = output_exp_dir / 'gt_prefixes.json'
    with open(prefix_file, 'w') as f:
        json.dump(prefix_info, f, indent=2)
    print(f"  Saved prefix info to {prefix_file.name}")

    # Cluster reads to GT
    clustering_results = cluster_reads_to_gt(noisy_reads, prefix_dict, prefix_len, edit_dist_threshold)

    # Save clustering results
    results_df = pd.DataFrame(clustering_results)
    results_file = output_exp_dir / 'clustering_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"  Saved clustering results to {results_file.name}")

    # Analyze clustering
    stats = analyze_clustering(clustering_results, len(gt_sequences))

    # Save statistics
    stats_file = output_exp_dir / 'cluster_stats.json'
    # Convert numpy types to python types for JSON serialization
    stats_json = {k: (v.tolist() if isinstance(v, np.ndarray) else
                      float(v) if isinstance(v, (np.floating, np.integer)) else v)
                  for k, v in stats.items()}
    with open(stats_file, 'w') as f:
        json.dump(stats_json, f, indent=2)
    print(f"  Saved statistics to {stats_file.name}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Cluster noisy reads to ground truth using prefix indexing')
    parser.add_argument('--processed-dir', type=str, required=True,
                        help='Path to processed_data directory')
    parser.add_argument('--edit-dist', type=int, default=0,
                        help='Edit distance threshold for matching (default: 0)')
    parser.add_argument('--min-prefix-len', type=int, default=10,
                        help='Minimum prefix length to try (default: 10)')
    parser.add_argument('--max-prefix-len', type=int, default=50,
                        help='Maximum prefix length to try (default: 50)')
    parser.add_argument('--experiments', type=str, default='all',
                        help='Comma-separated experiment IDs to process, or "all" (default: all)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of parallel workers for processing experiments (default: 1)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for clustering results (default: clustered_data_edit_dist_N)')

    args = parser.parse_args()

    # Set default output directory if not provided
    if args.output_dir is None:
        # Extract parent directory from processed_dir
        from pathlib import Path
        parent_dir = Path(args.processed_dir).parent
        args.output_dir = str(parent_dir / f'clustered_data_edit_dist_{args.edit_dist}')

    print("\n" + "="*70)
    print("CLUSTERING NOISY READS TO GROUND TRUTH")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Input directory: {args.processed_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Edit distance threshold: {args.edit_dist}")
    print(f"  Prefix length range: {args.min_prefix_len}-{args.max_prefix_len}")
    print(f"  Experiments: {args.experiments}")
    print(f"  Parallel workers: {args.num_workers}")
    print()

    # Determine which experiments to process
    if args.experiments == 'all':
        exp_ids = list(range(13))
    else:
        exp_ids = [int(x.strip()) for x in args.experiments.split(',')]

    # Process each experiment
    all_stats = []

    if args.num_workers == 1:
        # Sequential processing
        for exp_id in exp_ids:
            stats = process_experiment(
                exp_id,
                args.processed_dir,
                args.output_dir,
                args.min_prefix_len,
                args.max_prefix_len,
                args.edit_dist
            )
            if stats is not None:
                stats['exp_id'] = exp_id
                all_stats.append(stats)
            print()
    else:
        # Parallel processing
        print(f"Processing {len(exp_ids)} experiments in parallel with {args.num_workers} workers...\n")

        process_func = partial(
            process_experiment,
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            min_prefix_len=args.min_prefix_len,
            max_prefix_len=args.max_prefix_len,
            edit_dist_threshold=args.edit_dist
        )

        with mp.Pool(processes=args.num_workers) as pool:
            results = pool.map(process_func, exp_ids)

        for exp_id, stats in zip(exp_ids, results):
            if stats is not None:
                stats['exp_id'] = exp_id
                all_stats.append(stats)

        print()

    # Print overall summary
    print("="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    print(f"\nProcessed {len(all_stats)} experiments")

    if all_stats:
        total_reads = sum(s['total_reads'] for s in all_stats)
        total_mapped = sum(s['num_mapped'] for s in all_stats)
        total_unmapped = sum(s['num_unmapped'] for s in all_stats)

        print(f"Total reads: {total_reads:,}")
        print(f"Total mapped: {total_mapped:,} ({100*total_mapped/total_reads:.1f}%)")
        print(f"Total unmapped: {total_unmapped:,} ({100*total_unmapped/total_reads:.1f}%)")

    print("\nClustering complete!")


if __name__ == '__main__':
    main()
