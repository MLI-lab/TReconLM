#!/usr/bin/env python3
"""
Generate synthetic DNA storage data using DNAFormer's original IDS channel model.
Imports directly from DeepLearningBaselines/DNAFormer/DataGenerator/ (unmodified
code from https://github.com/itaiorr/Deep-DNA-based-storage).

This is an alternative synthetic data generator to TReconLM's error model
(estimate_error_model.py + data_generation_with_error_model()).

The output train.txt can be used directly for training or converted to reads.txt +
ground_truth.txt for inference comparison in cross_dataset.ipynb.

Usage (CLI):
    python data/error_model/generate_dnaformer_synthetic.py --dataset microsoft --num-clusters 5000 --output-dir data/error_model/results_dnaformer_synth_microsoft

Usage (from notebook):
    from data.error_model.generate_dnaformer_synthetic import generate
    generate(dataset='microsoft', num_clusters=5000, output_dir='path/to/output')

Technology mapping (only datasets with a matching DNAFormer profile):
    microsoft   | Twist + Nanopore MinION     | R21 (~4.3%)  | Good
    dnaformer   | Twist + MinION (short)      | B22 (~3.2%)  | Good

    Not supported (no matching DNAFormer profile):
    - noisy_dna: Photolithographic + Illumina NextSeq (~14.3%)
    - chandak: Custom Array + Nanopore (~10%)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

# Import DNAFormer's original DataGenerator (unmodified)
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / 'DeepLearningBaselines' / 'DNAFormer'))
from DataGenerator.error_rates_setup import ErrorRates
from DataGenerator.cluster_generator import ClusterGenerator


# Dataset -> (sequencing_tech, synthesis_tech, partial_flag, noise_coef) for ErrorRates.set_values()
DATASET_CONFIG = {
    'microsoft': {
        'sequencing_tech': 'MinION',
        'synthesis_tech': 'Twist Bioscience',
        'partial_flag': False,
        'noise_coef': None,
        'gt_length': 110,
        'description': 'R21: MinION + Twist Bioscience (Ross et al. 2021, ~4.3%)',
    },
    'dnaformer': {
        'sequencing_tech': 'MinIONShort',
        'synthesis_tech': 'Twist Bioscience',
        'partial_flag': False,
        'noise_coef': None,
        'gt_length': 128,  # 140 - 12 index
        'description': 'B22: MinIONShort + Twist Bioscience (Bar-Lev et al. 2022, ~3.2%)',
    },
}


def generate(dataset, num_clusters, output_dir, gt_length=None, min_copies=2,
             max_copies=10, delta=0.5, seed=42):
    """
    Generate synthetic data using DNAFormer's original IDS channel.

    Args:
        dataset: 'microsoft' or 'dnaformer' (selects technology profile)
        num_clusters: number of clusters to generate
        output_dir: where to save train.txt, reads.txt, ground_truth.txt
        gt_length: override GT length (default: from dataset config)
        min_copies: min reads per cluster
        max_copies: max reads per cluster
        delta: rate jittering (0=none, 0.5=50% std)
        seed: random seed

    Returns:
        dict with 'train_path', 'reads_path', 'gt_path', 'num_clusters', 'total_reads'
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    config = DATASET_CONFIG[dataset]
    gt_length = gt_length or config['gt_length']

    # Initialize error rates using DNAFormer's original ErrorRates class
    errors_prob = ErrorRates()
    errors_prob.set_values(
        config['sequencing_tech'],
        config['synthesis_tech'],
        config['partial_flag'],
        config['noise_coef']
    )

    total_err = sum(errors_prob.general_errors.values())
    print(f"Dataset: {dataset}")
    print(f"Profile: {config['description']}")
    print(f"Total error rate: {total_err:.4%}")
    print(f"GT length: {gt_length}, clusters: {num_clusters}, copies: {min_copies}-{max_copies}, delta: {delta}")

    # Generate clusters
    all_clusters = []
    all_gts = []
    total_reads = 0

    for i in range(num_clusters):
        gt = ''.join(random.choice('ACGT') for _ in range(gt_length))

        generator = ClusterGenerator(
            errors_prob.general_errors,
            errors_prob.per_base_errors,
            gt,
            min_copies,
            max_copies
        )
        generator.generate_cluster(delta=delta)

        all_clusters.append(generator.copies)
        all_gts.append(gt)
        total_reads += len(generator.copies)

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_clusters} clusters...")

    # Save train.txt (pipe-separated format)
    train_path = os.path.join(output_dir, 'train.txt')
    with open(train_path, 'w') as f:
        for copies, gt in zip(all_clusters, all_gts):
            f.write('|'.join(copies) + ':' + gt + '\n')

    # Save reads.txt + ground_truth.txt (for notebook inference)
    reads_path = os.path.join(output_dir, 'reads.txt')
    with open(reads_path, 'w') as f:
        for i, copies in enumerate(all_clusters):
            for read in copies:
                f.write(read + '\n')
            if i < len(all_clusters) - 1:
                f.write('===============================\n')

    gt_path = os.path.join(output_dir, 'ground_truth.txt')
    with open(gt_path, 'w') as f:
        for gt in all_gts:
            f.write(gt + '\n')

    # Save metadata
    meta = {
        'dataset': dataset,
        'description': config['description'],
        'sequencing_tech': config['sequencing_tech'],
        'synthesis_tech': config['synthesis_tech'],
        'gt_length': gt_length,
        'num_clusters': num_clusters,
        'min_copies': min_copies,
        'max_copies': max_copies,
        'delta': delta,
        'seed': seed,
        'general_error_rates': errors_prob.general_errors,
        'total_error_rate': total_err,
    }
    meta_path = os.path.join(output_dir, 'generation_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Done! {num_clusters} clusters ({total_reads} reads) -> {output_dir}/")

    return {
        'train_path': train_path,
        'reads_path': reads_path,
        'gt_path': gt_path,
        'num_clusters': num_clusters,
        'total_reads': total_reads,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic data using DNAFormer original IDS channel')
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--gt-length', type=int, default=None)
    parser.add_argument('--num-clusters', type=int, default=5000)
    parser.add_argument('--min-copies', type=int, default=2)
    parser.add_argument('--max-copies', type=int, default=10)
    parser.add_argument('--delta', type=float, default=0.5)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    generate(
        dataset=args.dataset,
        num_clusters=args.num_clusters,
        output_dir=args.output_dir,
        gt_length=args.gt_length,
        min_copies=args.min_copies,
        max_copies=args.max_copies,
        delta=args.delta,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
