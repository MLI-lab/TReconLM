#!/usr/bin/env python3
"""
Compute per-sequence nucleotide (A, C, T, G) distribution across all splits
(train, val, test) of a dataset, then report mean and std over all sequences.

Usage:
    python data/analyze_nucleotide_distribution.py data/noisy_dna/noisy_dna_data_storage/data/splits
    python data/analyze_nucleotide_distribution.py data/microsoft_data/data_microsoft
"""

import sys
import os
import numpy as np
from pathlib import Path


def parse_ground_truths(filepath):
    """Extract ground truth sequences from a split file (format: reads:gt)."""
    gts = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: read1|read2|...|readN:ground_truth
            if ':' in line:
                gt = line.rsplit(':', 1)[1]
            else:
                gt = line
            gts.append(gt)
    return gts


def nucleotide_fractions(seq):
    """Return [frac_A, frac_C, frac_G, frac_T] for a sequence."""
    n = len(seq)
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [seq.count(b) / n for b in 'ACGT']


def analyze_dataset(data_dir):
    data_dir = Path(data_dir)
    all_seqs = []

    # Collect ground truth sequences from all splits
    for split in ['train.txt', 'val.txt', 'test.txt']:
        filepath = data_dir / split
        if filepath.exists():
            seqs = parse_ground_truths(filepath)
            print(f"  {split}: {len(seqs)} sequences")
            all_seqs.extend(seqs)


    if not all_seqs:
        print(f"  No data found in {data_dir}")
        return None

    # Compute per-sequence nucleotide fractions
    fractions = np.array([nucleotide_fractions(seq) for seq in all_seqs])

    mean = fractions.mean(axis=0)
    std = fractions.std(axis=0)

    print(f"  Total sequences: {len(all_seqs)}")
    print(f"  {'Base':<6} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*24}")
    for i, base in enumerate('ACGT'):
        print(f"  {base:<6} {mean[i]:>8.4f} {std[i]:>8.4f}")

    return {'mean': dict(zip('ACGT', mean)), 'std': dict(zip('ACGT', std)), 'n': len(all_seqs)}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <data_dir> [<data_dir2> ...]")
        sys.exit(1)

    for data_dir in sys.argv[1:]:
        print(f"\n=== {data_dir} ===")
        analyze_dataset(data_dir)


if __name__ == '__main__':
    main()
