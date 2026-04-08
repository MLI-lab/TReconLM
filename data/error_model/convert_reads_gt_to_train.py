#!/usr/bin/env python3
"""
Convert reads.txt + ground_truth.txt format to train.txt (pipe-separated) format
for use with estimate_error_model.py.

Usage:
    python data/error_model/convert_reads_gt_to_train.py \
        data/dnaformer/without_index_data/BinnedNanoporeFirstFlowcell/reads.txt \
        data/dnaformer/without_index_data/BinnedNanoporeFirstFlowcell/ground_truth.txt \
        --output data/error_model/results_dnaformer_real/train.txt
"""

import argparse
import os


def convert(reads_path, gt_path, output_path):
    """Convert reads.txt + ground_truth.txt to train.txt format."""
    with open(reads_path) as f:
        content = f.read()

    separator = "==============================="
    clusters_raw = content.split(separator)
    clusters = []
    for c in clusters_raw:
        reads = [r.strip() for r in c.strip().split('\n') if r.strip()]
        if reads:
            clusters.append(reads)

    with open(gt_path) as f:
        gts = [line.strip() for line in f if line.strip()]

    if len(clusters) != len(gts):
        print(f"WARNING: {len(clusters)} clusters but {len(gts)} ground truths")
        n = min(len(clusters), len(gts))
        clusters = clusters[:n]
        gts = gts[:n]

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    total_reads = 0
    with open(output_path, 'w') as f:
        for reads, gt in zip(clusters, gts):
            line = '|'.join(reads) + ':' + gt
            f.write(line + '\n')
            total_reads += len(reads)

    print(f"Converted {len(clusters)} clusters ({total_reads} reads) -> {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert reads.txt + ground_truth.txt to train.txt')
    parser.add_argument('reads', help='Path to reads.txt')
    parser.add_argument('ground_truth', help='Path to ground_truth.txt')
    parser.add_argument('--output', '-o', required=True, help='Output train.txt path')
    args = parser.parse_args()
    convert(args.reads, args.ground_truth, args.output)
