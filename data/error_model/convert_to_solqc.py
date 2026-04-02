#!/usr/bin/env python3
"""
Convert TReconLM train.txt format to SOLQC input files.

Usage:
    python data/error_model/convert_to_solqc.py data/microsoft_data/data_microsoft/train.txt --output-dir data/error_model/microsoft
    python data/error_model/convert_to_solqc.py data/microsoft_data/data_microsoft/train.txt --output-dir data/error_model/microsoft --fraction 0.1

Input format (train.txt):
    read1|read2|...|readN:ground_truth

Output:
    design.csv         ground truth sequences (columns: barcode,sequence)
    reads_matching.csv pre-matched reads with alignment info (columns: sequence,count,variant_id,cigar_path)
"""

import argparse
import os
import random
from collections import Counter

import edlib
import re


# SOLQC cigar codes (from SOLQC/src/utils/biology.py)
CIGAR_DICT = {
    "=": "0",  # match
    "D": "1",  # deletion (base in query/read, not in target/variant)
    "I": "2",  # insertion (base in target/variant, not in query/read)
    "X": "3",  # mismatch/substitution
}


def parse_train_file(filepath):
    """Parse train.txt into list of (ground_truth, [reads]) tuples."""
    examples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            reads_part, gt = line.rsplit(':', 1)
            reads = reads_part.split('|')
            examples.append((gt, reads))
    return examples


def write_design_csv(gt_sequences, output_path):
    """Write ground truth sequences as SOLQC design CSV."""
    with open(output_path, 'w') as f:
        f.write('barcode,sequence\n')
        for i, seq in enumerate(gt_sequences):
            barcode = f'SEQ{i:06d}'
            f.write(f'{barcode},{seq}\n')


def compute_cigar_path(read, gt):
    """
    Align read to ground truth using edlib, exactly as SOLQC does internally.

    From SOLQC/src/library_reads/library_reads.py line 122:
        edlib.align(read_sequence, variant_sequence, task="path")

    From SOLQC/src/utils/biology.py:
        0 = match (=)
        1 = deletion (D)
        2 = insertion (I)
        3 = mismatch/substitution (X)
    """
    align = edlib.align(read, gt, task="path")
    cigar_str = align['cigar']

    # Parse edlib extended cigar into SOLQC's per-position format
    # (same as SOLQC/src/utils/biology.py parse_cigar)
    values = re.findall(r'\d+', cigar_str)
    chars = re.findall(r'[=IDX]', cigar_str)

    path = ""
    for val, char in zip(values, chars):
        path += CIGAR_DICT[char] * int(val)

    return path


def write_matching_csv(examples, output_path):
    """
    Write pre-matched reads as SOLQC matching CSV.

    Format: ,sequence,count,variant_id,cigar_path
    Since reads are already matched to ground truths in train.txt,
    variant_id is the index of the ground truth in the design CSV.
    """
    with open(output_path, 'w') as f:
        f.write(',sequence,count,variant_id,cigar_path\n')
        read_idx = 0
        for gt_idx, (gt, reads) in enumerate(examples):
            for read in reads:
                cigar = compute_cigar_path(read, gt)
                f.write(f'{read_idx},{read},1,{gt_idx},{cigar}\n')
                read_idx += 1


def main():
    parser = argparse.ArgumentParser(description='Convert train.txt to SOLQC input format')
    parser.add_argument('input_file', help='Path to train.txt file')
    parser.add_argument('--output-dir', required=True, help='Output directory for SOLQC files')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Fraction of data to use (e.g. 0.1 for 10%%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')

    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse input
    examples = parse_train_file(args.input_file)
    print(f"Loaded {len(examples)} examples from {args.input_file}")

    # Sample fraction
    if args.fraction < 1.0:
        n = max(1, int(len(examples) * args.fraction))
        examples = random.sample(examples, n)
        print(f"Sampled {len(examples)} examples ({args.fraction*100:.0f}%)")

    gt_sequences = [gt for gt, _ in examples]
    total_reads = sum(len(reads) for _, reads in examples)

    print(f"Ground truth sequences: {len(gt_sequences)}")
    print(f"Total reads: {total_reads}")

    # Write design CSV
    design_path = os.path.join(args.output_dir, 'design.csv')
    write_design_csv(gt_sequences, design_path)

    # Write pre-matched reads CSV with alignment
    matching_path = os.path.join(args.output_dir, 'reads_matching.csv')
    print(f"\nAligning reads to ground truths...")
    write_matching_csv(examples, matching_path)

    # Print design length for SOLQC library config
    lengths = Counter(len(s) for s in gt_sequences)
    most_common_len = lengths.most_common(1)[0][0]
    print(f"\nDesign length (for SOLQC library config): {most_common_len}")

    print(f"\nOutput files:")
    print(f"  {design_path}")
    print(f"  {matching_path}")
    print(f"\nIn SOLQC: upload design.csv as Design, reads_matching.csv as NGS (with 'matching' checkbox enabled)")


if __name__ == '__main__':
    main()
