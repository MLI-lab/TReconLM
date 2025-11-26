#!/usr/bin/env python3
"""
Primer Extraction and Demultiplexing for DNA Storage Data

This script extracts primers from ground truth and noisy reads,
demultiplexes reads by experiment, and removes primers to get clean payloads.

Usage:
    python extract_primers.py

Output:
    processed_data/
    ├── experiment_0/
    │   ├── gt.fa
    │   └── reads.fastq
    ├── experiment_1/
    │   ├── gt.fa
    │   └── reads.fastq
    ├── ... (through experiment_12)
    ├── primer_database.json
    ├── gt_metadata.csv
    ├── noisy_metadata.csv
    └── unmapped_reads.fastq
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict, Counter
import editdistance
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial

# Exact primers from encode_experiments.py
BARCODE_START = [
    'CTGGCTCCTCTGTATGTTGGAGAAT',  # 0
    'TGCGGATGCGGAAGTATGGTCCTCG',  # 1
    'AGTAACGCCTATTGATAACGAAGCA',  # 2
    'CTGGCGGCCTTGGCCGACTATCTGC',  # 3
    'TAGTCCGCGCTCGAATTCCGAGGCC',  # 4
    'ATGTTCGGAACGTCAAGACCGAGGA',  # 5
    'GCTAGTACGCGAACAGAGTGCAGTA',  # 6
    'CACCTGTGCTGCGTCAGGCTGTGTC',  # 7
    'CGTACAATCGTATTAGGCACCTTCC',  # 8
    'GTATACATTCCTTGCCAACATAGTA',  # 9
    'TATCGATTGCATGATACATCCGCAC',  # 10
    'GGCCTACCGAGGACCGCTTAGTAGG',  # 11
    'GATACTATCGAGATTACTCCAAGTC',  # 12
]

BARCODE_END = [
    'CCTATATGTACCTCTATCGTAAGTC',  # 0
    'CACTAGAAGCATGTCGCTATCGAGT',  # 1
    'TAACCTTCGCTGCTAGGAACTGTCT',  # 2
    'ACCATGTCGTACAGTCGTTGTAACA',  # 3
    'TACAAGACTACGCAAGATCGCGCTA',  # 4
    'TGGCTCCATTATGCTACAATCACTA',  # 5
    'ACAGATGCAGTAATTCTCACGAACT',  # 6
    'GCTGTCCGTTCCGCATTGACACGGC',  # 7
    'GCGGACCTCCAGATCCACTTGTCTG',  # 8
    'TGAATCTGGATACGCGTTCCTCAAC',  # 9
    'GACCTGTGGAAGTTCCTCATTACTA',  # 10
    'CCTATCATGAATTAGATGCTTGGAC',  # 11
    'GCTAGTCGATCCTCTGCTGCAATCG',  # 12
]

# Encoding parameters from encode_experiments.py
RS_REDUNDANCY = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.4, 0.3, 0.3]
CONV_M = [8, 11, 14, 8, 11, 14, 8, 11, 14, 11, 11, 11, 11]
CONV_R = [1, 1, 1, 3, 3, 3, 5, 5, 5, 3, 3, 3, 3]
PAD = [False, False, False, False, False, False, False, False, True, False, False, False, False]
BYTES_PER_OLIGO = [10, 10, 10, 18, 18, 18, 20, 20, 20, 18, 18, 18, 18]


def reverse_complement(seq):
    """Return reverse complement of DNA sequence"""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))


def find_primer_in_region(read_region, primer):
    """
    Find primer in a read region using sliding window and edit distance.

    Like helper.py:find_barcode_pos_in_post, this finds the position with
    MINIMUM edit distance, without any threshold. We just pick the best match.

    Args:
        read_region: DNA sequence region to search
        primer: Primer sequence to find

    Returns:
        (best_position, min_distance)
    """
    primer_len = len(primer)
    min_dist = float('inf')
    best_pos = -1

    # Slide window across region
    for i in range(max(0, len(read_region) - primer_len + 1)):
        window = read_region[i:i + primer_len]
        dist = editdistance.eval(window, primer)
        if dist < min_dist:
            min_dist = dist
            best_pos = i

    return best_pos, min_dist


def find_best_primer_match(read_seq, primer_db):
    """
    Find best matching primer pair and orientation for a read.

    Following helper.py:find_barcode_pos_in_post approach:
    - Search FIRST HALF of read for forward primer
    - Search SECOND HALF of read for reverse primer
    - Pick position with MINIMUM edit distance (no threshold)
    - Compare forward vs reverse complement, pick better orientation

    Key insight: If forward primer found at position 7, we need to:
    - Remove bases 0-6 (junk before primer)
    - Remove the primer itself (7 to 7+len(primer))
    - Payload starts after that

    Args:
        read_seq: DNA sequence of the read
        primer_db: Dictionary of {exp_id: {'forward': seq, 'reverse': seq}}

    Returns:
        Dictionary with match information or None if no good match found
    """
    best_match = None
    best_total_dist = float('inf')

    # Try both orientations
    for orientation in ['forward', 'reverse_complement']:
        if orientation == 'reverse_complement':
            test_seq = reverse_complement(read_seq)
        else:
            test_seq = read_seq

        read_len = len(test_seq)

        # Try each experiment's primers
        for exp_id, primers in primer_db.items():
            fwd_primer = primers['forward']
            rev_primer = primers['reverse']
            fwd_primer_len = len(fwd_primer)
            rev_primer_len = len(rev_primer)

            # Skip if read too short
            if read_len < fwd_primer_len + rev_primer_len:
                continue

            # Search FIRST HALF for forward primer (like their implementation)
            first_half_end = read_len // 2 + 1
            start_region = test_seq[:first_half_end]

            # Search SECOND HALF for reverse primer
            second_half_start = read_len // 2
            end_region = test_seq[second_half_start:]

            # Find forward primer in first half
            fwd_pos_in_region, fwd_dist = find_primer_in_region(start_region, fwd_primer)

            # Find reverse primer in second half
            rev_pos_in_region, rev_dist = find_primer_in_region(end_region, rev_primer)

            # Calculate absolute positions
            fwd_pos = fwd_pos_in_region  # Already absolute since we search from start
            rev_pos = second_half_start + rev_pos_in_region  # Adjust for second half offset

            total_dist = fwd_dist + rev_dist

            # Calculate payload boundaries
            payload_start = fwd_pos + fwd_primer_len
            payload_end = rev_pos

            # Sanity check: reverse primer should come after forward primer
            if payload_end <= payload_start:
                continue  # Invalid: primers overlap or in wrong order

            # Update best match if this is better
            if total_dist < best_total_dist:
                best_total_dist = total_dist
                best_match = {
                    'exp_id': int(exp_id),
                    'orientation': orientation,
                    'sequence': test_seq,
                    'fwd_pos': fwd_pos,  # Position where forward primer starts
                    'rev_pos': rev_pos,  # Position where reverse primer starts
                    'fwd_dist': fwd_dist,
                    'rev_dist': rev_dist,
                    'total_dist': total_dist,
                    'fwd_primer_len': fwd_primer_len,
                    'rev_primer_len': rev_primer_len,
                    'payload_start': payload_start,  # Start of payload (after forward primer)
                    'payload_end': payload_end,  # End of payload (before reverse primer)
                }

    return best_match


def process_ground_truth(oligo_dir, output_dir, primer_db):
    """
    Process ground truth sequences to extract payloads.

    For GT, primers should be at exact positions (start=0, end=-25),
    but we verify this.
    """
    print("\n" + "="*60)
    print("PROCESSING GROUND TRUTH SEQUENCES")
    print("="*60 + "\n")

    gt_metadata = []

    for exp_id in range(13):
        # Create experiment directory
        exp_dir = f'{output_dir}/experiment_{exp_id}'
        os.makedirs(exp_dir, exist_ok=True)
        oligo_file = f'{oligo_dir}/oligos_{exp_id}.fa'

        if not os.path.exists(oligo_file):
            print(f"Warning: {oligo_file} not found")
            continue

        print(f"Processing experiment {exp_id}...")

        exp_seqs = []
        fwd_primer = primer_db[exp_id]['forward']
        rev_primer = primer_db[exp_id]['reverse']

        for record in SeqIO.parse(oligo_file, 'fasta'):
            full_seq = str(record.seq)

            # Verify primers are at expected positions
            if full_seq.startswith(fwd_primer) and full_seq.endswith(rev_primer):
                # Standard case: primers at expected positions
                payload = full_seq[len(fwd_primer):-len(rev_primer)]
                primer_positions_correct = True
            else:
                # Find primers with edit distance (shouldn't happen for GT, but be safe)
                match = find_best_primer_match(
                    full_seq,
                    {exp_id: primer_db[exp_id]}
                )

                if match:
                    payload = match['sequence'][match['payload_start']:match['payload_end']]
                    primer_positions_correct = False
                    print(f"  Warning: Primers not at expected positions for {record.id}")
                else:
                    print(f"  Error: Could not find primers for {record.id}")
                    continue

            exp_seqs.append({
                'id': record.id,
                'exp': exp_id,
                'full_seq': full_seq,
                'payload': payload,
                'forward_primer': fwd_primer,
                'reverse_primer': rev_primer,
                'full_length': len(full_seq),
                'payload_length': len(payload),
                'primers_at_expected_pos': primer_positions_correct
            })

        # Write payload sequences to experiment directory
        output_file = f'{exp_dir}/gt.fa'
        with open(output_file, 'w') as f:
            for seq_data in exp_seqs:
                f.write(f">{seq_data['id']}_payload\n")
                f.write(f"{seq_data['payload']}\n")

        gt_metadata.extend(exp_seqs)

        payload_lengths = [s['payload_length'] for s in exp_seqs]
        print(f"  Processed {len(exp_seqs)} sequences")
        print(f"  Payload length: {np.mean(payload_lengths):.1f} ± {np.std(payload_lengths):.1f} bp")

    # Save metadata
    gt_df = pd.DataFrame(gt_metadata)
    gt_df.to_csv(f'{output_dir}/gt_metadata.csv', index=False)

    print(f"\nTotal GT sequences processed: {len(gt_metadata)}")
    print(f"Metadata saved to: {output_dir}/gt_metadata.csv")

    return gt_df


def process_single_read(record, primer_db):
    """
    Process a single read - designed to be called by multiprocessing workers.

    Returns:
        Tuple of (result_type, data) where:
        - result_type: 'mapped', 'unmapped_short', 'unmapped_invalid', 'unmapped_not_found'
        - data: depends on result_type
    """
    read_seq = str(record.seq)

    # Skip very short reads
    if len(read_seq) < 80:
        return ('unmapped_short', record)

    # Find best primer match
    match = find_best_primer_match(read_seq, primer_db)

    if not match:
        return ('unmapped_not_found', record)

    # Extract payload using the positions found
    payload_seq = match['sequence'][match['payload_start']:match['payload_end']]

    # Sanity check payload length
    if len(payload_seq) < 50 or len(payload_seq) > 200:
        return ('unmapped_invalid', record)

    # Get quality scores for payload region
    if match['orientation'] == 'reverse_complement':
        # Reverse quality scores to match reversed sequence
        all_quals_reversed = record.letter_annotations['phred_quality'][::-1]
        payload_qual = all_quals_reversed[match['payload_start']:match['payload_end']]
    else:
        payload_qual = record.letter_annotations['phred_quality'][match['payload_start']:match['payload_end']]

    # Create payload record
    payload_record = SeqRecord(
        Seq(payload_seq),
        id=f"{record.id}_exp{match['exp_id']}_{match['orientation'][:3]}",
        description=f"len={len(payload_seq)} edist={match['total_dist']} fwd_pos={match['fwd_pos']} rev_pos={match['rev_pos']}",
        letter_annotations={'phred_quality': payload_qual}
    )

    # Return mapped result with metadata
    metadata = {
        'read_id': record.id,
        'exp_id': match['exp_id'],
        'orientation': match['orientation'],
        'fwd_pos': match['fwd_pos'],
        'rev_pos': match['rev_pos'],
        'fwd_dist': match['fwd_dist'],
        'rev_dist': match['rev_dist'],
        'total_dist': match['total_dist'],
        'payload_length': len(payload_seq),
        'original_length': len(read_seq)
    }

    return ('mapped', (payload_record, metadata))


def process_noisy_reads(fastq_file, output_dir, primer_db, num_workers=None):
    """
    Process noisy reads to extract payloads and demultiplex by experiment.

    Using multiprocessing to parallelize across CPU cores.

    Following helper.py approach:
    - No edit distance threshold (just pick best match)
    - Search first half for forward primer, second half for reverse
    - Compare forward vs RC orientation, pick better one
    - Accept all reads (let decoder handle bad ones downstream)

    Args:
        num_workers: Number of parallel workers (default: all CPU cores)
    """
    print("\n" + "="*60)
    print("PROCESSING NOISY READS (PARALLEL)")
    print("="*60 + "\n")

    if num_workers is None:
        num_workers = mp.cpu_count()

    print(f"Using {num_workers} worker processes\n")

    # Statistics tracking
    read_metadata = []
    stats = {
        'total_reads': 0,
        'mapped_reads': 0,
        'unmapped_reads': 0,
        'by_experiment': defaultdict(int),
        'by_orientation': defaultdict(int),
        'edit_distances': [],
        'reads_too_short': 0,
        'primers_not_found': 0,
        'invalid_payload': 0
    }

    # Open output files for each experiment (in their respective directories)
    exp_files = {}
    for exp_id in range(13):
        exp_dir = f'{output_dir}/experiment_{exp_id}'
        os.makedirs(exp_dir, exist_ok=True)
        exp_files[exp_id] = open(f'{exp_dir}/reads.fastq', 'w')

    unmapped_file = open(f'{output_dir}/unmapped_reads.fastq', 'w')

    print(f"Processing reads from: {fastq_file}")
    print(f"This may take a while for large files...\n")

    start_time = time.time()

    # Count total reads for progress bar
    print("Counting total reads...")
    total_reads = sum(1 for _ in SeqIO.parse(fastq_file, 'fastq'))
    print(f"Total reads to process: {total_reads:,}\n")

    # Create partial function with primer_db baked in
    process_func = partial(process_single_read, primer_db=primer_db)

    # Process reads in parallel with progress bar
    with mp.Pool(processes=num_workers) as pool:
        # Use imap for ordered results with progress bar
        results = pool.imap(process_func, SeqIO.parse(fastq_file, 'fastq'), chunksize=100)

        # Process results with progress bar
        for result_type, data in tqdm(results, total=total_reads, desc="Processing reads"):
            stats['total_reads'] += 1

            if result_type == 'mapped':
                payload_record, metadata = data

                # Write to experiment-specific file
                SeqIO.write(payload_record, exp_files[metadata['exp_id']], 'fastq')

                # Track statistics
                stats['mapped_reads'] += 1
                stats['by_experiment'][metadata['exp_id']] += 1
                stats['by_orientation'][metadata['orientation']] += 1
                stats['edit_distances'].append(metadata['total_dist'])

                # Save metadata
                read_metadata.append(metadata)

            elif result_type == 'unmapped_short':
                stats['unmapped_reads'] += 1
                stats['reads_too_short'] += 1
                SeqIO.write(data, unmapped_file, 'fastq')

            elif result_type == 'unmapped_invalid':
                stats['unmapped_reads'] += 1
                stats['invalid_payload'] += 1
                SeqIO.write(data, unmapped_file, 'fastq')

            elif result_type == 'unmapped_not_found':
                stats['unmapped_reads'] += 1
                stats['primers_not_found'] += 1
                SeqIO.write(data, unmapped_file, 'fastq')

    # Close all files
    for f in exp_files.values():
        f.close()
    unmapped_file.close()

    elapsed_time = time.time() - start_time

    # Print final statistics
    print(f"\n{'='*60}")
    print("PROCESSING STATISTICS")
    print(f"{'='*60}")
    print(f"Total reads: {stats['total_reads']:,}")
    print(f"Mapped reads: {stats['mapped_reads']:,} ({stats['mapped_reads']/stats['total_reads']*100:.2f}%)")
    print(f"Unmapped reads: {stats['unmapped_reads']:,} ({stats['unmapped_reads']/stats['total_reads']*100:.2f}%)")
    print(f"  - Too short: {stats['reads_too_short']:,}")
    print(f"  - Primers not found: {stats['primers_not_found']:,}")
    print(f"  - Invalid payload: {stats['invalid_payload']:,}")

    print(f"\nReads per experiment:")
    for exp_id in sorted(stats['by_experiment'].keys()):
        count = stats['by_experiment'][exp_id]
        print(f"  Exp {exp_id:2d}: {count:,} reads")

    print(f"\nOrientation distribution:")
    for orient, count in stats['by_orientation'].items():
        print(f"  {orient:20s}: {count:,} reads ({count/stats['mapped_reads']*100:.2f}%)")

    if stats['edit_distances']:
        print(f"\nEdit distance statistics:")
        print(f"  Mean:   {np.mean(stats['edit_distances']):.2f}")
        print(f"  Median: {np.median(stats['edit_distances']):.2f}")
        print(f"  Std:    {np.std(stats['edit_distances']):.2f}")
        print(f"  Min:    {np.min(stats['edit_distances'])}")
        print(f"  Max:    {np.max(stats['edit_distances'])}")

    print(f"\nProcessing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Reads per second: {stats['total_reads']/elapsed_time:.1f}")

    # Save statistics
    stats_dict = {
        'total_reads': stats['total_reads'],
        'mapped_reads': stats['mapped_reads'],
        'unmapped_reads': stats['unmapped_reads'],
        'mapping_rate': stats['mapped_reads'] / stats['total_reads'] * 100 if stats['total_reads'] > 0 else 0,
        'reads_too_short': stats['reads_too_short'],
        'primers_not_found': stats['primers_not_found'],
        'invalid_payload': stats['invalid_payload'],
        'by_experiment': dict(stats['by_experiment']),
        'by_orientation': dict(stats['by_orientation']),
        'avg_edit_distance': float(np.mean(stats['edit_distances'])) if stats['edit_distances'] else 0,
        'median_edit_distance': float(np.median(stats['edit_distances'])) if stats['edit_distances'] else 0,
        'processing_time_seconds': elapsed_time
    }

    with open(f'{output_dir}/processing_statistics.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)

    # Save read metadata
    read_df = pd.DataFrame(read_metadata)
    read_df.to_csv(f'{output_dir}/read_metadata.csv', index=False)

    print(f"\nOutput files saved to: {output_dir}")

    return read_df, stats_dict


def main():
    parser = argparse.ArgumentParser(description='Extract primers from DNA storage data')
    parser.add_argument('--oligo-dir', default='/Users/macpc770036980/DNAEmbedding/Tryout/nanopore_dna_storage_data/oligo_files',
                        help='Directory containing ground truth oligo files')
    parser.add_argument('--fastq-file', default='/Users/macpc770036980/DNAEmbedding/Tryout/nanopore_dna_storage_data/fastq/merged.fastq',
                        help='FASTQ file containing noisy reads')
    parser.add_argument('--output-dir', default='/Users/macpc770036980/DNAEmbedding/Tryout/processed_data',
                        help='Output directory for processed files')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: all CPU cores)')
    parser.add_argument('--skip-gt', action='store_true',
                        help='Skip ground truth processing')
    parser.add_argument('--skip-noisy', action='store_true',
                        help='Skip noisy reads processing')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build primer database from exact sequences
    print("Building primer database from encode_experiments.py...")
    primer_db = {}
    for exp_id in range(13):
        primer_db[exp_id] = {
            'forward': BARCODE_START[exp_id],
            'reverse': BARCODE_END[exp_id],
            'rs_redundancy': RS_REDUNDANCY[exp_id],
            'conv_m': CONV_M[exp_id],
            'conv_r': CONV_R[exp_id],
            'pad': PAD[exp_id],
            'bytes_per_oligo': BYTES_PER_OLIGO[exp_id]
        }

    # Save primer database
    with open(f'{args.output_dir}/primer_database.json', 'w') as f:
        json.dump(primer_db, f, indent=2)

    print(f"\nPrimer database saved to: {args.output_dir}/primer_database.json")
    print(f"Number of experiments: {len(primer_db)}")

    # Process ground truth
    if not args.skip_gt:
        gt_df = process_ground_truth(args.oligo_dir, args.output_dir, primer_db)

    # Process noisy reads
    if not args.skip_noisy:
        read_df, stats = process_noisy_reads(
            args.fastq_file,
            args.output_dir,
            primer_db,
            num_workers=args.num_workers
        )

    print("\n" + "="*60)
    print("PRIMER EXTRACTION COMPLETE")
    print("="*60)
    print(f"\nAll output files saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - experiment_*/gt.fa: Primer-free ground truth per experiment")
    print("  - experiment_*/reads.fastq: Demultiplexed noisy reads per experiment")
    print("  - primer_database.json: Primer sequences and encoding params")
    print("  - gt_metadata.csv: Ground truth sequence metadata")
    print("  - noisy_metadata.csv: Noisy read processing metadata")
    print("  - unmapped_reads.fastq: Reads without identifiable primers")
    print("\nReady for error correction and decoding!")


if __name__ == '__main__':
    main()
