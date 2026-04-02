#!/usr/bin/env python3
"""
Estimate error model from real-world DNA storage data.

Usage:
    python data/error_model/estimate_error_model.py data/microsoft_data/data_microsoft/train.txt
    python data/error_model/estimate_error_model.py data/microsoft_data/data_microsoft/train.txt --fraction 0.1
    python data/error_model/estimate_error_model.py data/microsoft_data/data_microsoft/train.txt --output-dir data/error_model/results_microsoft

Analyses:
    1. Overall error rates (sub/del/ins)
    2. 4x4 substitution matrix
    3. Per-nucleotide error rates (sub/del/ins conditioned on reference base)
    4. Homopolymer error analysis (rates by homopolymer length 2, 3, 4, 5+)
    5. Position-dependent error rates
    6. Error burst length distributions (consecutive errors of same type)
    7. GC content vs error rate
    8. Read length distribution
    9. Edit distance distribution

All with statistical tests (chi-squared / binomial CIs).
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

import sys
from pathlib import Path

import edlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add repo root to path so we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.helper_functions import compute_homopolymer_map


BASES = ['A', 'C', 'G', 'T']


# ============================================================
# Parsing
# ============================================================

def parse_train_file(filepath):
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


# ============================================================
# Alignment
# ============================================================

def align_read_to_gt(read, gt):
    """
    Align read to ground truth using edlib.
    Returns list of (op, gt_base_or_None, read_base_or_None) tuples.

    op is one of: '=', 'X', 'D', 'I'
        = : match
        X : substitution
        D : deletion (base in gt, gap in read)
        I : insertion (gap in gt, base in read)
    """
    # edlib finds the optimal alignment between read and gt using edit distance (minimum insertions + deletions + substitutions to transform one into the other)
    # uses a dynamic programming algorithm (like Needleman-Wunsch but optimized)
    result = edlib.align(read, gt, task="path") # path to give exact alignment 
    cigar = result['cigar']
    # takes the raw alignment and produces a human-readable version with three strings
    # query_aligned: the read, with - gaps inserted where deletions happened
    # target_aligned (the gt, with - gaps inserted where insertions happened)
    # matched_aligned (shows | for matches, . for mismatches) 
    nice = edlib.getNiceAlignment(result, read, gt)

    ops = []
    for q, t in zip(nice['query_aligned'], nice['target_aligned']):
        if q == '-':
            # deletion: base in target(gt), gap in query(read)
            ops.append(('D', t, None))
        elif t == '-':
            # insertion: gap in target(gt), base in query(read)
            ops.append(('I', None, q))
        elif q == t:
            ops.append(('=', t, q))
        else:
            ops.append(('X', t, q))
    # each position a tuple like ('D', 'G', None) or ('X', 'C', 'T')
    return ops


# ============================================================
# Statistics collection
# ============================================================

def collect_statistics(examples):
    """Process all examples and collect raw counts for every analysis."""

    # Overall counts
    total_matches = 0
    total_subs = 0
    total_dels = 0
    total_ins = 0
    total_bases = 0  # gt bases (for del/sub rate denominator)

    # Substitution matrix: from_base -> to_base -> count
    sub_matrix = {b: {b2: 0 for b2 in BASES} for b in BASES} # sub_matrix['A']['G'] = X means "A was substituted to G X times across all reads

    # Per-nucleotide counts: base -> {matches, subs, dels, ins, total}
    per_nt = {b: {'matches': 0, 'subs': 0, 'dels': 0, 'ins': 0, 'total': 0} for b in BASES}

    # Position-dependent: pos -> {subs, dels, ins, total}
    pos_stats = defaultdict(lambda: {'subs': 0, 'dels': 0, 'ins': 0, 'total': 0})

    # Homopolymer: (length_bucket, error_type) -> {errors, total}
    # length_bucket: 1 (non-homopolymer), 2, 3, 4, '5+'
    homo_stats = {}
    for lb in [1, 2, 3, 4, '5+']:
        homo_stats[lb] = {'subs': 0, 'dels': 0, 'ins': 0, 'total': 0}

    # Error burst lengths
    del_bursts = []
    ins_bursts = []
    sub_bursts = []

    # GC content per read -> error rates by type
    gc_error_data = []  # list of (gc_fraction, sub_rate, del_rate, ins_rate)

    # Read length distribution
    read_lengths = []

    # Edit distance distribution
    edit_distances = []

    # Position-dependent nucleotide frequencies in ground truth
    # pos -> base -> count (for generating realistic synthetic GTs)
    gt_pos_nt = defaultdict(lambda: {b: 0 for b in BASES})

    # Joint statistics: (nucleotide, homopolymer_bucket, gc_bin, position_zone) -> {subs, dels, ins, total}
    # gc_bin: 'low' (<0.30), 'mid' (0.30-0.70), 'high' (>0.70)
    # position_zone: 'start' (0-9), 'middle' (10 to L-11), 'end' (L-10 to L-1)
    joint_stats = defaultdict(lambda: {'subs': 0, 'dels': 0, 'ins': 0, 'total': 0})

    for gt, reads in examples:
        gc_frac = (gt.count('G') + gt.count('C')) / len(gt)
        gc_bin = 'low' if gc_frac < 0.30 else ('high' if gc_frac > 0.70 else 'mid')

        # Count nucleotide at each position in this GT
        for pos, base in enumerate(gt):
            if base in BASES:
                gt_pos_nt[pos][base] += 1
        # Precompute homopolymer map for this ground truth
        homo_map = compute_homopolymer_map(gt)
        gt_len = len(gt)

        for read in reads:
            read_lengths.append(len(read))
            ops = align_read_to_gt(read, gt)

            # Edit distance
            n_errors = sum(1 for op, _, _ in ops if op != '=')
            edit_distances.append(n_errors)

            # GC content per-error-type rates (gc_frac already computed above per GT)
            n_sub = sum(1 for op, _, _ in ops if op == 'X')
            n_del = sum(1 for op, _, _ in ops if op == 'D')
            n_ins = sum(1 for op, _, _ in ops if op == 'I')
            gc_error_data.append((gc_frac, n_sub / gt_len, n_del / gt_len, n_ins / gt_len))

            # Walk through alignment ops
            gt_pos = 0
            i = 0
            while i < len(ops):
                op, gt_base, read_base = ops[i]

                # Compute joint key for this position
                if gt_pos < gt_len:
                    hl = homo_map[gt_pos] if gt_pos < len(homo_map) else 1
                    zone = get_position_zone(gt_pos, gt_len)
                    jkey = (gt_base if gt_base else (gt[gt_pos] if gt_pos < gt_len else 'A'),
                            hl, gc_bin, zone)
                else:
                    hl = 1
                    zone = 'end'
                    jkey = None

                if op == '=':
                    total_matches += 1
                    total_bases += 1
                    per_nt[gt_base]['matches'] += 1
                    per_nt[gt_base]['total'] += 1
                    pos_stats[gt_pos]['total'] += 1
                    homo_stats[hl]['total'] += 1
                    if jkey:
                        joint_stats[jkey]['total'] += 1
                    gt_pos += 1

                elif op == 'X':
                    total_subs += 1
                    total_bases += 1
                    sub_matrix[gt_base][read_base] += 1
                    per_nt[gt_base]['subs'] += 1
                    per_nt[gt_base]['total'] += 1
                    pos_stats[gt_pos]['subs'] += 1
                    pos_stats[gt_pos]['total'] += 1
                    homo_stats[hl]['subs'] += 1
                    homo_stats[hl]['total'] += 1
                    if jkey:
                        joint_stats[jkey]['subs'] += 1
                        joint_stats[jkey]['total'] += 1
                    gt_pos += 1

                elif op == 'D':
                    total_dels += 1
                    total_bases += 1
                    per_nt[gt_base]['dels'] += 1
                    per_nt[gt_base]['total'] += 1
                    pos_stats[gt_pos]['dels'] += 1
                    pos_stats[gt_pos]['total'] += 1
                    homo_stats[hl]['dels'] += 1
                    homo_stats[hl]['total'] += 1
                    if jkey:
                        joint_stats[jkey]['dels'] += 1
                        joint_stats[jkey]['total'] += 1
                    gt_pos += 1

                elif op == 'I':
                    total_ins += 1
                    # Insertion: attribute to nearest gt_base for per-nt stats
                    if gt_pos > 0:
                        ref_base = gt[gt_pos - 1]
                    elif gt_pos < len(gt):
                        ref_base = gt[gt_pos]
                    else:
                        ref_base = 'A'
                    per_nt[ref_base]['ins'] += 1
                    if gt_pos < len(gt):
                        pos_stats[gt_pos]['ins'] += 1
                    # Homopolymer: attribute to nearest gt position
                    nearest_pos = min(gt_pos, len(gt) - 1)
                    hl_ins = homo_map[nearest_pos] if nearest_pos < len(homo_map) else 1
                    homo_stats[hl_ins]['ins'] += 1
                    if jkey:
                        joint_stats[jkey]['ins'] += 1

                i += 1

            # Collect burst lengths from this alignment
            collect_bursts(ops, del_bursts, ins_bursts, sub_bursts)

    return {
        'total_matches': total_matches,
        'total_subs': total_subs,
        'total_dels': total_dels,
        'total_ins': total_ins,
        'total_bases': total_bases,
        'sub_matrix': sub_matrix,
        'per_nt': per_nt,
        'pos_stats': dict(pos_stats),
        'homo_stats': homo_stats,
        'del_bursts': del_bursts,
        'ins_bursts': ins_bursts,
        'sub_bursts': sub_bursts,
        'gc_error_data': gc_error_data,
        'read_lengths': read_lengths,
        'edit_distances': edit_distances,
        'gt_pos_nt': dict(gt_pos_nt),
        'joint_stats': dict(joint_stats),
    }



def get_position_zone(pos, seq_len, zone_size=10):
    """Classify position into start/middle/end zone."""
    if pos < zone_size:
        return 'start'
    elif pos >= seq_len - zone_size:
        return 'end'
    else:
        return 'middle'


def collect_bursts(ops, del_bursts, ins_bursts, sub_bursts):
    """Extract consecutive error burst lengths from alignment ops."""
    i = 0
    while i < len(ops):
        op = ops[i][0]
        if op in ('D', 'I', 'X'):
            j = i + 1
            while j < len(ops) and ops[j][0] == op:
                j += 1
            burst_len = j - i
            if op == 'D':
                del_bursts.append(burst_len)
            elif op == 'I':
                ins_bursts.append(burst_len)
            elif op == 'X':
                sub_bursts.append(burst_len)
            i = j
        else:
            i += 1


# ============================================================
# Statistical tests
# ============================================================

def binomial_ci(successes, trials, confidence=0.95):
    """Wilson score interval for binomial proportion."""
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denom
    spread = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0, center - spread), min(1, center + spread))


def chi2_test_rates(counts_dict):
    """
    Chi-squared test for whether error rates differ across categories.
    counts_dict: {category: (errors, total)}
    Returns: chi2 statistic, p-value
    """
    categories = list(counts_dict.keys())
    observed_errors = [counts_dict[c][0] for c in categories]
    observed_ok = [counts_dict[c][1] - counts_dict[c][0] for c in categories]
    table = np.array([observed_errors, observed_ok])
    if table.min() < 0 or table.sum() == 0:
        return 0.0, 1.0
    chi2, p, _, _ = stats.chi2_contingency(table)
    return chi2, p


# ============================================================
# Reporting
# ============================================================

def print_report(s, output_dir):
    """Print summary and save JSON."""
    total = s['total_bases']
    total_with_ins = total + s['total_ins']

    print("\n" + "=" * 70)
    print("ERROR MODEL ESTIMATION REPORT")
    print("=" * 70)

    # 1. Overall rates
    sub_rate = s['total_subs'] / total if total > 0 else 0
    del_rate = s['total_dels'] / total if total > 0 else 0
    ins_rate = s['total_ins'] / total if total > 0 else 0

    print(f"\n--- 1. Overall Error Rates ---")
    print(f"  Total GT bases:    {total:,}")
    print(f"  Substitutions:     {s['total_subs']:,}  ({sub_rate:.4%})")
    print(f"  Deletions:         {s['total_dels']:,}  ({del_rate:.4%})")
    print(f"  Insertions:        {s['total_ins']:,}  ({ins_rate:.4%})")
    print(f"  Total error rate:  {(sub_rate + del_rate + ins_rate):.4%}")

    # 2. Substitution matrix
    print(f"\n--- 2. Substitution Matrix (row=from, col=to) ---")
    print(f"  {'':>6s}", end='')
    for b in BASES:
        print(f"  {b:>8s}", end='')
    print()
    for b1 in BASES:
        row_total = sum(s['sub_matrix'][b1][b2] for b2 in BASES if b2 != b1)
        print(f"  {b1:>6s}", end='')
        for b2 in BASES:
            if b1 == b2:
                print(f"  {'---':>8s}", end='')
            else:
                count = s['sub_matrix'][b1][b2]
                rate = count / row_total if row_total > 0 else 0
                print(f"  {rate:>8.4f}", end='')
        print(f"  (n={row_total})")

    # 3. Per-nucleotide error rates
    print(f"\n--- 3. Per-Nucleotide Error Rates ---")
    print(f"  {'Base':>6s} {'Sub rate':>10s} {'Del rate':>10s} {'Ins rate':>10s} {'Total obs':>12s}")
    nt_test_sub = {}
    nt_test_del = {}
    nt_test_ins = {}
    for b in BASES:
        t = s['per_nt'][b]['total']
        sr = s['per_nt'][b]['subs'] / t if t > 0 else 0
        dr = s['per_nt'][b]['dels'] / t if t > 0 else 0
        ir = s['per_nt'][b]['ins'] / t if t > 0 else 0
        ci_s = binomial_ci(s['per_nt'][b]['subs'], t)
        ci_d = binomial_ci(s['per_nt'][b]['dels'], t)
        ci_i = binomial_ci(s['per_nt'][b]['ins'], t)
        print(f"  {b:>6s} {sr:>10.4%} {dr:>10.4%} {ir:>10.4%} {t:>12,}")
        print(f"  {'':>6s} [{ci_s[0]:.4%},{ci_s[1]:.4%}] [{ci_d[0]:.4%},{ci_d[1]:.4%}] [{ci_i[0]:.4%},{ci_i[1]:.4%}]")
        nt_test_sub[b] = (s['per_nt'][b]['subs'], t)
        nt_test_del[b] = (s['per_nt'][b]['dels'], t)
        nt_test_ins[b] = (s['per_nt'][b]['ins'], t)

    chi2_s, p_s = chi2_test_rates(nt_test_sub)
    chi2_d, p_d = chi2_test_rates(nt_test_del)
    chi2_i, p_i = chi2_test_rates(nt_test_ins)
    print(f"\n  Chi-squared test (rates differ across bases?):")
    print(f"    Substitution: chi2={chi2_s:.1f}, p={p_s:.2e} {'***' if p_s < 0.001 else '**' if p_s < 0.01 else '*' if p_s < 0.05 else 'n.s.'}")
    print(f"    Deletion:     chi2={chi2_d:.1f}, p={p_d:.2e} {'***' if p_d < 0.001 else '**' if p_d < 0.01 else '*' if p_d < 0.05 else 'n.s.'}")
    print(f"    Insertion:    chi2={chi2_i:.1f}, p={p_i:.2e} {'***' if p_i < 0.001 else '**' if p_i < 0.01 else '*' if p_i < 0.05 else 'n.s.'}")

    # 4. Homopolymer analysis
    print(f"\n--- 4. Homopolymer Error Rates ---")
    print(f"  {'Length':>8s} {'Sub rate':>10s} {'Del rate':>10s} {'Ins rate':>10s} {'Total obs':>12s}")
    homo_test_sub = {}
    homo_test_del = {}
    homo_test_ins = {}
    for lb in [1, 2, 3, 4, '5+']:
        h = s['homo_stats'][lb]
        t = h['total']
        sr = h['subs'] / t if t > 0 else 0
        dr = h['dels'] / t if t > 0 else 0
        ir = h['ins'] / t if t > 0 else 0
        label = str(lb) if lb != 1 else '1 (non-hp)'
        print(f"  {label:>8s} {sr:>10.4%} {dr:>10.4%} {ir:>10.4%} {t:>12,}")
        homo_test_sub[lb] = (h['subs'], t)
        homo_test_del[lb] = (h['dels'], t)
        homo_test_ins[lb] = (h['ins'], t)

    chi2_hs, p_hs = chi2_test_rates(homo_test_sub)
    chi2_hd, p_hd = chi2_test_rates(homo_test_del)
    chi2_hi, p_hi = chi2_test_rates(homo_test_ins)
    print(f"\n  Chi-squared test (rates differ by homopolymer length?):")
    print(f"    Substitution: chi2={chi2_hs:.1f}, p={p_hs:.2e} {'***' if p_hs < 0.001 else 'n.s.'}")
    print(f"    Deletion:     chi2={chi2_hd:.1f}, p={p_hd:.2e} {'***' if p_hd < 0.001 else 'n.s.'}")
    print(f"    Insertion:    chi2={chi2_hi:.1f}, p={p_hi:.2e} {'***' if p_hi < 0.001 else 'n.s.'}")

    # Pairwise: each homopolymer length vs non-homopolymer (length=1)
    print(f"\n  Pairwise vs non-homopolymer (length=1):")
    ref = s['homo_stats'][1]
    ref_total = ref['total']
    for lb in [2, 3, 4, '5+']:
        h = s['homo_stats'][lb]
        t = h['total']
        if t == 0 or ref_total == 0:
            continue
        for err_type in ['subs', 'dels', 'ins']:
            rate_ref = ref[err_type] / ref_total
            rate_hp = h[err_type] / t
            ratio = rate_hp / rate_ref if rate_ref > 0 else float('inf')
            table = np.array([[h[err_type], t - h[err_type]],
                              [ref[err_type], ref_total - ref[err_type]]])
            _, p = stats.chi2_contingency(table)[:2]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            print(f"    hp={str(lb):>3s} {err_type:>4s}: {rate_hp:.4%} vs {rate_ref:.4%} (ratio={ratio:.2f}x, p={p:.2e} {sig})")

    # 5. Burst length distributions
    print(f"\n--- 5. Error Burst Length Distributions ---")
    for name, bursts in [('Deletion', s['del_bursts']), ('Insertion', s['ins_bursts']), ('Substitution', s['sub_bursts'])]:
        if not bursts:
            print(f"  {name}: no bursts")
            continue
        c = Counter(bursts)
        total_bursts = sum(c.values())
        print(f"  {name} bursts (n={total_bursts:,}):")
        for length in sorted(c.keys())[:10]:
            print(f"    length {length}: {c[length]:,} ({c[length]/total_bursts:.4%})")

    # 6. Summary stats
    print(f"\n--- 6. Read Length Distribution ---")
    rl = np.array(s['read_lengths'])
    print(f"  Mean: {rl.mean():.1f}, Std: {rl.std():.1f}, Min: {rl.min()}, Max: {rl.max()}")

    print(f"\n--- 7. Edit Distance Distribution ---")
    ed = np.array(s['edit_distances'])
    print(f"  Mean: {ed.mean():.2f}, Std: {ed.std():.2f}, Median: {np.median(ed):.0f}")

    # 8. GT nucleotide distribution by position
    if s['gt_pos_nt']:
        positions = sorted(s['gt_pos_nt'].keys())
        print(f"\n--- 8. GT Nucleotide Distribution by Position ---")
        print(f"  Positions: 0 to {max(positions)}")
        # Show global average
        global_counts = {b: 0 for b in BASES}
        global_total = 0
        for pos in positions:
            for b in BASES:
                global_counts[b] += s['gt_pos_nt'][pos][b]
                global_total += s['gt_pos_nt'][pos][b]
        print(f"  Global: " + ", ".join(f"{b}={global_counts[b]/global_total:.4f}" for b in BASES))
        # Show a few positions
        for pos in [0, 1, max(positions)//2, max(positions)-1, max(positions)]:
            if pos in s['gt_pos_nt']:
                t = sum(s['gt_pos_nt'][pos].values())
                freqs = ", ".join(f"{b}={s['gt_pos_nt'][pos][b]/t:.3f}" for b in BASES)
                print(f"  Pos {pos:>3d}: {freqs}")

    # 9. Multipliers and sanity check
    print(f"\n--- 9. Multipliers ---")

    # Homopolymer multipliers (relative to non-homopolymer)
    ref_hp = s['homo_stats'][1]
    print(f"\n  Homopolymer multipliers (vs non-homopolymer):")
    print(f"  {'HP len':>8s} {'Sub mult':>10s} {'Del mult':>10s} {'Ins mult':>10s}")
    for lb in [2, 3, 4, '5+']:
        h = s['homo_stats'][lb]
        for err in ['subs', 'dels', 'ins']:
            pass
        ref_rates = {e: ref_hp[e] / ref_hp['total'] if ref_hp['total'] > 0 else 0 for e in ['subs', 'dels', 'ins']}
        hp_rates = {e: h[e] / h['total'] if h['total'] > 0 else 0 for e in ['subs', 'dels', 'ins']}
        mults = {e: hp_rates[e] / ref_rates[e] if ref_rates[e] > 0 else 1.0 for e in ['subs', 'dels', 'ins']}
        print(f"  {str(lb):>8s} {mults['subs']:>10.2f}x {mults['dels']:>10.2f}x {mults['ins']:>10.2f}x")

    # Position zone multipliers (relative to middle)
    positions = sorted(s['pos_stats'].keys())
    if positions:
        max_pos = max(positions)
        zone_counts = {'start': {'subs': 0, 'dels': 0, 'ins': 0, 'total': 0},
                       'middle': {'subs': 0, 'dels': 0, 'ins': 0, 'total': 0},
                       'end': {'subs': 0, 'dels': 0, 'ins': 0, 'total': 0}}
        for p in positions:
            zone = get_position_zone(p, max_pos + 1)
            for e in ['subs', 'dels', 'ins', 'total']:
                zone_counts[zone][e] += s['pos_stats'][p][e]

        print(f"\n  Position zone multipliers (vs middle zone):")
        print(f"  {'Zone':>8s} {'Sub mult':>10s} {'Del mult':>10s} {'Ins mult':>10s} {'Total obs':>12s}")
        mid = zone_counts['middle']
        mid_rates = {e: mid[e] / mid['total'] if mid['total'] > 0 else 0 for e in ['subs', 'dels', 'ins']}
        for zone in ['start', 'middle', 'end']:
            z = zone_counts[zone]
            z_rates = {e: z[e] / z['total'] if z['total'] > 0 else 0 for e in ['subs', 'dels', 'ins']}
            mults = {e: z_rates[e] / mid_rates[e] if mid_rates[e] > 0 else 1.0 for e in ['subs', 'dels', 'ins']}
            print(f"  {zone:>8s} {mults['subs']:>10.2f}x {mults['dels']:>10.2f}x {mults['ins']:>10.2f}x {z['total']:>12,}")

    # GC multipliers (relative to mid GC bin)
    gc_bins = {'low': [], 'mid': [], 'high': []}
    for gc_frac, sr, dr, ir in s['gc_error_data']:
        b = 'low' if gc_frac < 0.30 else ('high' if gc_frac > 0.70 else 'mid')
        gc_bins[b].append((sr, dr, ir))

    print(f"\n  GC content multipliers (vs mid GC bin 0.30-0.70):")
    print(f"  {'GC bin':>8s} {'Sub mult':>10s} {'Del mult':>10s} {'Ins mult':>10s} {'N reads':>10s}")
    if gc_bins['mid']:
        mid_gc = {
            'sub': np.mean([x[0] for x in gc_bins['mid']]),
            'del': np.mean([x[1] for x in gc_bins['mid']]),
            'ins': np.mean([x[2] for x in gc_bins['mid']]),
        }
        for b_name, b_range in [('low', '<0.30'), ('mid', '0.30-0.70'), ('high', '>0.70')]:
            if gc_bins[b_name]:
                gc_r = {
                    'sub': np.mean([x[0] for x in gc_bins[b_name]]),
                    'del': np.mean([x[1] for x in gc_bins[b_name]]),
                    'ins': np.mean([x[2] for x in gc_bins[b_name]]),
                }
                mults = {e: gc_r[e] / mid_gc[e] if mid_gc[e] > 0 else 1.0 for e in ['sub', 'del', 'ins']}
                print(f"  {b_range:>8s} {mults['sub']:>10.2f}x {mults['del']:>10.2f}x {mults['ins']:>10.2f}x {len(gc_bins[b_name]):>10,}")

    # 10. Sanity check: predicted (multiplicative) vs observed (joint) rates
    print(f"\n--- 10. Sanity Check: Predicted vs Observed Joint Rates ---")
    print(f"  {'Key (nt, hp, gc, zone)':>40s} {'Obs sub':>10s} {'Pred sub':>10s} {'Obs del':>10s} {'Pred del':>10s} {'Obs ins':>10s} {'Pred ins':>10s} {'N obs':>8s}")

    # Compute all multipliers for prediction
    hp_mults = {}
    for lb in [1, 2, 3, 4, '5+']:
        h = s['homo_stats'][lb]
        ref_r = {e: ref_hp[e] / ref_hp['total'] if ref_hp['total'] > 0 else 0 for e in ['subs', 'dels', 'ins']}
        hp_r = {e: h[e] / h['total'] if h['total'] > 0 else 0 for e in ['subs', 'dels', 'ins']}
        hp_mults[lb] = {e: hp_r[e] / ref_r[e] if ref_r[e] > 0 else 1.0 for e in ['subs', 'dels', 'ins']}

    zone_mults = {}
    if positions:
        for zone in ['start', 'middle', 'end']:
            z = zone_counts[zone]
            z_rates = {e: z[e] / z['total'] if z['total'] > 0 else 0 for e in ['subs', 'dels', 'ins']}
            zone_mults[zone] = {e: z_rates[e] / mid_rates[e] if mid_rates[e] > 0 else 1.0 for e in ['subs', 'dels', 'ins']}

    gc_mults = {}
    if gc_bins['mid']:
        for b_name in ['low', 'mid', 'high']:
            if gc_bins[b_name]:
                gc_r = {
                    'sub': np.mean([x[0] for x in gc_bins[b_name]]),
                    'del': np.mean([x[1] for x in gc_bins[b_name]]),
                    'ins': np.mean([x[2] for x in gc_bins[b_name]]),
                }
                gc_mults[b_name] = {e: gc_r[e] / mid_gc[e] if mid_gc[e] > 0 else 1.0 for e in ['sub', 'del', 'ins']}

    # Compare predicted vs observed for joint keys with enough data
    n_shown = 0
    for jkey, jdata in sorted(s['joint_stats'].items(), key=lambda x: -x[1]['total']):
        if jdata['total'] < 100:
            continue
        nt, hp, gc, zone = jkey
        # Observed
        obs = {e: jdata[e] / jdata['total'] if jdata['total'] > 0 else 0 for e in ['subs', 'dels', 'ins']}
        # Predicted: base_rate * hp_mult * zone_mult * gc_mult
        base = {
            'subs': s['per_nt'][nt]['subs'] / s['per_nt'][nt]['total'] if s['per_nt'][nt]['total'] > 0 else 0,
            'dels': s['per_nt'][nt]['dels'] / s['per_nt'][nt]['total'] if s['per_nt'][nt]['total'] > 0 else 0,
            'ins': s['per_nt'][nt]['ins'] / s['per_nt'][nt]['total'] if s['per_nt'][nt]['total'] > 0 else 0,
        }
        pred = {}
        for e, e_short in [('subs', 'sub'), ('dels', 'del'), ('ins', 'ins')]:
            hp_m = hp_mults.get(hp, {}).get(e, 1.0)
            z_m = zone_mults.get(zone, {}).get(e, 1.0)
            g_m = gc_mults.get(gc, {}).get(e_short, 1.0)
            pred[e] = base[e] * hp_m * z_m * g_m

        key_str = f"({nt}, hp={hp}, gc={gc}, {zone})"
        print(f"  {key_str:>40s} {obs['subs']:>10.4f} {pred['subs']:>10.4f} {obs['dels']:>10.4f} {pred['dels']:>10.4f} {obs['ins']:>10.4f} {pred['ins']:>10.4f} {jdata['total']:>8,}")
        n_shown += 1
        if n_shown >= 20:
            print(f"  ... ({len([k for k, v in s['joint_stats'].items() if v['total'] >= 100])} total joint bins with >=100 observations)")
            break

    # Save JSON
    save_json(s, output_dir)


def _compute_hp_multipliers(s):
    """Homopolymer multipliers relative to non-homopolymer (length 1)."""
    ref = s['homo_stats'][1]
    ref_rates = {e: ref[e] / ref['total'] if ref['total'] > 0 else 0 for e in ['subs', 'dels', 'ins']}
    result = {}
    for lb in [1, 2, 3, 4, '5+']:
        h = s['homo_stats'][lb]
        hp_rates = {e: h[e] / h['total'] if h['total'] > 0 else 0 for e in ['subs', 'dels', 'ins']}
        result[str(lb)] = {
            'sub': hp_rates['subs'] / ref_rates['subs'] if ref_rates['subs'] > 0 else 1.0,
            'del': hp_rates['dels'] / ref_rates['dels'] if ref_rates['dels'] > 0 else 1.0,
            'ins': hp_rates['ins'] / ref_rates['ins'] if ref_rates['ins'] > 0 else 1.0,
        }
    return result


def _compute_zone_multipliers(s):
    """Position zone multipliers relative to middle zone."""
    positions = sorted(s['pos_stats'].keys())
    if not positions:
        return {}
    max_pos = max(positions)
    zone_counts = {'start': {'subs': 0, 'dels': 0, 'ins': 0, 'total': 0},
                   'middle': {'subs': 0, 'dels': 0, 'ins': 0, 'total': 0},
                   'end': {'subs': 0, 'dels': 0, 'ins': 0, 'total': 0}}
    for p in positions:
        zone = get_position_zone(p, max_pos + 1)
        for e in ['subs', 'dels', 'ins', 'total']:
            zone_counts[zone][e] += s['pos_stats'][p][e]
    mid = zone_counts['middle']
    mid_rates = {e: mid[e] / mid['total'] if mid['total'] > 0 else 0 for e in ['subs', 'dels', 'ins']}
    result = {}
    for zone in ['start', 'middle', 'end']:
        z = zone_counts[zone]
        z_rates = {e: z[e] / z['total'] if z['total'] > 0 else 0 for e in ['subs', 'dels', 'ins']}
        result[zone] = {
            'sub': z_rates['subs'] / mid_rates['subs'] if mid_rates['subs'] > 0 else 1.0,
            'del': z_rates['dels'] / mid_rates['dels'] if mid_rates['dels'] > 0 else 1.0,
            'ins': z_rates['ins'] / mid_rates['ins'] if mid_rates['ins'] > 0 else 1.0,
        }
    return result


def _compute_gc_multipliers(s):
    """GC content multipliers relative to mid GC bin (0.30-0.70)."""
    gc_bins = {'low': [], 'mid': [], 'high': []}
    for gc_frac, sr, dr, ir in s['gc_error_data']:
        b = 'low' if gc_frac < 0.30 else ('high' if gc_frac > 0.70 else 'mid')
        gc_bins[b].append((sr, dr, ir))
    if not gc_bins['mid']:
        return {}
    mid_gc = {
        'sub': np.mean([x[0] for x in gc_bins['mid']]),
        'del': np.mean([x[1] for x in gc_bins['mid']]),
        'ins': np.mean([x[2] for x in gc_bins['mid']]),
    }
    result = {}
    for b_name, b_label in [('low', '<0.30'), ('mid', '0.30-0.70'), ('high', '>0.70')]:
        if gc_bins[b_name]:
            gc_r = {
                'sub': np.mean([x[0] for x in gc_bins[b_name]]),
                'del': np.mean([x[1] for x in gc_bins[b_name]]),
                'ins': np.mean([x[2] for x in gc_bins[b_name]]),
            }
            result[b_label] = {
                'sub': gc_r['sub'] / mid_gc['sub'] if mid_gc['sub'] > 0 else 1.0,
                'del': gc_r['del'] / mid_gc['del'] if mid_gc['del'] > 0 else 1.0,
                'ins': gc_r['ins'] / mid_gc['ins'] if mid_gc['ins'] > 0 else 1.0,
            }
    return result


def _compute_burst_weights(s):
    """Convert burst length counts into weights (sum to 1) for sampling with weighted_choice."""
    result = {}
    for err_type, key in [('deletion', 'del_bursts'), ('insertion', 'ins_bursts'), ('substitution', 'sub_bursts')]:
        raw = Counter(s[key])
        if not raw:
            result[err_type] = {'lengths': [1], 'weights': [1.0]}
            continue
        lengths_counts = sorted(raw.items())
        total = sum(c for _, c in lengths_counts)
        lengths = [l for l, _ in lengths_counts]
        weights = [c / total for _, c in lengths_counts]
        result[err_type] = {'lengths': lengths, 'weights': weights}
    return result


def _compute_sub_weights(s):
    """Convert substitution matrix counts into per-base sampling weights (sum to 1)."""
    result = {}
    for base in BASES:
        targets = []
        counts = []
        for other in BASES:
            if other == base:
                continue
            count = s['sub_matrix'][base][other]
            targets.append(other)
            counts.append(count)
        total = sum(counts)
        weights = [c / total if total > 0 else 1.0 / len(counts) for c in counts]
        result[base] = {'targets': targets, 'weights': weights}
    return result


def _rate_with_ci(counts):
    """Compute rate + 95% CI for a counts dict with {subs, dels, ins, total}."""
    t = counts['total']
    result = {'total': t}
    for err_key, rate_key in [('subs', 'sub_rate'), ('dels', 'del_rate'), ('ins', 'ins_rate')]:
        e = counts[err_key]
        rate = e / t if t > 0 else 0
        ci_lo, ci_hi = binomial_ci(e, t)
        result[rate_key] = rate
        result[rate_key + '_ci'] = [ci_lo, ci_hi]
    return result


def _rate_with_ci_and_reliability(counts, max_rel_ci_width=0.30):
    """Like _rate_with_ci but also flags whether each rate is reliable.

    A rate is 'reliable' if CI width < max_rel_ci_width * rate.
    When reliable=True, use the joint rate directly.
    When reliable=False, fall back to multiplicative model.
    """
    t = counts['total']
    result = {'total': t}
    for err_key, rate_key in [('subs', 'sub_rate'), ('dels', 'del_rate'), ('ins', 'ins_rate')]:
        e = counts[err_key]
        rate = e / t if t > 0 else 0
        ci_lo, ci_hi = binomial_ci(e, t)
        ci_width = ci_hi - ci_lo
        reliable = bool(ci_width < max_rel_ci_width * rate) if rate > 0 else False
        result[rate_key] = rate
        result[rate_key + '_ci'] = [ci_lo, ci_hi]
        result[rate_key + '_reliable'] = reliable
    return result


def save_json(s, output_dir):
    """Save all statistics as JSON."""
    total = s['total_bases']

    rl = s['read_lengths']
    result = {
        'read_length_min': int(min(rl)) if rl else 0,
        'read_length_max': int(max(rl)) if rl else 0,
        'overall': {
            'total_bases': total,
            'sub_rate': s['total_subs'] / total if total > 0 else 0,
            'del_rate': s['total_dels'] / total if total > 0 else 0,
            'ins_rate': s['total_ins'] / total if total > 0 else 0,
        },
        'sub_matrix': {
            f'{b1}->{b2}': s['sub_matrix'][b1][b2]
            for b1 in BASES for b2 in BASES if b1 != b2
        },
        'per_nucleotide': {
            b: _rate_with_ci(s['per_nt'][b]) for b in BASES
        },
        'homopolymer': {
            str(lb): _rate_with_ci(s['homo_stats'][lb]) for lb in [1, 2, 3, 4, '5+']
        },
        'burst_lengths': {
            'deletion': dict(Counter(s['del_bursts'])),
            'insertion': dict(Counter(s['ins_bursts'])),
            'substitution': dict(Counter(s['sub_bursts'])),
        },
        'burst_length_weights': _compute_burst_weights(s),
        'sub_weights': _compute_sub_weights(s),
        'gt_nucleotide_global': {
            b: sum(s['gt_pos_nt'][pos][b] for pos in s['gt_pos_nt'])
               / sum(sum(s['gt_pos_nt'][pos].values()) for pos in s['gt_pos_nt'])
            for b in BASES
        },
        'gt_nucleotide_by_position': {
            str(pos): {
                b: s['gt_pos_nt'][pos][b] / sum(s['gt_pos_nt'][pos].values())
                for b in BASES
            }
            for pos in sorted(s['gt_pos_nt'].keys())
        },
        'multipliers': {
            'homopolymer': _compute_hp_multipliers(s),
            'position_zone': _compute_zone_multipliers(s),
            'gc_content': _compute_gc_multipliers(s),
        },
        'joint_rates': {
            f"{nt}|hp={hp}|gc={gc}|{zone}": _rate_with_ci_and_reliability(jdata)
            for (nt, hp, gc, zone), jdata in s['joint_stats'].items()
            if jdata['total'] >= 100
        },
    }

    path = os.path.join(output_dir, 'error_model.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved JSON: {path}")


# ============================================================
# Plotting
# ============================================================

def make_plots(s, output_dir):
    """Generate all plots."""

    # 1. Substitution matrix heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    mat = np.zeros((4, 4))
    for i, b1 in enumerate(BASES):
        row_total = sum(s['sub_matrix'][b1][b2] for b2 in BASES if b2 != b1)
        for j, b2 in enumerate(BASES):
            if b1 == b2:
                mat[i, j] = 0
            else:
                mat[i, j] = s['sub_matrix'][b1][b2] / row_total if row_total > 0 else 0
    im = ax.imshow(mat, cmap='YlOrRd', vmin=0)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(BASES)
    ax.set_yticklabels(BASES)
    ax.set_xlabel('To')
    ax.set_ylabel('From')
    ax.set_title('Substitution Matrix')
    for i in range(4):
        for j in range(4):
            if i != j:
                ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center', fontsize=10)
            else:
                ax.text(j, i, '—', ha='center', va='center', fontsize=10, color='gray')
    plt.colorbar(im, ax=ax, label='Fraction')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'substitution_matrix.pdf'), dpi=150)
    plt.close()

    # 2. Per-nucleotide error rates (with 95% CIs)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(4)
    width = 0.25
    for offset, err_type, label, color in [
        (-width, 'subs', 'Substitution', '#e74c3c'),
        (0, 'dels', 'Deletion', '#3498db'),
        (width, 'ins', 'Insertion', '#2ecc71'),
    ]:
        rates = []
        ci_low = []
        ci_high = []
        for b in BASES:
            t = s['per_nt'][b]['total']
            e = s['per_nt'][b][err_type]
            r = e / t if t > 0 else 0
            lo, hi = binomial_ci(e, t)
            rates.append(r)
            ci_low.append(r - lo)
            ci_high.append(hi - r)
        ax.bar(x + offset, rates, width, label=label, color=color,
               yerr=[ci_low, ci_high], capsize=3, error_kw={'linewidth': 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(BASES)
    ax.set_ylabel('Error rate')
    ax.set_title('Error Rates by Nucleotide (with 95% CI)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_nucleotide_rates.pdf'), dpi=150)
    plt.close()

    # 3. Position-dependent error rates (with 95% CI bands)
    positions = sorted(s['pos_stats'].keys())
    if positions:
        max_pos = max(positions)
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for idx, (err_type, label, color) in enumerate([
            ('subs', 'Substitution', '#e74c3c'),
            ('dels', 'Deletion', '#3498db'),
            ('ins', 'Insertion', '#2ecc71'),
        ]):
            rates = []
            ci_los = []
            ci_his = []
            for p in range(max_pos + 1):
                if p in s['pos_stats'] and s['pos_stats'][p]['total'] > 0:
                    t = s['pos_stats'][p]['total']
                    e = s['pos_stats'][p][err_type]
                    rates.append(e / t)
                    lo, hi = binomial_ci(e, t)
                    ci_los.append(lo)
                    ci_his.append(hi)
                else:
                    rates.append(0)
                    ci_los.append(0)
                    ci_his.append(0)
            pos_range = range(max_pos + 1)
            axes[idx].plot(pos_range, rates, '.', markersize=3, color=color)
            axes[idx].fill_between(pos_range, ci_los, ci_his, alpha=0.2, color=color)
            axes[idx].set_ylabel(f'{label} rate')
            axes[idx].set_title(f'{label} Rate per Position (with 95% CI)')
            axes[idx].grid(True, alpha=0.3)
        axes[-1].set_xlabel('Position in ground truth')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'position_error_rates.pdf'), dpi=150)
        plt.close()

    # 4. Homopolymer error rates (with 95% CIs)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ['1\n(non-hp)', '2', '3', '4', '5+']
    keys = [1, 2, 3, 4, '5+']
    x = np.arange(len(keys))
    width = 0.25
    for offset, err_type, label, color in [
        (-width, 'subs', 'Substitution', '#e74c3c'),
        (0, 'dels', 'Deletion', '#3498db'),
        (width, 'ins', 'Insertion', '#2ecc71'),
    ]:
        rates = []
        ci_low = []
        ci_high = []
        for k in keys:
            t = s['homo_stats'][k]['total']
            e = s['homo_stats'][k][err_type]
            r = e / t if t > 0 else 0
            lo, hi = binomial_ci(e, t)
            rates.append(r)
            ci_low.append(r - lo)
            ci_high.append(hi - r)
        ax.bar(x + offset, rates, width, label=label, color=color,
               yerr=[ci_low, ci_high], capsize=3, error_kw={'linewidth': 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Homopolymer run length')
    ax.set_ylabel('Error rate')
    ax.set_title('Error Rates by Homopolymer Length (with 95% CI)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'homopolymer_rates.pdf'), dpi=150)
    plt.close()

    # 5. Error burst length distributions
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for idx, (name, bursts, color) in enumerate([
        ('Deletion', s['del_bursts'], '#3498db'),
        ('Insertion', s['ins_bursts'], '#2ecc71'),
        ('Substitution', s['sub_bursts'], '#e74c3c'),
    ]):
        if not bursts:
            axes[idx].set_title(f'{name} Burst Length')
            continue
        c = Counter(bursts)
        total_b = sum(c.values())
        lengths = sorted(c.keys())[:10]
        rates = [c[l] / total_b for l in lengths]
        axes[idx].bar(lengths, rates, color=color)
        axes[idx].set_xlabel('Burst length')
        axes[idx].set_ylabel('Fraction of bursts')
        axes[idx].set_title(f'{name} Burst Length')
        axes[idx].set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'burst_lengths.pdf'), dpi=150)
    plt.close()

    # 6. GC content vs error rate (3 stacked subplots: sub, del, ins)
    if s['gc_error_data']:
        gc_fracs = [x[0] for x in s['gc_error_data']]
        sub_rates = [x[1] for x in s['gc_error_data']]
        del_rates = [x[2] for x in s['gc_error_data']]
        ins_rates = [x[3] for x in s['gc_error_data']]

        bins = np.linspace(min(gc_fracs) - 0.01, max(gc_fracs) + 0.01, 11)

        fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
        for idx, (rates, label, color) in enumerate([
            (sub_rates, 'Substitution', '#e74c3c'),
            (del_rates, 'Deletion', '#3498db'),
            (ins_rates, 'Insertion', '#2ecc71'),
        ]):
            bin_centers = []
            bin_means = []
            bin_sems = []
            for i in range(len(bins) - 1):
                mask = [(gc_fracs[j] >= bins[i] and gc_fracs[j] < bins[i + 1]) for j in range(len(gc_fracs))]
                vals = [rates[j] for j in range(len(rates)) if mask[j]]
                if vals:
                    bin_centers.append((bins[i] + bins[i + 1]) / 2)
                    bin_means.append(np.mean(vals))
                    bin_sems.append(np.std(vals) / np.sqrt(len(vals)))
            axes[idx].errorbar(bin_centers, bin_means, yerr=bin_sems, fmt='o-', capsize=3, color=color)
            axes[idx].set_ylabel(f'{label} rate')
            axes[idx].set_title(f'{label} Rate vs GC Content')
            axes[idx].grid(True, alpha=0.3)
        axes[-1].set_xlabel('GC content')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gc_vs_error.pdf'), dpi=150)
        plt.close()

    # 7. Read length distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(s['read_lengths'], bins=range(min(s['read_lengths']), max(s['read_lengths']) + 2),
            color='#3498db', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Read length')
    ax.set_ylabel('Count')
    ax.set_title('Read Length Distribution')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'read_length_dist.pdf'), dpi=150)
    plt.close()

    # 8. Edit distance CDF
    fig, ax = plt.subplots(figsize=(6, 4))
    ed = np.array(s['edit_distances'])
    max_ed = min(int(np.percentile(ed, 99)), max(ed))
    x_vals = range(0, max_ed + 1)
    cdf = [np.mean(ed <= x) for x in x_vals]
    ax.plot(x_vals, cdf, 'o-', markersize=3, color='#2c3e50')
    ax.set_xlabel('Number of edit errors')
    ax.set_ylabel('Fraction of reads')
    ax.set_title('Cumulative Distribution of Edit Errors')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'edit_distance_cdf.pdf'), dpi=150)
    plt.close()

    # 9. GT nucleotide frequency by position
    if s['gt_pos_nt']:
        positions = sorted(s['gt_pos_nt'].keys())
        max_pos = max(positions)
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = {'A': '#e74c3c', 'C': '#3498db', 'G': '#2ecc71', 'T': '#f39c12'}
        for base in BASES:
            freqs = []
            for p in range(max_pos + 1):
                if p in s['gt_pos_nt']:
                    total_at_pos = sum(s['gt_pos_nt'][p].values())
                    freqs.append(s['gt_pos_nt'][p][base] / total_at_pos if total_at_pos > 0 else 0.25)
                else:
                    freqs.append(0.25)
            ax.plot(range(max_pos + 1), freqs, '-', linewidth=1, color=colors[base], label=base, alpha=0.8)
        ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Uniform (0.25)')
        ax.set_xlabel('Position in ground truth')
        ax.set_ylabel('Nucleotide frequency')
        ax.set_title('GT Nucleotide Frequency by Position')
        ax.legend()
        ax.set_ylim(0, 0.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gt_nucleotide_by_position.pdf'), dpi=150)
        plt.close()

    print(f"Saved plots to {output_dir}/")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Estimate error model from DNA storage data')
    parser.add_argument('input_file', help='Path to train.txt file')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: data/error_model/results)')
    parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of data to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = 'data/error_model/results'

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    examples = parse_train_file(args.input_file)
    print(f"Loaded {len(examples)} examples from {args.input_file}")

    if args.fraction < 1.0:
        n = max(1, int(len(examples) * args.fraction))
        examples = random.sample(examples, n)
        print(f"Sampled {len(examples)} examples ({args.fraction*100:.0f}%)")

    total_reads = sum(len(reads) for _, reads in examples)
    print(f"Total reads to process: {total_reads:,}")

    # Collect statistics
    print("Aligning reads and collecting statistics...")
    s = collect_statistics(examples)

    # Report
    print_report(s, args.output_dir)
    make_plots(s, args.output_dir)


if __name__ == '__main__':
    main()
