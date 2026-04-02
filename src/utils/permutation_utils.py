"""
Utilities for permutation-based ensemble inference with majority voting.

This module provides functions to:
- Generate unique permutations of read orderings
- Reorder reads in input sequences
- Perform positional majority voting across multiple predictions
- Compute diversity metrics between predictions
"""

import itertools
import math
import numpy as np
from typing import List, Tuple
from collections import Counter


def generate_unique_permutations(n_reads: int, max_perms: int, seed: int = 42) -> List[Tuple[int, ...]]:
    """
    Generate unique permutations of read indices.

    The number of permutations returned is min(max_perms, n_reads!).
    If n_reads! <= max_perms, all permutations are returned.
    Otherwise, a random sample of unique permutations is returned.

    Args:
        n_reads: Number of reads (cluster size)
        max_perms: Maximum permutations to generate
        seed: Random seed for sampling

    Returns:
        List of permutation tuples, e.g., [(0,1,2), (1,0,2), (2,1,0)]

    Examples:
        >>> generate_unique_permutations(2, 10, 42)
        [(0, 1), (1, 0)]  # All 2! = 2 permutations

        >>> generate_unique_permutations(5, 10, 42)
        # Returns 10 randomly sampled permutations out of 5! = 120
    """
    factorial_n = math.factorial(n_reads)
    num_perms = min(max_perms, factorial_n)

    if factorial_n <= max_perms:
        # Use all permutations if N! is small enough
        perms = list(itertools.permutations(range(n_reads)))
        return perms[:num_perms]
    else:
        # Sample unique permutations without replacement
        rng = np.random.RandomState(seed)
        all_perms = list(itertools.permutations(range(n_reads)))
        sampled_indices = rng.choice(len(all_perms), size=num_perms, replace=False)
        return [all_perms[i] for i in sampled_indices]


def permute_reads(input_string: str, permutation: Tuple[int, ...]) -> str:
    """
    Reorder reads in input string according to permutation.

    The input format is "read1|read2|read3:ground_truth".
    Only the reads are reordered; the ground truth remains unchanged.

    Args:
        input_string: Input in format "read1|read2|read3:ground_truth"
        permutation: Tuple of indices, e.g., (1, 0, 2) swaps first two reads

    Returns:
        Permuted string with same format: "read2|read1|read3:ground_truth"

    Examples:
        >>> permute_reads("ACGT|TGCA|GGCC:ACGTACGT", (2, 0, 1))
        "GGCC|ACGT|TGCA:ACGTACGT"

        >>> permute_reads("AAA|BBB:CCC", (1, 0))
        "BBB|AAA:CCC"
    """
    if ':' not in input_string:
        raise ValueError(f"Input string must contain ':' separator. Got: {input_string}")

    reads_part, gt_part = input_string.split(':', 1)
    reads = reads_part.split('|')

    if len(reads) != len(permutation):
        raise ValueError(
            f"Permutation length ({len(permutation)}) must match number of reads ({len(reads)})"
        )

    # Apply permutation
    permuted_reads = [reads[i] for i in permutation]

    return '|'.join(permuted_reads) + ':' + gt_part


def positional_majority_vote(predictions: List[str], tie_breaking_strategy: str = 'random') -> Tuple[str, dict]:
    """
    Perform positional majority voting across predictions.

    At each position, the most common nucleotide across all predictions wins.
    In case of a tie (e.g., with 2 predictions where both nucleotides have equal count),
    the tie is resolved according to the tie_breaking_strategy parameter.
    If predictions have different lengths, shorter ones are treated as having '-'
    at positions beyond their length.

    Args:
        predictions: List of predicted sequences, e.g., ["ACGT", "ACCT", "ACGT"]
        tie_breaking_strategy: Strategy for breaking ties. Options:
            - 'random': Choose randomly among tied candidates (default)
            - 'first_prediction': Use the nucleotide from the first prediction

    Returns:
        Tuple of (voted_sequence, statistics_dict)
        - voted_sequence: String with majority vote at each position
        - statistics_dict: Contains 'mean_agreement' and 'min_agreement'

    Examples:
        >>> positional_majority_vote(["ACGT", "ACCT", "ACGT"])
        ("ACGT", {'mean_agreement': 0.917, 'min_agreement': 0.667})
        # Position 0: A (3/3) = 1.0
        # Position 1: C (3/3) = 1.0
        # Position 2: G (2/3) vs C (1/3) → G wins with 0.667
        # Position 3: T (3/3) = 1.0
        # Mean agreement: (1.0 + 1.0 + 0.667 + 1.0) / 4 = 0.917
    """
    if not predictions:
        return "", {'mean_agreement': 0.0, 'min_agreement': 0.0}

    if len(predictions) == 1:
        return predictions[0], {'mean_agreement': 1.0, 'min_agreement': 1.0}

    max_len = max(len(p) for p in predictions)
    voted_seq = []
    agreements = []

    for pos in range(max_len):
        # Get nucleotide at this position from each prediction
        nucleotides = [p[pos] if pos < len(p) else '-' for p in predictions]
        counts = Counter(nucleotides)

        # Most common nucleotide wins
        # In case of tie, use tie_breaking_strategy to resolve
        max_count = max(counts.values())
        candidates = [nuc for nuc, count in counts.items() if count == max_count]

        if len(candidates) > 1:
            # Tie detected - resolve based on strategy
            if tie_breaking_strategy == 'first_prediction':
                # Use nucleotide from first prediction if it's among tied candidates
                base_nuc = predictions[0][pos] if pos < len(predictions[0]) else '-'
                winner = base_nuc if base_nuc in candidates else candidates[0]
            else:
                # Default: random choice among tied candidates
                winner = np.random.choice(candidates)
        else:
            # No tie - use the winner
            winner = candidates[0]

        voted_seq.append(winner)

        # Agreement = fraction of predictions agreeing with winner
        agreement = max_count / len(predictions)
        agreements.append(agreement)

    # Remove trailing '-' characters if present
    voted_string = ''.join(voted_seq).rstrip('-')

    return voted_string, {
        'mean_agreement': float(np.mean(agreements)),
        'min_agreement': float(np.min(agreements)),
    }


def compute_pairwise_hamming_distances(predictions: List[str]) -> dict:
    """
    Compute pairwise Hamming distances between all predictions.

    This measures diversity: how much do predictions differ across permutations?
    Uses Hamming distance (substitutions only) since we're comparing predictions
    to each other, not to ground truth.

    Args:
        predictions: List of predicted sequences

    Returns:
        Dictionary with diversity statistics:
        - mean_pairwise_hamming: Average Hamming distance across all pairs
        - std_pairwise_hamming: Standard deviation
        - max_pairwise_hamming: Maximum distance
        - min_pairwise_hamming: Minimum distance

    Examples:
        >>> compute_pairwise_hamming_distances(["ACGT", "ACGT", "ACGT"])
        {'mean_pairwise_hamming': 0.0, 'std_pairwise_hamming': 0.0, ...}

        >>> compute_pairwise_hamming_distances(["ACGT", "TCGT", "ACCT"])
        # 3 pairs: (ACGT, TCGT)=1, (ACGT, ACCT)=1, (TCGT, ACCT)=2
        # Mean: 1.33, std: 0.47, max: 2, min: 1
    """
    from itertools import combinations

    if len(predictions) <= 1:
        return {
            'mean_pairwise_hamming': 0.0,
            'std_pairwise_hamming': 0.0,
            'max_pairwise_hamming': 0,
            'min_pairwise_hamming': 0,
        }

    distances = []
    for pred1, pred2 in combinations(predictions, 2):
        # Simple Hamming distance: count mismatches
        min_len = min(len(pred1), len(pred2))
        max_len = max(len(pred1), len(pred2))

        # Count mismatches in overlapping region
        mismatches = sum(c1 != c2 for c1, c2 in zip(pred1[:min_len], pred2[:min_len]))

        # Add length difference as additional mismatches
        length_diff = max_len - min_len
        total_distance = mismatches + length_diff

        distances.append(total_distance)

    return {
        'mean_pairwise_hamming': float(np.mean(distances)),
        'std_pairwise_hamming': float(np.std(distances)),
        'max_pairwise_hamming': int(np.max(distances)),
        'min_pairwise_hamming': int(np.min(distances)),
    }
