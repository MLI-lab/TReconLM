"""
Logit margin profiling utilities for analyzing model confidence.

This module provides functionality to compute the gap between top-1 and top-2 logits
during generation, which helps explain why beam search provides minimal improvements
when the model's predictive distributions are very peaked.
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import random


def compute_confidence_metrics(logits: torch.Tensor) -> tuple:
    """
    Compute top-1 and top-2 probabilities from logits.

    Args:
        logits: Tensor of shape [batch_size, seq_len, vocab_size] or [seq_len, vocab_size]

    Returns:
        top1_probs: Probability of top-1 token at each position [B, seq_len]
        top2_probs: Probability of top-2 token at each position [B, seq_len]
    """
    # Handle both batched and unbatched logits
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)  # [1, seq_len, vocab_size]

    # Convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=-1)  # [B, seq_len, vocab_size]

    # Get top-2 probabilities for each position
    top2_probs, top2_indices = torch.topk(probs, k=2, dim=-1)  # [B, seq_len, 2]

    # Extract top-1 and top-2 probabilities
    top1_probs = top2_probs[..., 0]  # [B, seq_len]
    top2_probs_vals = top2_probs[..., 1]  # [B, seq_len]

    return top1_probs, top2_probs_vals


def profile_logit_margins_single_batch(
    test_examples: torch.Tensor,
    attn_mask: torch.Tensor,
    model: torch.nn.Module,
    ctx: Any,
    max_new_tokens: int,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    temperature: float = 1.0,
    constrained_generation: bool = False,
    cluster_sizes: List[int] = None
) -> Dict[str, Any]:
    """
    Profile logit margins for a batch of examples during generation.

    Args:
        test_examples: Input tensor [B, T]
        attn_mask: Attention mask [B, T]
        model: The model to profile
        ctx: Context manager for inference (e.g., autocast)
        max_new_tokens: Number of tokens to generate
        stoi: String to index mapping
        itos: Index to string mapping
        temperature: Sampling temperature
        constrained_generation: Whether to use constrained generation
        cluster_sizes: List of cluster sizes for each example in batch

    Returns:
        Dictionary containing margin statistics per example
    """
    with torch.no_grad(), ctx:
        # Use the model's generation with logit collection
        Y, logits = model.generate_cpred(
            idx=test_examples,
            attn_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None,
            sampling='greedy',
            constrained_generation=constrained_generation,
            itos=itos,
            collect_logits=True
        )

    # Compute confidence metrics for each example
    batch_results = []

    for i in range(logits.shape[0]):
        example_logits = logits[i]  # [seq_len, vocab_size]
        top1_probs, top2_probs = compute_confidence_metrics(example_logits)  # [1, seq_len]
        top1_probs = top1_probs.squeeze(0)
        top2_probs = top2_probs.squeeze(0)

        # Convert to numpy for easier statistics (convert to float32 first since numpy doesn't support bfloat16)
        top1_probs_np = top1_probs.cpu().float().numpy()
        top2_probs_np = top2_probs.cpu().float().numpy()

        cluster_size = cluster_sizes[i] if cluster_sizes else None

        batch_results.append({
            'cluster_size': cluster_size,
            'mean_top1_prob': float(np.mean(top1_probs_np)),
            'mean_top2_prob': float(np.mean(top2_probs_np)),
            'num_tokens': len(top1_probs_np)
        })

    return batch_results


def aggregate_margin_statistics(
    all_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate confidence statistics across all examples and by cluster size.

    Args:
        all_results: List of result dictionaries from profile_logit_margins_single_batch

    Returns:
        Dictionary with aggregated statistics
    """
    # Group by cluster size
    by_cluster = defaultdict(lambda: {'top1_probs': [], 'top2_probs': []})
    all_top1_means = []
    all_top2_means = []

    for result in all_results:
        cluster_size = result['cluster_size']
        mean_top1 = result['mean_top1_prob']
        mean_top2 = result['mean_top2_prob']

        all_top1_means.append(mean_top1)
        all_top2_means.append(mean_top2)

        if cluster_size is not None:
            by_cluster[cluster_size]['top1_probs'].append(mean_top1)
            by_cluster[cluster_size]['top2_probs'].append(mean_top2)

    # Compute overall statistics
    overall_stats = {
        'mean_top1_prob': float(np.mean(all_top1_means)),
        'mean_top2_prob': float(np.mean(all_top2_means)),
        'num_examples': len(all_top1_means)
    }

    # Compute per-cluster-size statistics
    cluster_stats = {}
    for cluster_size, data in sorted(by_cluster.items()):
        top1_vals = data['top1_probs']
        top2_vals = data['top2_probs']
        cluster_stats[cluster_size] = {
            'count': len(top1_vals),
            'mean_top1_prob': float(np.mean(top1_vals)),
            'mean_top2_prob': float(np.mean(top2_vals))
        }

    # Find cluster size with largest and smallest top-1 probability
    if cluster_stats:
        largest_cluster = max(cluster_stats.items(), key=lambda x: x[1]['mean_top1_prob'])
        smallest_cluster = min(cluster_stats.items(), key=lambda x: x[1]['mean_top1_prob'])
    else:
        largest_cluster = (None, None)
        smallest_cluster = (None, None)

    return {
        'overall': overall_stats,
        'by_cluster_size': cluster_stats,
        'highest_confidence_cluster': {
            'cluster_size': largest_cluster[0],
            'mean_top1_prob': largest_cluster[1]['mean_top1_prob'] if largest_cluster[1] else None
        },
        'lowest_confidence_cluster': {
            'cluster_size': smallest_cluster[0],
            'mean_top1_prob': smallest_cluster[1]['mean_top1_prob'] if smallest_cluster[1] else None
        }
    }


def print_margin_statistics(stats: Dict[str, Any]):
    """
    Pretty-print confidence statistics.

    Args:
        stats: Dictionary from aggregate_margin_statistics
    """
    print("\n" + "="*80)
    print("MODEL CONFIDENCE PROFILING RESULTS")
    print("="*80)

    overall = stats['overall']
    print(f"\nOverall Statistics (across all examples):")
    print(f"  Number of examples: {overall['num_examples']}")
    print(f"  Mean top-1 probability: {overall['mean_top1_prob']:.1%}")
    print(f"  Mean top-2 probability: {overall['mean_top2_prob']:.1%}")

    print(f"\nPer-Cluster-Size Statistics:")
    print(f"{'N':<6} {'Count':<8} {'Mean Top-1':<16} {'Mean Top-2':<16}")
    print("-" * 50)

    for cluster_size, cluster_data in sorted(stats['by_cluster_size'].items()):
        print(f"{cluster_size:<6} {cluster_data['count']:<8} "
              f"{cluster_data['mean_top1_prob']:<16.1%} {cluster_data['mean_top2_prob']:<16.1%}")

    highest = stats['highest_confidence_cluster']
    lowest = stats['lowest_confidence_cluster']

    if highest['cluster_size'] is not None:
        print(f"\nHighest confidence: N={highest['cluster_size']} (mean top-1 = {highest['mean_top1_prob']:.1%})")

    if lowest['cluster_size'] is not None:
        print(f"Lowest confidence: N={lowest['cluster_size']} (mean top-1 = {lowest['mean_top1_prob']:.1%})")

    print("\n" + "="*80)
    print()


def subsample_data_uniformly(
    all_data: List[Tuple],
    num_samples: int,
    seed: int = 42
) -> List[Tuple]:
    """
    Subsample data uniformly across cluster sizes.

    Args:
        all_data: List of (x_tensor, ground_truth, cluster_size) tuples
        num_samples: Number of samples to select
        seed: Random seed for reproducibility

    Returns:
        Subsampled data list
    """
    random.seed(seed)

    # Group by cluster size
    by_cluster = defaultdict(list)
    for item in all_data:
        cluster_size = item[2]
        by_cluster[cluster_size].append(item)

    # Calculate samples per cluster size (uniform sampling)
    cluster_sizes = sorted(by_cluster.keys())
    samples_per_cluster = num_samples // len(cluster_sizes)
    remainder = num_samples % len(cluster_sizes)

    subsampled = []
    for i, cluster_size in enumerate(cluster_sizes):
        cluster_data = by_cluster[cluster_size]
        # Give remainder samples to first few clusters
        n_samples = samples_per_cluster + (1 if i < remainder else 0)
        n_samples = min(n_samples, len(cluster_data))

        sampled = random.sample(cluster_data, n_samples)
        subsampled.extend(sampled)

    print(f"Subsampled {len(subsampled)} examples from {len(all_data)} total (uniformly across {len(cluster_sizes)} cluster sizes)")

    return subsampled
