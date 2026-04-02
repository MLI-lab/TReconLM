"""Vote confidence analysis utilities for analyzing model predictions."""

import numpy as np
from collections import defaultdict


def collect_position_metrics(prediction, ground_truth, traces, itos):
    """
    Collect position-wise voting metrics for analyzing model confidence.

    Parameters:
        prediction (str): Model's predicted sequence
        ground_truth (str): True ground truth sequence
        traces (list): List of input traces
        itos (dict): Index to string mapping for vocab

    Returns:
        dict: Position-wise metrics including vote margins, agreement, etc.
    """
    metrics = {
        'position_votes': [],
        'position_margins': [],
        'position_correct': [],
        'position_agreement': []
    }

    # Analyze each position
    for pos in range(min(len(prediction), len(ground_truth))):
        # Count votes at this position across traces
        votes = defaultdict(int)
        for trace in traces:
            if pos < len(trace):
                votes[trace[pos]] += 1

        # Calculate metrics
        if votes:
            sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            winner_base = sorted_votes[0][0] if sorted_votes else ''
            winner_count = sorted_votes[0][1] if sorted_votes else 0
            runner_up_count = sorted_votes[1][1] if len(sorted_votes) > 1 else 0

            # Vote margin (difference between top 2)
            margin = winner_count - runner_up_count

            # Agreement rate (fraction voting for winner)
            agreement = winner_count / len(traces) if traces else 0

            # Correctness
            is_correct = prediction[pos] == ground_truth[pos] if pos < len(ground_truth) else False

            metrics['position_votes'].append(dict(votes))
            metrics['position_margins'].append(margin)
            metrics['position_correct'].append(is_correct)
            metrics['position_agreement'].append(agreement)

    return metrics


def analyze_and_log_vote_confidence(all_results, use_wandb=True):
    """
    Analyze vote confidence patterns across all results and log to wandb.

    Parameters:
        all_results (list): List of result dictionaries from inference
        use_wandb (bool): Whether to log to Weights & Biases

    Returns:
        dict: Summary statistics of vote confidence analysis
    """
    if not all_results:
        return {}

    # Aggregate metrics across all examples
    all_margins = []
    all_agreements = []
    correct_margins = []
    incorrect_margins = []

    for result in all_results:
        if 'position_metrics' in result:
            metrics = result['position_metrics']
            all_margins.extend(metrics.get('position_margins', []))
            all_agreements.extend(metrics.get('position_agreement', []))

            # Separate by correctness
            for margin, correct in zip(metrics.get('position_margins', []),
                                       metrics.get('position_correct', [])):
                if correct:
                    correct_margins.append(margin)
                else:
                    incorrect_margins.append(margin)

    # Calculate summary statistics
    summary = {
        'mean_vote_margin': np.mean(all_margins) if all_margins else 0.0,
        'std_vote_margin': np.std(all_margins) if all_margins else 0.0,
        'mean_agreement': np.mean(all_agreements) if all_agreements else 0.0,
        'mean_correct_margin': np.mean(correct_margins) if correct_margins else 0.0,
        'mean_incorrect_margin': np.mean(incorrect_margins) if incorrect_margins else 0.0,
        'num_positions': len(all_margins)
    }

    # Log to wandb if enabled
    if use_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'vote_confidence/mean_margin': summary['mean_vote_margin'],
                    'vote_confidence/mean_agreement': summary['mean_agreement'],
                    'vote_confidence/correct_margin': summary['mean_correct_margin'],
                    'vote_confidence/incorrect_margin': summary['mean_incorrect_margin']
                })
        except:
            pass  # Silently skip if wandb not available

    return summary
