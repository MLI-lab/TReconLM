import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from Levenshtein import distance as levenshtein_distance
from scipy import stats

# LaTeX style settings (matching plots.ipynb style)
FONTSIZE = 8
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{type1cm}',
    'font.size': FONTSIZE,
})


def calculate_inter_trace_disagreement(traces):
    """
    Calculate inter-trace disagreement metrics.

    Args:
        traces: List of noisy trace sequences

    Returns:
        dict with disagreement metrics
    """
    if len(traces) <= 1:
        return {'pairwise_distance': 0.0, 'std_length': 0.0}

    # Calculate pairwise Levenshtein distances
    pairwise_distances = []
    for i in range(len(traces)):
        for j in range(i + 1, len(traces)):
            dist = levenshtein_distance(traces[i], traces[j])
            pairwise_distances.append(dist)

    # Calculate length variability
    lengths = [len(t) for t in traces]

    return {
        'pairwise_distance': np.mean(pairwise_distances),
        'pairwise_distance_std': np.std(pairwise_distances),
        'std_length': np.std(lengths),
        'max_pairwise_distance': max(pairwise_distances) if pairwise_distances else 0
    }


def calculate_avg_trace_to_gt_distance(traces, ground_truth):
    """
    Calculate average distance from noisy traces to ground truth.

    Args:
        traces: List of noisy trace sequences
        ground_truth: Ground truth sequence

    Returns:
        float: Average Levenshtein distance
    """
    if not traces:
        return 0.0

    distances = [levenshtein_distance(trace, ground_truth) for trace in traces]
    return np.mean(distances)


def analyze_disagreement_impact(filepath, bin_width=5):
    """
    Analyze the impact of inter-trace disagreement on reconstruction performance,
    controlling for average distance to ground truth.

    Args:
        filepath: Path to predictions_output.tsv file
        bin_width: Width of bins for average distance to GT (in edits)

    Returns:
        dict: Results organized by cluster size and distance bins
    """
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"  ⚠ File not found: {filepath}")
        return None

    print("="*70)
    print("Inter-trace disagreement analysis")
    print("="*70)

    # Read data
    clusters_by_size = defaultdict(list)

    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cluster_size = int(row["cluster_size"])
            ground_truth = row["ground_truth"]
            prediction = row["prediction"]

            # Parse noisy traces (pipe-separated)
            noisy_traces_str = row.get("noisy_traces", "")
            if not noisy_traces_str:
                continue  # Skip if no traces available

            noisy_traces = noisy_traces_str.split('|')

            # Calculate metrics
            avg_trace_gt_dist = calculate_avg_trace_to_gt_distance(noisy_traces, ground_truth)
            disagreement_metrics = calculate_inter_trace_disagreement(noisy_traces)
            reconstruction_error = levenshtein_distance(prediction, ground_truth)

            clusters_by_size[cluster_size].append({
                'avg_trace_gt_dist': avg_trace_gt_dist,
                'disagreement': disagreement_metrics['pairwise_distance'],
                'disagreement_std': disagreement_metrics['pairwise_distance_std'],
                'reconstruction_error': reconstruction_error,
                'gt_length': len(ground_truth),
                'is_perfect': reconstruction_error == 0
            })

    # Analyze each cluster size separately
    results = {}

    for cluster_size in sorted(clusters_by_size.keys()):
        clusters = clusters_by_size[cluster_size]

        print(f"\n{'='*70}")
        print(f"CLUSTER SIZE: {cluster_size} ({len(clusters)} clusters)")
        print(f"{'='*70}")

        # Bin clusters by average distance to ground truth FIRST
        max_dist = max(c['avg_trace_gt_dist'] for c in clusters)
        bins = range(0, int(max_dist) + bin_width, bin_width)

        binned_clusters = defaultdict(list)
        for cluster in clusters:
            bin_idx = int(cluster['avg_trace_gt_dist'] // bin_width) * bin_width
            binned_clusters[bin_idx].append(cluster)

        # Analyze each bin
        bin_results = {}
        print(f"\nAnalysis by avg trace-to-GT distance bin:")

        for bin_start in sorted(binned_clusters.keys()):
            bin_clusters = binned_clusters[bin_start]

            if len(bin_clusters) < 10:  # Skip bins with too few samples
                print(f"\n  Distance bin: {bin_start}-{bin_start+bin_width} edits - SKIPPED (only {len(bin_clusters)} clusters)")
                continue

            bin_end = bin_start + bin_width

            # NOW check variability in disagreement WITHIN this bin
            disagreements_in_bin = [c['disagreement'] for c in bin_clusters]
            mean_disagreement = np.mean(disagreements_in_bin)
            std_disagreement = np.std(disagreements_in_bin)
            cv_disagreement = std_disagreement / mean_disagreement if mean_disagreement > 0 else 0

            print(f"\n  Distance bin: {bin_start}-{bin_end} edits (n={len(bin_clusters)})")
            print(f"    Avg trace-to-GT distance in bin: {np.mean([c['avg_trace_gt_dist'] for c in bin_clusters]):.2f} ± {np.std([c['avg_trace_gt_dist'] for c in bin_clusters]):.2f}")
            print(f"    Inter-trace disagreement in bin:")
            print(f"      - Mean: {mean_disagreement:.2f}")
            print(f"      - Std: {std_disagreement:.2f}")
            print(f"      - Min: {np.min(disagreements_in_bin):.2f}")
            print(f"      - Max: {np.max(disagreements_in_bin):.2f}")
            print(f"      - Coefficient of variation: {cv_disagreement:.3f}")

            # Only proceed if there's meaningful variability in disagreement WITHIN this bin
            if cv_disagreement < 0.2:
                print(f"    Low variability in disagreement within this bin (CV={cv_disagreement:.3f}) - skipping")
                continue

            # Split by disagreement level (quartile split: top 25% vs bottom 25%)
            q25 = np.percentile(disagreements_in_bin, 25)
            q75 = np.percentile(disagreements_in_bin, 75)

            low_disagreement = [c for c in bin_clusters if c['disagreement'] <= q25]
            high_disagreement = [c for c in bin_clusters if c['disagreement'] >= q75]

            if len(low_disagreement) < 5 or len(high_disagreement) < 5:
                print(f"    Too few clusters in each group after split (low={len(low_disagreement)}, high={len(high_disagreement)})")
                continue

            # Calculate statistics
            low_errors = [c['reconstruction_error'] for c in low_disagreement]
            high_errors = [c['reconstruction_error'] for c in high_disagreement]

            low_perfect_rate = sum(c['is_perfect'] for c in low_disagreement) / len(low_disagreement)
            high_perfect_rate = sum(c['is_perfect'] for c in high_disagreement) / len(high_disagreement)

            # Calculate mean disagreement in each group
            low_mean_disagreement = np.mean([c['disagreement'] for c in low_disagreement])
            high_mean_disagreement = np.mean([c['disagreement'] for c in high_disagreement])

            # Statistical test
            stat_result = stats.mannwhitneyu(low_errors, high_errors, alternative='two-sided')

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(low_errors)**2 + np.std(high_errors)**2) / 2)
            cohens_d = (np.mean(high_errors) - np.mean(low_errors)) / pooled_std if pooled_std > 0 else 0

            print(f"    Comparison (quartile split: bottom 25% vs top 25%):")
            print(f"      Q25 threshold: {q25:.2f}, Q75 threshold: {q75:.2f}")
            print(f"    Low disagreement (bottom 25%, n={len(low_disagreement)}):")
            print(f"      - Mean inter-trace disagreement: {low_mean_disagreement:.2f}")
            print(f"      - Mean reconstruction error: {np.mean(low_errors):.2f} ± {np.std(low_errors):.2f}")
            print(f"      - Perfect reconstruction rate: {low_perfect_rate:.1%}")
            print(f"    High disagreement (top 25%, n={len(high_disagreement)}):")
            print(f"      - Mean inter-trace disagreement: {high_mean_disagreement:.2f}")
            print(f"      - Mean reconstruction error: {np.mean(high_errors):.2f} ± {np.std(high_errors):.2f}")
            print(f"      - Perfect reconstruction rate: {high_perfect_rate:.1%}")
            print(f"    Effect size (Cohen's d): {cohens_d:.3f}")
            print(f"    Statistical test (Mann-Whitney U): p = {stat_result.pvalue:.4f}")

            if stat_result.pvalue < 0.05:
                direction = "WORSE" if np.mean(high_errors) > np.mean(low_errors) else "BETTER"
                print(f"     Statistically significant: High disagreement leads to {direction} reconstruction!")

            bin_results[bin_start] = {
                'n_total': len(bin_clusters),
                'n_low': len(low_disagreement),
                'n_high': len(high_disagreement),
                'q25': q25,
                'q75': q75,
                'low_mean_disagreement': low_mean_disagreement,
                'high_mean_disagreement': high_mean_disagreement,
                'cv_disagreement': cv_disagreement,
                'low_mean_error': np.mean(low_errors),
                'low_std_error': np.std(low_errors),
                'low_perfect_rate': low_perfect_rate,
                'high_mean_error': np.mean(high_errors),
                'high_std_error': np.std(high_errors),
                'high_perfect_rate': high_perfect_rate,
                'cohens_d': cohens_d,
                'p_value': stat_result.pvalue
            }

        results[cluster_size] = {
            'bins': bin_results,
            'total_clusters': len(clusters)
        }

    return results


def plot_disagreement_impact(results, output_path='./plots/disagreement_impact.pdf'):
    """
    Create visualization of disagreement impact.

    Args:
        results: Results dictionary from analyze_disagreement_impact
        output_path: Path to save the plot
    """
    if not results:
        print("No results to plot")
        return

    # Filter cluster sizes with meaningful results
    cluster_sizes_with_data = [cs for cs, res in results.items() if res['bins']]

    if not cluster_sizes_with_data:
        print("No cluster sizes with sufficient data for plotting")
        return

    # Create figure
    n_cluster_sizes = len(cluster_sizes_with_data)
    fig, axes = plt.subplots(1, n_cluster_sizes, figsize=(3 * n_cluster_sizes, 2.5), squeeze=False)
    axes = axes[0]  # Flatten if only one row

    for idx, cluster_size in enumerate(cluster_sizes_with_data):
        ax = axes[idx] if n_cluster_sizes > 1 else axes[0]
        bin_results = results[cluster_size]['bins']

        # Prepare data for plotting
        bin_centers = []
        low_means = []
        low_stds = []
        high_means = []
        high_stds = []
        p_values = []

        for bin_start in sorted(bin_results.keys()):
            br = bin_results[bin_start]
            bin_centers.append(bin_start + 2.5)  # Assuming bin_width=5
            low_means.append(br['low_mean_error'])
            low_stds.append(br['low_std_error'])
            high_means.append(br['high_mean_error'])
            high_stds.append(br['high_std_error'])
            p_values.append(br['p_value'])

        x = np.arange(len(bin_centers))
        width = 0.35

        # Plot bars
        ax.bar(x - width/2, low_means, width, yerr=low_stds,
               label='Low disagreement', color='#6699CC', alpha=0.8, capsize=3)
        ax.bar(x + width/2, high_means, width, yerr=high_stds,
               label='High disagreement', color='#EE6677', alpha=0.8, capsize=3)

        # Add significance markers
        for i, p_val in enumerate(p_values):
            if p_val < 0.001:
                marker = '***'
            elif p_val < 0.01:
                marker = '**'
            elif p_val < 0.05:
                marker = '*'
            else:
                marker = ''

            if marker:
                max_height = max(high_means[i] + high_stds[i], low_means[i] + low_stds[i])
                ax.text(i, max_height + 0.5, marker, ha='center', fontsize=10)

        ax.set_xlabel('Avg. trace-to-GT distance (edits)')
        ax.set_ylabel('Reconstruction error (edits)')
        ax.set_title(f'Cluster size {cluster_size}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(bc)}' for bc in bin_centers])
        ax.legend(loc='best', frameon=False, fontsize=FONTSIZE-1)

        # Style
        for spine in ax.spines.values():
            spine.set_color('lightgray')
        ax.tick_params(axis='both', which='both', color='lightgray', labelcolor='black')

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\n{'='*70}")
    print(f"Saved plot to: {output_path}")
    print(f"{'='*70}")

    plt.show()


# Run the analysis
if __name__ == "__main__":
    # Update with your actual file paths
    test_files = {
        'microsoft': '<your.data.path>/TReconLM/pred_gt/finetune_microsoft/predictions_output.tsv',
        'noisy_dna': '<your.data.path>/TReconLM/pred_gt/finetune_noisy/predictions_output.tsv',
        'synthetic': '<your.data.path>/TReconLM/pred_gt/synthetic_L110/predictions_output.tsv',
    }

    # Analyze each dataset
    for dataset_name, filepath in test_files.items():
        print(f"\n{'#'*70}")
        print(f"# ANALYZING {dataset_name.upper()} DATASET")
        print(f"{'#'*70}")

        results = analyze_disagreement_impact(filepath, bin_width=5)

        if results:
            # Create plot
            output_path = f'./plots/disagreement_impact_{dataset_name}.pdf'
            plot_disagreement_impact(results, output_path)
