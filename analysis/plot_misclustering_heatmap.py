#!/usr/bin/env python3
"""
Script to generate side-by-side misclustering robustness heatmaps for comparison.
Configure the variables below and run to fetch data from WandB and generate comparison heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import os
from collections import defaultdict
import wandb
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm


def truncate_colormap(cmap, minval=0.1, maxval=1.0, n=100):
    """Truncate colormap to skip lightest colors."""
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval},{maxval})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


# Baseline Levenshtein per cluster size (no misclustering).
# These are subtracted so the heatmap shows the *increase* caused by misclustering.
# Update with your own baseline values if needed.
BASELINE_LEVENSHTEIN_BY_CLUSTER = {
    2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
    6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0,
}


def fetch_from_wandb(entity, project, run_id):
    """
    Fetch misclustering experiment results from a W&B run summary.

    Returns a dict with:
      - 'results_by_condition': {condition_name: {'mean_levenshtein': float, ...}}
      - 'contamination_rates': list of floats
      - 'cluster_size_matrices': dict with 2D matrices keyed by cluster size
        (levenshtein_matrix, counts_matrix, contamination_rates, cluster_sizes)
    """
    wandb.login()
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    summary = dict(run.summary)

    # ── Try to grab pre-built cluster-size 2D matrices first ──
    cs_lev = summary.get('misclustering_cluster_size_levenshtein_matrix')
    cs_counts = summary.get('misclustering_cluster_size_counts_matrix')
    cs_rates = summary.get('misclustering_cluster_size_contamination_rates')
    cs_sizes = summary.get('misclustering_cluster_size_cluster_sizes')

    if cs_lev is not None:
        return {
            'source': 'cluster_size_matrices',
            'levenshtein_matrix': np.array(cs_lev, dtype=float),
            'counts_matrix': np.array(cs_counts, dtype=float),
            'contamination_rates': cs_rates,
            'cluster_sizes': cs_sizes,
        }

    # ── Fallback: reconstruct from per-condition / per-cluster keys ──
    import re

    # Collect all per-condition, per-cluster-size metrics
    # Keys look like: misclustering_{condition}_N{N}_mean_levenshtein
    #   where condition encodes the contamination rate, e.g. "cont_0.02_cs10"
    per_condition = defaultdict(dict)
    contamination_rates_set = set()
    cluster_sizes_set = set()

    pattern = re.compile(
        r'^misclustering_(.+?)_N(\d+)_mean_levenshtein$'
    )
    for key, val in summary.items():
        m = pattern.match(key)
        if m:
            condition = m.group(1)
            N = int(m.group(2))
            per_condition[(condition, N)]['mean_levenshtein'] = float(val)

    # Also grab contaminated-only levenshtein
    pattern_cont = re.compile(
        r'^misclustering_(.+?)_N(\d+)_contaminated_mean_levenshtein$'
    )
    for key, val in summary.items():
        m = pattern_cont.match(key)
        if m:
            condition = m.group(1)
            N = int(m.group(2))
            per_condition[(condition, N)]['contaminated_mean_levenshtein'] = float(val)

    # Parse contamination rate from condition name (e.g. "cont_0.02" or "rate_0.02_...")
    rate_re = re.compile(r'(?:cont|rate)[_=]?([\d.]+)')
    for (condition, N), _ in per_condition.items():
        rm = rate_re.search(condition)
        if rm:
            contamination_rates_set.add(float(rm.group(1)))
        cluster_sizes_set.add(N)

    return {
        'source': 'per_condition',
        'per_condition': per_condition,
        'contamination_rates_set': contamination_rates_set,
        'cluster_sizes_set': cluster_sizes_set,
        'raw_summary': summary,
    }


def process_results_to_matrices(results):
    """
    Convert fetched W&B results into the matrix dict expected by the plotting code.

    Returns dict with keys:
      - levenshtein_matrix:                shape (n_clusters, n_rates)
      - contaminated_levenshtein_matrix:   shape (n_clusters, n_rates)
      - counts_matrix:                     shape (n_clusters, n_rates)
      - avg_contaminants_all_matrix:       shape (n_clusters, n_rates)
      - avg_contaminants_contaminated_matrix: shape (n_clusters, n_rates)
      - bin_labels:          list of str  (cluster sizes)
      - contamination_rates: list of float
    """
    if results is None:
        return None

    # ── Pre-built matrices from W&B ──
    if results.get('source') == 'cluster_size_matrices':
        lev = np.array(results['levenshtein_matrix'], dtype=float)
        counts = np.array(results['counts_matrix'], dtype=float)
        rates = [float(r) for r in results['contamination_rates']]
        sizes = [str(int(s)) for s in results['cluster_sizes']]
        return {
            'levenshtein_matrix': lev,
            'contaminated_levenshtein_matrix': lev.copy(),
            'counts_matrix': counts,
            'avg_contaminants_all_matrix': np.zeros_like(lev),
            'avg_contaminants_contaminated_matrix': np.zeros_like(lev),
            'bin_labels': sizes,
            'contamination_rates': rates,
        }

    # ── Reconstruct from per-condition keys ──
    import re
    per_condition = results['per_condition']
    contamination_rates = sorted(results['contamination_rates_set'])
    cluster_sizes = sorted(results['cluster_sizes_set'])

    n_clusters = len(cluster_sizes)
    n_rates = len(contamination_rates)

    lev_all = np.full((n_clusters, n_rates), np.nan)
    lev_cont = np.full((n_clusters, n_rates), np.nan)
    counts = np.ones((n_clusters, n_rates))

    rate_re = re.compile(r'(?:cont|rate)[_=]?([\d.]+)')

    for (condition, N), metrics in per_condition.items():
        rm = rate_re.search(condition)
        if not rm:
            continue
        rate = float(rm.group(1))
        if rate not in contamination_rates or N not in cluster_sizes:
            continue
        ri = contamination_rates.index(rate)
        ci = cluster_sizes.index(N)
        if 'mean_levenshtein' in metrics:
            lev_all[ci, ri] = metrics['mean_levenshtein']
        if 'contaminated_mean_levenshtein' in metrics:
            lev_cont[ci, ri] = metrics['contaminated_mean_levenshtein']

    # Where contaminated is missing, fall back to all
    mask = np.isnan(lev_cont)
    lev_cont[mask] = lev_all[mask]

    return {
        'levenshtein_matrix': lev_all,
        'contaminated_levenshtein_matrix': lev_cont,
        'counts_matrix': counts,
        'avg_contaminants_all_matrix': np.zeros_like(lev_all),
        'avg_contaminants_contaminated_matrix': np.zeros_like(lev_all),
        'bin_labels': [str(s) for s in cluster_sizes],
        'contamination_rates': contamination_rates,
    }

# Font size configuration
fontsize = 13

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{type1cm}',
    'font.size': fontsize,
})

# 
# Configuration - set your WandB and output settings
# 

# WandB settings
ENTITY = "franziweindel-technical-university-of-munich"  # your wandb entity (e.g., "<your.wandb.entity>")
PROJECT = "Misclustering"  # your wandb project name (e.g., "Misclustering")
RUN_ID_1 = "6f93r3di"  # first run id (e.g., "6f93r3di")
RUN_ID_2 = "4118jgqw"  # second run id (e.g., "4118jgqw")

# Output settings
SAVE_DIR = "./plots"
METRIC = "levenshtein"
SAVE_PATH = "./plots/miscluster_heatmap.pdf"

# Figure size
BASE_W, BASE_H = 10, 4  # increased height for better visibility

# 

# Colormap
trunc_cmap = truncate_colormap(cm.PuBu, minval=0.0, maxval=1.0)

def plot_combined_heatmap(ax, matrix_data_1, matrix_data_2, vmin=None, vmax=None):
    """Plot a single heatmap combining data from both runs with 4 horizontal quarters per cell."""

    bin_labels = matrix_data_1['bin_labels']
    contamination_rates = matrix_data_1['contamination_rates']

    # Prepare matrices for both runs
    # Run 1
    lev_matrix_all_1 = matrix_data_1['levenshtein_matrix'].copy()
    lev_matrix_cont_1 = matrix_data_1.get('contaminated_levenshtein_matrix', lev_matrix_all_1).copy()
    avg_contam_all_1 = matrix_data_1.get('avg_contaminants_all_matrix', np.zeros_like(lev_matrix_all_1))
    avg_contam_cont_1 = matrix_data_1.get('avg_contaminants_contaminated_matrix', np.zeros_like(lev_matrix_all_1))

    # Run 2
    lev_matrix_all_2 = matrix_data_2['levenshtein_matrix'].copy()
    lev_matrix_cont_2 = matrix_data_2.get('contaminated_levenshtein_matrix', lev_matrix_all_2).copy()
    avg_contam_all_2 = matrix_data_2.get('avg_contaminants_all_matrix', np.zeros_like(lev_matrix_all_2))
    avg_contam_cont_2 = matrix_data_2.get('avg_contaminants_contaminated_matrix', np.zeros_like(lev_matrix_all_2))

    counts_matrix = matrix_data_1['counts_matrix']

    # Subtract cluster-specific baseline
    for i, label in enumerate(bin_labels):
        try:
            cluster_size = int(label)
            if cluster_size in BASELINE_LEVENSHTEIN_BY_CLUSTER:
                baseline = BASELINE_LEVENSHTEIN_BY_CLUSTER[cluster_size]
                lev_matrix_all_1[i, :] -= baseline
                lev_matrix_cont_1[i, :] -= baseline
                lev_matrix_all_2[i, :] -= baseline
                lev_matrix_cont_2[i, :] -= baseline
        except (ValueError, KeyError):
            pass

    # Calculate vmin/vmax if not provided
    if vmin is None or vmax is None:
        zero_mask = (counts_matrix == 0) | np.isnan(lev_matrix_all_1)
        all_data = np.concatenate([
            lev_matrix_all_1[~zero_mask],
            lev_matrix_cont_1[~np.isnan(lev_matrix_cont_1)],
            lev_matrix_all_2[~zero_mask],
            lev_matrix_cont_2[~np.isnan(lev_matrix_cont_2)]
        ])
        if len(all_data) > 0:
            vmin = np.nanmin(all_data)
            vmax = np.nanmax(all_data)
        else:
            vmin, vmax = 0, 1

    zero_mask = (counts_matrix == 0) | np.isnan(lev_matrix_all_1)
    rows, cols = lev_matrix_all_1.shape

    # Draw cells with 4 horizontal quarters
    for i in range(rows):
        for j in range(cols):
            if zero_mask[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=cm.PuBu(0.1), ec='none'))
                ax.text(j + 0.5, i + 0.5, '0', ha='center', va='center',
                       fontsize=fontsize-5, weight='bold')
            else:
                # Quarter 1 (bottom): run1 all examples
                val_1_all = lev_matrix_all_1[i, j]
                color_1_all = trunc_cmap((val_1_all - vmin) / (vmax - vmin)) if not np.isnan(val_1_all) else cm.PuBu(0.1)
                ax.add_patch(plt.Rectangle((j, i), 1, 0.25, fill=True, color=color_1_all, ec='none'))

                # Quarter 2: run2 all examples
                val_2_all = lev_matrix_all_2[i, j]
                color_2_all = trunc_cmap((val_2_all - vmin) / (vmax - vmin)) if not np.isnan(val_2_all) else cm.PuBu(0.1)
                ax.add_patch(plt.Rectangle((j, i + 0.25), 1, 0.25, fill=True, color=color_2_all, ec='none'))

                # Quarter 3: run1 contaminated only
                val_1_cont = lev_matrix_cont_1[i, j]
                color_1_cont = trunc_cmap((val_1_cont - vmin) / (vmax - vmin)) if not np.isnan(val_1_cont) else cm.PuBu(0.1)
                ax.add_patch(plt.Rectangle((j, i + 0.5), 1, 0.25, fill=True, color=color_1_cont, ec='none'))

                # Quarter 4 (top): run2 contaminated only
                val_2_cont = lev_matrix_cont_2[i, j]
                color_2_cont = trunc_cmap((val_2_cont - vmin) / (vmax - vmin)) if not np.isnan(val_2_cont) else cm.PuBu(0.1)
                ax.add_patch(plt.Rectangle((j, i + 0.75), 1, 0.25, fill=True, color=color_2_cont, ec='none'))

                # Annotations - write contamination counts once for bottom two and once for top two
                contam_all = avg_contam_all_1[i, j]  # Same for both runs
                contam_cont = avg_contam_cont_1[i, j]  # Same for both runs

                # Text color based on average of the two quarters
                avg_bottom = (val_1_all + val_2_all) / 2 if not (np.isnan(val_1_all) or np.isnan(val_2_all)) else val_1_all
                text_color_bottom = 'white' if not np.isnan(avg_bottom) and (avg_bottom - vmin) / (vmax - vmin) > 0.5 else 'black'

                avg_top = (val_1_cont + val_2_cont) / 2 if not (np.isnan(val_1_cont) or np.isnan(val_2_cont)) else val_1_cont
                text_color_top = 'white' if not np.isnan(avg_top) and (avg_top - vmin) / (vmax - vmin) > 0.5 else 'black'

                # Write contamination count once for bottom two quarters (centered at 0.25)
                if not np.isnan(contam_all):
                    ax.text(j + 0.5, i + 0.25, f'{contam_all:.1f}', ha='center', va='center',
                           fontsize=fontsize-5, color=text_color_bottom)

                # Write contamination count once for top two quarters (centered at 0.75)
                if not np.isnan(contam_cont):
                    ax.text(j + 0.5, i + 0.75, f'{contam_cont:.1f}', ha='center', va='center',
                           fontsize=fontsize-5, color=text_color_top)

    # Grid lines
    for j in range(1, cols):
        ax.plot([j, j], [0, rows], color='white', linewidth=1.5, zorder=10)
    for i in range(1, rows):
        ax.plot([0, cols], [i, i], color='white', linewidth=1.5, zorder=10)

    # Axis settings
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('auto')
    ax.invert_yaxis()

    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_xticklabels([f"{r:.2f}" for r in contamination_rates])
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_yticklabels(bin_labels)
    ax.set_ylabel('Cluster size', fontsize=fontsize)
    ax.set_xlabel(r'Miscluster rate $p_m$', fontsize=fontsize)

    for spine in ax.spines.values():
        spine.set_edgecolor('lightgrey')
        spine.set_linewidth(1)

    ax.tick_params(axis='both', which='both', color='lightgrey', labelcolor='black', length=4, labelsize=fontsize)

    return vmin, vmax


def create_comparison_heatmap(matrix_data_1, matrix_data_2, save_path=None):
    """Create single wide heatmap combining data from both runs."""

    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(BASE_W, BASE_H))

    # Get shared vmin/vmax from both datasets
    zero_mask_1 = (matrix_data_1['counts_matrix'] == 0) | np.isnan(matrix_data_1['levenshtein_matrix'])
    zero_mask_2 = (matrix_data_2['counts_matrix'] == 0) | np.isnan(matrix_data_2['levenshtein_matrix'])

    all_data_1 = matrix_data_1['levenshtein_matrix'][~zero_mask_1]
    all_data_2 = matrix_data_2['levenshtein_matrix'][~zero_mask_2]
    all_data = np.concatenate([all_data_1, all_data_2])

    if len(all_data) > 0:
        vmin = np.nanmin(all_data)
        vmax = np.nanmax(all_data)
    else:
        vmin, vmax = 0, 1

    print(f"Shared colorbar range: {vmin:.3f} - {vmax:.3f}")

    # Plot combined heatmap
    vmin, vmax = plot_combined_heatmap(ax, matrix_data_1, matrix_data_2, vmin, vmax)

    # Add colorbar
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    # Adjust spacing to make room for colorbar
    plt.subplots_adjust(right=0.88)

    pos = ax.get_position()
    cbar_width = 0.015
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0, cbar_width, pos.height])
    norm = Normalize(vmin=vmin, vmax=vmax)
    cb = ColorbarBase(cax, cmap=trunc_cmap, norm=norm)
    cb.set_label(r'$d_L$ increase', fontsize=fontsize)

    cb.outline.set_edgecolor('lightgrey')
    cb.outline.set_linewidth(1)

    num_ticks = 5
    tick_values = np.linspace(vmin, vmax, num_ticks)
    cb.set_ticks(tick_values)
    cb.set_ticklabels([f"{val:.3f}" for val in tick_values])
    cax.tick_params(color='lightgrey', labelcolor='black', length=4, labelsize=fontsize)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison heatmap to {save_path}")

    plt.show()

    return fig


if __name__ == "__main__":
    print("Misclustering robustness comparison heatmap generator")
    print("=" * 60)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Fetch data for both runs
    print(f"Fetching data for run 1: {ENTITY}/{PROJECT}/{RUN_ID_1}")
    results_1 = fetch_from_wandb(ENTITY, PROJECT, RUN_ID_1)
    matrix_data_1 = process_results_to_matrices(results_1)

    print(f"\nFetching data for run 2: {ENTITY}/{PROJECT}/{RUN_ID_2}")
    results_2 = fetch_from_wandb(ENTITY, PROJECT, RUN_ID_2)
    matrix_data_2 = process_results_to_matrices(results_2)

    if matrix_data_1 is None or matrix_data_2 is None:
        print("Error: could not process results into matrices")
        exit(1)

    # Generate comparison plot
    print(f"\nGenerating comparison heatmap...")
    create_comparison_heatmap(matrix_data_1, matrix_data_2, save_path=SAVE_PATH)

    print("Done")
