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

# Import from original script
from plot_misclustering_heatmap import (
    fetch_from_wandb,
    process_results_to_matrices,
    BASELINE_LEVENSHTEIN_BY_CLUSTER,
    truncate_colormap
)

# Font size configuration
fontsize = 15.5

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
ENTITY = ""  # your wandb entity (e.g., "<your.wandb.entity>")
PROJECT = ""  # your wandb project name (e.g., "Misclustering")
RUN_ID_1 = ""  # first run id (e.g., "6f93r3di")
RUN_ID_2 = ""  # second run id (e.g., "4118jgqw")

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
