"""
Plot timing and cost analysis from WandB runs.

This script downloads all runs from the "TimingCost" WandB project and creates
two side-by-side vertical bar plots:
1. Throughput (examples/hour) with log scale
2. Throughput per dollar (examples/hour/$) with log scale

Uses the same color scheme and style as main_plots.ipynb.
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

# Use LaTeX for text rendering (same as main_plots.ipynb)
fontsize = 8.8
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{type1cm}',
    'font.size': fontsize,
})

# WandB configuration
ENTITY = "franziweindel-technical-university-of-munich"
PROJECT = "TimingCost"

# Color scheme (same as main_plots.ipynb)
COLOR_DICT = {
    "TReconLM": "#6699CC",
    "ITR": "#66CCEE",
    "DNAformer": "#9CAF88",
    "TrellisBMA": "#CCBB44",
    "BMALA": "#EE6677",
    "RobuSeqNet": "#F8DB57",
    "MUSCLE": "#F287BD",
    "VS": "#FFAA00",
}

MARKER_DICT = {
    "TReconLM": "o",
    "ITR": "s",
    "DNAformer": "^",
    "TrellisBMA": "v",
    "BMALA": "D",
    "RobuSeqNet": "P",
    "MUSCLE": "X",
    "VS": "*",
}

# Hardware costs (per hour)
COST_PER_HOUR = {
    "TReconLM": 3.4,      # A100 GPU
    "RobuSeqNet": 3.4,    # A100 GPU
    "DNAformer": 3.4,     # A100 GPU
    "ITR": 4.0,           # 64 CPUs
    "TrellisBMA": 4.0,    # 64 CPUs
    "BMALA": 4.0,         # 64 CPUs
    "MUSCLE": 4.0,        # 64 CPUs
    "VS": 4.0,            # 64 CPUs
}

# Algorithm name mapping
ALGO_NAME_MAP = {
    "treconlm": "TReconLM",
    "robuseqnet": "RobuSeqNet",
    "robseqnet": "RobuSeqNet",
    "dnaformer": "DNAformer",
    "trellisbma": "TrellisBMA",
    "bmala": "BMALA",
    "muscle": "MUSCLE",
    "itr": "ITR",
    "vs": "VS",
}

# Preferred display order (same as main_plots.ipynb legend)
PREFERRED_ORDER = ["TReconLM", "DNAformer", "ITR", "MUSCLE", "BMALA", "TrellisBMA", "VS", "RobuSeqNet"]


def extract_algorithm_name(run_name):
    """Extract algorithm name from WandB run name."""
    run_name_lower = run_name.lower()
    for key, clean_name in ALGO_NAME_MAP.items():
        if key in run_name_lower:
            return clean_name
    return run_name.split('_')[0].capitalize()


def download_timing_runs():
    """Download all runs from TimingCost project."""
    print(f"Downloading runs from {ENTITY}/{PROJECT}...")

    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    timing_data = {}

    for run in runs:
        run_name = run.name
        algo = extract_algorithm_name(run_name)
        summary = run.summary._json_dict

        if 'timing_mean_throughput_per_hour' not in summary:
            print(f"  Skipping {run_name} (no timing data)")
            continue

        mean_throughput = summary.get('timing_mean_throughput_per_hour')
        std_throughput = summary.get('timing_std_throughput_per_hour')
        cv_percent = summary.get('timing_cv_throughput_percent')

        if mean_throughput is None or std_throughput is None:
            print(f"  Skipping {run_name} (incomplete timing data)")
            continue

        print(f"  [Timing] Using run: {run_name} (algo={algo}): {mean_throughput:.0f} +/- {std_throughput:.0f} ex/hr")

        if algo not in timing_data or mean_throughput > timing_data[algo]['mean']:
            timing_data[algo] = {
                'mean': mean_throughput,
                'std': std_throughput,
                'cv': cv_percent,
                'run_name': run_name,
            }

    return timing_data


def plot_timing(timing_data, output_path='./plots/timing_combined.pdf'):
    """
    Two side-by-side vertical bar plots matching main_plots.ipynb style.
    Left: throughput per hour. Right: throughput per dollar.
    """

    # Calculate throughput per dollar
    cost_efficiency = {}
    for algo, data in timing_data.items():
        if algo in COST_PER_HOUR:
            cost = COST_PER_HOUR[algo]
            cost_efficiency[algo] = {
                'mean': data['mean'] / cost,
                'std': data['std'] / cost,
            }

    # Sort by preferred order, only include algos we have data for
    sorted_algos = [a for a in PREFERRED_ORDER if a in timing_data and a in cost_efficiency]

    n_algos = len(sorted_algos)
    x = np.arange(n_algos)
    colors = [COLOR_DICT.get(algo, '#000000') for algo in sorted_algos]

    bar_width = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 1.37), dpi=300, gridspec_kw={'wspace': 0.4})

    # Left: throughput per hour
    throughput_means = [timing_data[a]['mean'] for a in sorted_algos]
    throughput_stds = [timing_data[a]['std'] for a in sorted_algos]

    ax1.bar(x, throughput_means, width=bar_width, color=colors, edgecolor=None, linewidth=0, zorder=2)
    for i in range(n_algos):
        ax1.errorbar(x[i], throughput_means[i], yerr=throughput_stds[i], fmt='none',
                     ecolor='gray', elinewidth=0.4, capsize=1.5, capthick=0.4, zorder=3)
    ax1.set_yscale('log')
    ax1.set_ylabel('Throughput per hour')
    ax1.yaxis.set_label_coords(-0.2, 0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_algos, rotation=45, ha='right', fontsize=fontsize - 1)
    ax1.set_xlim(-0.5, n_algos - 0.5)
    ax1.grid(False)

    # Right: throughput per dollar
    efficiency_means = [cost_efficiency[a]['mean'] for a in sorted_algos]
    efficiency_stds = [cost_efficiency[a]['std'] for a in sorted_algos]

    ax2.bar(x, efficiency_means, width=bar_width, color=colors, edgecolor=None, linewidth=0, zorder=2)
    for i in range(n_algos):
        ax2.errorbar(x[i], efficiency_means[i], yerr=efficiency_stds[i], fmt='none',
                     ecolor='gray', elinewidth=0.4, capsize=1.5, capthick=0.4, zorder=3)
    ax2.set_yscale('log')
    ax2.set_ylabel('Throughput per dollar')
    ax2.yaxis.set_label_coords(-0.2, 0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sorted_algos, rotation=45, ha='right', fontsize=fontsize - 1)
    ax2.set_xlim(-0.5, n_algos - 0.5)
    ax2.grid(False)

    # Style spines and ticks (same as main_plots.ipynb)
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_color('lightgray')
        ax.tick_params(axis='both', which='both', color='lightgray', labelcolor='black')

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.show()


def print_summary_table(timing_data):
    """Print a summary table with all metrics."""
    print("\n" + "=" * 100)
    print("TIMING SUMMARY")
    print("=" * 100)
    print(f"{'Algorithm':<15} {'Throughput (ex/hr)':<25} {'CV (%)':<10} {'Cost ($/hr)':<12} {'Ex/hr/$':<15}")
    print("-" * 100)

    sorted_algos = sorted(timing_data.keys(),
                          key=lambda a: timing_data[a]['mean'] / COST_PER_HOUR.get(a, 1),
                          reverse=True)

    for algo in sorted_algos:
        data = timing_data[algo]
        cost = COST_PER_HOUR.get(algo, float('nan'))
        efficiency = data['mean'] / cost if cost else float('nan')
        print(f"{algo:<15} {data['mean']:>10.0f} +/- {data['std']:<8.0f} {data['cv']:>8.2f} "
              f"{cost:>10.1f}   {efficiency:>12.0f}")

    print("=" * 100)
    print("\nHardware costs:")
    print(f"  GPU (A100): $3.4/hour")
    print(f"  CPU (64 cores): $4.0/hour")
    print("=" * 100)


def main():
    """Main function to download data and create plots."""
    timing_data = download_timing_runs()

    if not timing_data:
        print("No timing data found!")
        return

    print_summary_table(timing_data)
    plot_timing(timing_data)

    print("\nDone!")


if __name__ == "__main__":
    main()
