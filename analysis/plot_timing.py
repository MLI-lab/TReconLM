"""
Plot timing and cost analysis from WandB runs.

This script downloads all runs from the "TimingCost" WandB project and creates:
1. Throughput plot (examples/hour) with log scale
2. Throughput per dollar plot (examples/hour/$)

Uses the same color scheme as plots.ipynb for consistency.
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pathlib import Path

# Use LaTeX for text rendering (same as plots.ipynb)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']

# Global font size
FONTSIZE = 8

# Plot settings
SHOW_GRID = False  # Set to True to enable grid lines

# WandB configuration
ENTITY = "<your.wandb.entity>"
PROJECT = "TimingCost"

# Color scheme from plots.ipynb
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
    # GPU methods
    "TReconLM": 3.4,      # A100 GPU
    "RobuSeqNet": 3.4,    # A100 GPU
    "DNAformer": 3.4,     # A100 GPU
    # CPU methods
    "ITR": 4.0,           # 64 CPUs
    "TrellisBMA": 4.0,    # 64 CPUs
    "BMALA": 4.0,         # 64 CPUs
    "MUSCLE": 4.0,        # 64 CPUs
    "VS": 4.0,            # 64 CPUs
}

# Algorithm name mapping (clean names for plot)
ALGO_NAME_MAP = {
    "treconlm": "TReconLM",
    "robuseqnet": "RobuSeqNet",
    "robseqnet": "RobuSeqNet",  # Handle typo variant without 'u'
    "dnaformer": "DNAformer",
    "trellisbma": "TrellisBMA",
    "bmala": "BMALA",
    "muscle": "MUSCLE",
    "itr": "ITR",
    "vs": "VS",
}


def extract_algorithm_name(run_name):
    """Extract algorithm name from WandB run name."""
    run_name_lower = run_name.lower()

    # Check each algorithm
    for key, clean_name in ALGO_NAME_MAP.items():
        if key in run_name_lower:
            return clean_name

    # Fallback: return first word
    return run_name.split('_')[0].capitalize()


def download_timing_runs():
    """Download all runs from TimingCost project."""
    print(f"Downloading runs from {ENTITY}/{PROJECT}...")

    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    timing_data = {}

    for run in runs:
        # Get run name and extract algorithm
        run_name = run.name
        algo = extract_algorithm_name(run_name)

        # Get timing metrics
        summary = run.summary._json_dict

        # Check if this run has timing data
        if 'timing_mean_throughput_per_hour' not in summary:
            print(f"Skipping {run_name} (no timing data)")
            continue

        mean_throughput = summary.get('timing_mean_throughput_per_hour')
        std_throughput = summary.get('timing_std_throughput_per_hour')
        cv_percent = summary.get('timing_cv_throughput_percent')

        if mean_throughput is None or std_throughput is None:
            print(f"Skipping {run_name} (incomplete timing data)")
            continue

        print(f"Found: {run_name} {algo}: {mean_throughput:.0f} ± {std_throughput:.0f} ex/hr")

        # Store data (keep the best/latest run for each algorithm)
        if algo not in timing_data or mean_throughput > timing_data[algo]['mean']:
            timing_data[algo] = {
                'mean': mean_throughput,
                'std': std_throughput,
                'cv': cv_percent,
                'run_name': run_name,
            }

    return timing_data


def plot_combined_throughput_and_cost(timing_data, output_path='./plots/timing_combined.pdf'):
    """
    Plot throughput and throughput per dollar as grouped bar plots.

    Two bars per algorithm:
    - Throughput (examples/hour) - solid colored bar
    - Throughput per dollar (examples/hour/$) - transparent colored bar

    Both use log scale, with numbers and std displayed on bars.
    Algorithm legend with colors shown on top.
    """

    # Calculate throughput per dollar
    cost_efficiency = {}
    for algo, data in timing_data.items():
        if algo in COST_PER_HOUR:
            cost = COST_PER_HOUR[algo]
            mean_efficiency = data['mean'] / cost
            std_efficiency = data['std'] / cost
            cost_efficiency[algo] = {
                'mean': mean_efficiency,
                'std': std_efficiency,
            }

    # Sort algorithms by throughput per dollar (descending)
    sorted_algos = sorted(cost_efficiency.keys(),
                         key=lambda a: cost_efficiency[a]['mean'],
                         reverse=True)

    # Prepare data
    n_algos = len(sorted_algos)
    y_positions = np.arange(n_algos)
    bar_height = 0.4  # Increase for thicker bars (default was 0.35)

    throughput_means = [timing_data[algo]['mean'] for algo in sorted_algos]
    throughput_stds = [timing_data[algo]['std'] for algo in sorted_algos]

    efficiency_means = [cost_efficiency[algo]['mean'] for algo in sorted_algos]
    efficiency_stds = [cost_efficiency[algo]['std'] for algo in sorted_algos]

    colors = [COLOR_DICT.get(algo, '#000000') for algo in sorted_algos]

    # Create figure
    fig, ax = plt.subplots(figsize=(6.3, 2.95))

    # Plot bars with algorithm colors
    # Per hour = transparent (alpha=0.3), Per dollar = solid (alpha=0.9)
    bars1 = ax.barh(y_positions - bar_height/2, throughput_means, bar_height,
                    color=colors, alpha=0.3, edgecolor='white', linewidth=0.5)
    bars2 = ax.barh(y_positions + bar_height/2, efficiency_means, bar_height,
                    color=colors, alpha=0.9, edgecolor='white', linewidth=0.5)

    # Add text labels with mean ± std on bars
    # Try to place text inside bar at the end, otherwise outside
    xlim = ax.get_xlim()
    for i, (algo, t_mean, t_std, e_mean, e_std) in enumerate(
        zip(sorted_algos, throughput_means, throughput_stds, efficiency_means, efficiency_stds)):

        # Throughput per hour bar text (transparent bars - black text)
        text_str1 = f'{t_mean:,.0f}$\\pm${t_std:,.0f}'
        # Check if text fits inside bar (rough estimate: log scale makes this tricky)
        # If bar is long enough, place inside at end; otherwise outside
        if t_mean > xlim[0] * 10:  # Bar is long enough
            ax.text(t_mean * 0.98, i - bar_height/2, f'{text_str1}  ',
                   va='center', ha='right', fontsize=FONTSIZE-1, fontweight='bold', color='black')
        else:  # Bar too short, place outside
            ax.text(t_mean, i - bar_height/2, f'  {text_str1}',
                   va='center', ha='left', fontsize=FONTSIZE-1, fontweight='bold', color='black')

        # Throughput per dollar bar text (solid bars - white text inside)
        text_str2 = f'{e_mean:,.0f}$\\pm${e_std:,.0f}'
        # Special case: TrellisBMA per dollar doesn't fit, always place outside
        if algo == 'TrellisBMA':
            ax.text(e_mean, i + bar_height/2, f'  {text_str2}',
                   va='center', ha='left', fontsize=FONTSIZE-1, fontweight='bold', color='black')
        elif e_mean > xlim[0] * 10:  # Bar is long enough
            ax.text(e_mean * 0.98, i + bar_height/2, f'{text_str2}  ',
                   va='center', ha='right', fontsize=FONTSIZE-1, fontweight='bold', color='white')
        else:  # Bar too short, place outside
            ax.text(e_mean, i + bar_height/2, f'  {text_str2}',
                   va='center', ha='left', fontsize=FONTSIZE-1, fontweight='bold', color='black')

    # Set log scale on x-axis
    ax.set_xscale('log')

    # Set x-axis limits to cut off earlier (trim right side)
    max_throughput = max(max(throughput_means), max(efficiency_means))
    ax.set_xlim([None, max_throughput * 1.3])  # 1.3x max value, adjust multiplier as needed

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_algos, fontsize=FONTSIZE)

    # X-axis label
    ax.set_xlabel('Throughput:', fontsize=FONTSIZE, x=0.32)

    # Add bar type indicators to legend (per hour / per dollar) - keep only these
    handles = [
        Patch(facecolor='gray', alpha=0.9, edgecolor='white', linewidth=0.5),
        Patch(facecolor='gray', alpha=0.3, edgecolor='white', linewidth=0.5),
    ]
    labels = ['per dollar', 'per hour']

    # Legend at the bottom below x-axis label
    fig.legend(handles, labels,
              loc='lower center',
              ncol=2,
              bbox_to_anchor=(0.6, 0.096),
              columnspacing=0.7,
              frameon=False,
              fontsize=FONTSIZE)

    # Grid - consistent with scaling laws (optional)
    if SHOW_GRID:
        ax.grid(True, which='both', axis='x', ls='--', lw=0.3)
        ax.set_axisbelow(True)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')

    # Set tick marks gray, labels black - consistent with scaling laws
    ax.tick_params(axis='both', which='both', color='lightgray', labelcolor='black', labelsize=FONTSIZE)

    # Tight layout with space for legend at bottom
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to {output_path}")

    plt.show()


def print_summary_table(timing_data):
    """Print a summary table with all metrics."""
    print("\n" + "="*100)
    print("TIMING SUMMARY")
    print("="*100)
    print(f"{'Algorithm':<15} {'Throughput (ex/hr)':<25} {'CV (%)':<10} {'Cost ($/hr)':<12} {'Ex/hr/$':<15}")
    print("-"*100)

    # Sort by throughput per dollar
    sorted_algos = sorted(timing_data.keys(),
                         key=lambda a: timing_data[a]['mean'] / COST_PER_HOUR.get(a, 1),
                         reverse=True)

    for algo in sorted_algos:
        data = timing_data[algo]
        cost = COST_PER_HOUR.get(algo, float('nan'))
        efficiency = data['mean'] / cost if cost else float('nan')

        print(f"{algo:<15} {data['mean']:>10.0f} ± {data['std']:<8.0f} {data['cv']:>8.2f} "
              f"{cost:>10.1f}   {efficiency:>12.0f}")

    print("="*100)
    print("\nHardware costs:")
    print(f"  GPU (A100): $3.4/hour")
    print(f"  CPU (64 cores): $4.0/hour")
    print("="*100)


def main():
    """Main function to download data and create plots."""
    # Download timing data
    timing_data = download_timing_runs()

    if not timing_data:
        print("No timing data found!")
        return

    # Print summary table
    print_summary_table(timing_data)

    # Create combined plot
    plot_combined_throughput_and_cost(timing_data)

    print("\nDone! Plot saved to ./plots/timing_combined.pdf")


if __name__ == "__main__":
    main()
