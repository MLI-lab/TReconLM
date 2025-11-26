# 
# VOTE MARGIN vs PROBABILITY MARGIN & ENTROPY - MULTI-COMPUTE BUDGETS
# 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# LaTeX style settings (matching your paper style)
fontsize = 7.7
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{type1cm}',
    'font.size': fontsize,
})

# Set your position metrics file paths
compute_budgets = {
    '6e17': '',  # path to position_metrics parquet file for 6e17 FLOPs
    '1e18': '',  # path to position_metrics parquet file for 1e18 FLOPs
    '3e18': '',  # path to position_metrics parquet file for 3e18 FLOPs
}

# Labels for legend
budget_labels = {
    '6e17': r'$6 \times 10^{17}$ FLOPs',
    '1e18': r'$1 \times 10^{18}$ FLOPs',
    '3e18': r'$3 \times 10^{18}$ FLOPs'
}

save_dir = "./plots"
os.makedirs(save_dir, exist_ok=True)

def load_position_metrics(path):
    """Load position metrics from saved Parquet file."""
    if path is None or path == 'None':
        return None
    try:
        df = pd.read_parquet(path)
        print(f"  Loaded {len(df):,} positions from {path.split('/')[-1]}")
        return df
    except Exception as e:
        print(f"  Error loading {path}: {e}")
        return None

def plot_multi_compute_budgets(compute_data, save_path="vote_margin_multi_compute.pdf"):
    """Create dual plot for multiple compute budgets."""

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 1.1), dpi=300,
                                    gridspec_kw={'wspace': 0.4})
    

    # Define specific colors and markers
    prob_colors = ['#6699CC', '#8AB3DD', '#B3D1EE']  # Blue shades
    entropy_colors = ['#9D9AC5', '#B5B0D6', '#CEC6E7']  # Purple shades
    markers = ['o', 's', '^']  # Circle, square, triangle

    # Process each compute budget
    for idx, (budget, df) in enumerate(compute_data.items()):
        if df is None:
            continue

        # Group by vote margin and calculate statistics
        grouped = df.groupby('vote_margin').agg({
            'prob_margin': ['mean', 'std'],
            'entropy': ['mean', 'std'],
            'vote_margin': 'count'
        }).round(4)

        # Flatten column names
        grouped.columns = ['prob_margin_mean', 'prob_margin_std',
                          'entropy_mean', 'entropy_std', 'count']
        grouped = grouped.reset_index()
        grouped = grouped.sort_values('vote_margin')

        # LEFT PLOT: Vote margin vs Probability margin
        ax1.plot(grouped['vote_margin'], grouped['prob_margin_mean'],
                color=prob_colors[idx], linewidth=0.5,
                marker=markers[idx], markersize=2,
                label=budget_labels[budget])

        # Shaded region with matching color but lighter
        ax1.fill_between(grouped['vote_margin'],
                        grouped['prob_margin_mean'] - grouped['prob_margin_std'],
                        grouped['prob_margin_mean'] + grouped['prob_margin_std'],
                        alpha=0.15, color=prob_colors[idx])

        # RIGHT PLOT: Vote margin vs Entropy
        ax2.plot(grouped['vote_margin'], grouped['entropy_mean'],
                color=entropy_colors[idx], linewidth=0.5,
                marker=markers[idx], markersize=2,
                label=budget_labels[budget])

        # Shaded region with matching color but lighter
        ax2.fill_between(grouped['vote_margin'],
                        grouped['entropy_mean'] - grouped['entropy_std'],
                        grouped['entropy_mean'] + grouped['entropy_std'],
                        alpha=0.15, color=entropy_colors[idx])
    
    # Configure left plot
    ax1.set_xlabel(r'Vote margin', fontweight='bold')
    ax1.set_ylabel(r'Probability margin', fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='lower right', fontsize=fontsize-1.5, frameon=False)

    # Configure right plot
    ax2.set_xlabel(r'Vote margin', fontweight='bold')
    ax2.set_ylabel(r'Entropy', fontweight='bold')
    ax2.set_xlim(0, 1.1)
    max_entropy = np.log(4)
    ax2.set_ylim(-0.1, max_entropy-0.3)
    ax2.legend(loc='upper right', fontsize=fontsize-1.5, frameon=False)
    
    # Style both axes
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color('lightgray')
        ax.tick_params(axis='both', which='both', color='lightgray', labelcolor='black')
        ax.grid(True, which='both', ls='--', lw=0.3)
    
    # Tight layout and save
    fig.tight_layout()
    full_path = os.path.join(save_dir, save_path)
    plt.savefig(full_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved plot to: {full_path}")

    plt.show()
    plt.close(fig)

# Main execution
print("Loading position metrics for multiple compute budgets...")

# Load data for each compute budget
compute_data = {}
for budget, path in compute_budgets.items():
    print(f"\nLoading {budget}:")
    compute_data[budget] = load_position_metrics(path)

# Filter out None values
compute_data = {k: v for k, v in compute_data.items() if v is not None}

if compute_data:
    print(f"\nSuccessfully loaded {len(compute_data)} compute budgets")

    # Print summary for each budget
    for budget, df in compute_data.items():
        if df is not None:
            print(f"\n{budget_labels[budget]}:")
            print(f"  Total positions: {len(df):,}")
            if 'correct' in df.columns:
                print(f"  Accuracy: {df['correct'].mean():.3f}")

    # Create the multi-budget plot
    plot_multi_compute_budgets(compute_data)

else:
    print("No data could be loaded. Please check the paths in compute_budgets.")