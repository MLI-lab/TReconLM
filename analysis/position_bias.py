import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    from .helpers import get_edit_operations, extract_context
except ImportError:
    from helpers import get_edit_operations, extract_context

# LaTeX style settings
fontsize = 7.7
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{type1cm}',
    'font.size': fontsize,
})


def analyze_positional_errors(filepath):
    """
    Analyze where in sequences errors tend to occur.
    Returns None if file doesn't exist.
    """
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return None

    # Track errors by relative position
    position_bins = {'start': 0, 'middle': 0, 'end': 0}
    position_counts = {'start': 0, 'middle': 0, 'end': 0}

    # Track by error type and position
    error_types = ['sub', 'del', 'ins']
    position_bins_by_type = {error_type: {'start': 0, 'middle': 0, 'end': 0} for error_type in error_types}

    # Track by absolute position
    absolute_position_errors = defaultdict(int)
    absolute_position_counts = defaultdict(int)

    # Track by fine-grained relative position (10% bins)
    relative_bins = 10
    relative_position_errors = [0] * relative_bins
    relative_position_counts = [0] * relative_bins

    # Track by error type and fine-grained position
    relative_position_errors_by_type = {error_type: [0] * relative_bins for error_type in error_types}

    sequence_lengths = []
    total_sequences = 0

    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gt = row["ground_truth"]
            pred = row["prediction"]
            seq_len = len(gt)
            sequence_lengths.append(seq_len)
            total_sequences += 1
            
            # Get edit operations
            operations = get_edit_operations(gt, pred)
            
            # Count all positions
            for i in range(seq_len):
                relative_pos = i / seq_len
                
                # Coarse bins (start/middle/end)
                if relative_pos < 0.2:
                    position_counts['start'] += 1
                elif relative_pos > 0.8:
                    position_counts['end'] += 1
                else:
                    position_counts['middle'] += 1
                
                # Fine bins (10% each)
                bin_idx = min(int(relative_pos * relative_bins), relative_bins - 1)
                relative_position_counts[bin_idx] += 1
                
                # Absolute position
                absolute_position_counts[i] += 1
            
            # Count error positions
            for op in operations:
                op_type = op[0]  # 'sub', 'del', 'ins'
                pos = op[1]
                if pos >= seq_len:  # Handle insertions at the end
                    pos = seq_len - 1

                relative_pos = pos / seq_len

                # Coarse bins
                if relative_pos < 0.2:
                    position_bins['start'] += 1
                    position_bins_by_type[op_type]['start'] += 1
                elif relative_pos > 0.8:
                    position_bins['end'] += 1
                    position_bins_by_type[op_type]['end'] += 1
                else:
                    position_bins['middle'] += 1
                    position_bins_by_type[op_type]['middle'] += 1

                # Fine bins
                bin_idx = min(int(relative_pos * relative_bins), relative_bins - 1)
                relative_position_errors[bin_idx] += 1
                relative_position_errors_by_type[op_type][bin_idx] += 1

                # Absolute position
                absolute_position_errors[pos] += 1
    
    # Calculate statistics
    total_errors = sum(position_bins.values())
    total_positions = sum(position_counts.values())
    overall_error_rate = total_errors / total_positions if total_positions > 0 else 0
    
    # Print results
    print("="*70)
    print("Positional Error Analysis")
    print("="*70)

    print(f"\nDataset statistics:")
    print(f"  Total sequences: {total_sequences:,}")
    print(f"  Mean sequence length: {np.mean(sequence_lengths):.1f}")
    print(f"  Std sequence length: {np.std(sequence_lengths):.1f}")
    print(f"  Min/Max length: {min(sequence_lengths)}/{max(sequence_lengths)}")

    print(f"\nOverall error statistics:")
    print(f"  Total positions: {total_positions:,}")
    print(f"  Total errors: {total_errors:,}")
    print(f"  Overall error rate: {overall_error_rate:.3%}")

    # Find absolute positions with very low error rates
    print(f"\nAbsolute positions with low error rates (enrichment < 0.5):")
    for pos, count in absolute_position_counts.items():
        if count > 0:
            rate = absolute_position_errors[pos] / count
            enrichment = rate / overall_error_rate if overall_error_rate > 0 else 0

            # Report positions with enrichment < 0.5 (less than half the average error rate)
            if enrichment < 0.5:
                print(f"  Position {pos}: enrichment = {enrichment:.3f}, error rate = {rate:.3%}, count = {count}")
    
    print(f"\nError Distribution by Region:")
    for region in ['start', 'middle', 'end']:
        if position_counts[region] > 0:
            error_rate = position_bins[region] / position_counts[region]
            enrichment = error_rate / overall_error_rate if overall_error_rate > 0 else 0
            percentage = position_bins[region] / total_errors * 100 if total_errors > 0 else 0
            
            region_label = {
                'start': 'Start (0-20%)',
                'middle': 'Middle (20-80%)',
                'end': 'End (80-100%)'
            }[region]
            
            print(f"\n  {region_label}:")
            print(f"    Errors: {position_bins[region]:,} ({percentage:.1f}% of all errors)")
            print(f"    Positions: {position_counts[region]:,}")
            print(f"    Error rate: {error_rate:.3%}")
            print(f"    Enrichment: {enrichment:.2f}x")
    
    # Fine-grained analysis (10% bins)
    print(f"\nError rate by 10% bins:")
    print("  (% of positions in each bin that have errors)")
    print("  Bin:        ", end="")
    for i in range(relative_bins):
        print(f"{i*10}-{(i+1)*10}%".center(8), end="")
    print("\n  Error rate: ", end="")
    for i in range(relative_bins):
        if relative_position_counts[i] > 0:
            rate = relative_position_errors[i] / relative_position_counts[i] * 100
            print(f"{rate:7.2f}%", end="")
        else:
            print("      - ", end="")
    print("\n  Enrichment: ", end="")
    for i in range(relative_bins):
        if relative_position_counts[i] > 0:
            rate = relative_position_errors[i] / relative_position_counts[i]
            enrichment = rate / overall_error_rate if overall_error_rate > 0 else 0
            print(f"{enrichment:7.2f}x", end="")
        else:
            print("      - ", end="")
    print()

    # Error distribution by type and position
    print(f"\nError distribution by type and region:")
    for error_type in error_types:
        total_type_errors = sum(position_bins_by_type[error_type].values())
        if total_type_errors > 0:
            error_type_name = {'sub': 'Substitutions', 'del': 'Deletions', 'ins': 'Insertions'}[error_type]
            print(f"\n{error_type_name} ({total_type_errors:,} total):")

            for region in ['start', 'middle', 'end']:
                if position_counts[region] > 0:
                    type_errors = position_bins_by_type[error_type][region]
                    error_rate = type_errors / position_counts[region]
                    type_overall_rate = total_type_errors / total_positions if total_positions > 0 else 0
                    enrichment = error_rate / type_overall_rate if type_overall_rate > 0 else 0
                    percentage = type_errors / total_type_errors * 100 if total_type_errors > 0 else 0

                    region_label = {
                        'start': 'Start',
                        'middle': 'Middle',
                        'end': 'End'
                    }[region]

                    print(f"  {region_label}: {type_errors:,} ({percentage:.1f}%) - Rate: {error_rate:.3%} - Enrichment: {enrichment:.2f}x")

    # Fine-grained analysis by error type
    print(f"\nError rate by type and 10% bins:")
    for error_type in error_types:
        total_type_errors = sum(relative_position_errors_by_type[error_type])
        if total_type_errors > 0:
            error_type_name = {'sub': 'Sub', 'del': 'Del', 'ins': 'Ins'}[error_type]
            print(f"\n{error_type_name}:")
            print("  Bin:        ", end="")
            for i in range(relative_bins):
                print(f"{i*10}-{(i+1)*10}%".center(8), end="")
            print("\n  Error rate: ", end="")
            for i in range(relative_bins):
                if relative_position_counts[i] > 0:
                    rate = relative_position_errors_by_type[error_type][i] / relative_position_counts[i] * 100
                    print(f"{rate:7.2f}%", end="")
                else:
                    print("      - ", end="")
            print("\n  Enrichment: ", end="")
            type_overall_rate = total_type_errors / total_positions if total_positions > 0 else 0
            for i in range(relative_bins):
                if relative_position_counts[i] > 0:
                    rate = relative_position_errors_by_type[error_type][i] / relative_position_counts[i]
                    enrichment = rate / type_overall_rate if type_overall_rate > 0 else 0
                    print(f"{enrichment:7.2f}x", end="")
                else:
                    print("      - ", end="")
            print()

    # Plot for most common length
    most_common_length = int(np.median(sequence_lengths))
    print(f"\nError rate by absolute position (for median length ~{most_common_length}):")
    
    # Sample positions to display
    positions_to_plot = list(range(0, min(most_common_length, 20))) + \
                       list(range(max(20, most_common_length - 10), most_common_length))
    
    if len(positions_to_plot) > 30:  # If too many, sample them
        positions_to_plot = list(range(0, 10)) + \
                           list(range(most_common_length // 2 - 5, most_common_length // 2 + 5)) + \
                           list(range(most_common_length - 10, most_common_length))
    
    print("  Pos: ", end="")
    for pos in positions_to_plot[:15]:
        print(f"{pos:3d}", end=" ")
    if len(positions_to_plot) > 15:
        print("...", end="")
        for pos in positions_to_plot[-5:]:
            print(f"{pos:3d}", end=" ")
    
    print("\n  Err%:", end="")
    for pos in positions_to_plot[:15]:
        if absolute_position_counts[pos] > 0:
            rate = absolute_position_errors[pos] / absolute_position_counts[pos] * 100
            print(f"{rate:3.0f}", end=" ")
        else:
            print("  -", end=" ")
    if len(positions_to_plot) > 15:
        print("...", end="")
        for pos in positions_to_plot[-5:]:
            if absolute_position_counts[pos] > 0:
                rate = absolute_position_errors[pos] / absolute_position_counts[pos] * 100
                print(f"{rate:3.0f}", end=" ")
            else:
                print("  -", end=" ")
    print()
    
    return {
        'position_bins': position_bins,
        'position_counts': position_counts,
        'relative_errors': relative_position_errors,
        'relative_counts': relative_position_counts,
        'overall_error_rate': overall_error_rate,
        'relative_position_errors_by_type': relative_position_errors_by_type,
        'error_types': error_types
    }


def plot_positional_bias(test_files, output_path='./plots/positional_bias.pdf'):
    """
    Create a plot showing positional bias for multiple datasets.

    Args:
        test_files: dict mapping dataset names to file paths
        output_path: where to save the plot
    """
    # Analyze all datasets and collect plotting data
    all_enrichments = {}
    all_enrichments_by_type = {}

    for dataset_name, filepath in test_files.items():
        print(f"\nAnalyzing {dataset_name} for plotting...")
        results = analyze_positional_errors(filepath)

        # Skip if file doesn't exist
        if results is None:
            print(f"  Skipping {dataset_name} (file not found)")
            continue

        # Calculate enrichment factors
        relative_bins = len(results['relative_errors'])
        enrichment_factors = []

        for i in range(relative_bins):
            if results['relative_counts'][i] > 0:
                rate = results['relative_errors'][i] / results['relative_counts'][i]
                enrichment = rate / results['overall_error_rate'] if results['overall_error_rate'] > 0 else 0
                enrichment_factors.append(enrichment)
            else:
                enrichment_factors.append(0)

        all_enrichments[dataset_name] = enrichment_factors

        # Calculate enrichment by error type for ALL datasets
        enrichment_by_type = {}
        total_positions = sum(results['relative_counts'])

        for error_type in results['error_types']:
            total_type_errors = sum(results['relative_position_errors_by_type'][error_type])
            type_overall_rate = total_type_errors / total_positions if total_positions > 0 else 0

            enrichments = []
            for i in range(relative_bins):
                if results['relative_counts'][i] > 0:
                    rate = results['relative_position_errors_by_type'][error_type][i] / results['relative_counts'][i]
                    enrichment = rate / type_overall_rate if type_overall_rate > 0 else 0
                    enrichments.append(enrichment)
                else:
                    enrichments.append(0)

            enrichment_by_type[error_type] = enrichments

        all_enrichments_by_type[dataset_name] = enrichment_by_type

    # Create figure with side-by-side subplots (matching plots.ipynb style)
    fig, axs = plt.subplots(1, 2, figsize=(6, 1.3), dpi=300, gridspec_kw={'wspace': 0.4})

    ax1 = axs[0]
    ax2 = axs[1]

    # Define colors for datasets
    dataset_colors = {
        'synthetic': '#6699CC',
        'microsoft': '#9D9AC5',
        'noisy_dna': '#F287BD'
    }

    # Define color shades for error types (lighter shades of dataset colors)
    error_type_colors = {
        'synthetic': {
            'sub': '#6699CC',   # Base color
            'del': '#8AB3DD',   # Lighter shade
            'ins': '#B3D1EE'    # Even lighter shade
        },
        'microsoft': {
            'sub': '#9D9AC5',   # Base color
            'del': '#B5B0D6',   # Lighter shade
            'ins': '#CEC6E7'    # Even lighter shade
        },
        'noisy_dna': {
            'sub': '#F287BD',   # Base color
            'del': '#F5A5D0',   # Lighter shade
            'ins': '#F8C3E3'    # Even lighter shade
        }
    }

    # Dataset display names with abbreviations
    dataset_display_names = {
        'synthetic': 'Synthetic (S)',
        'microsoft': 'Microsoft (M)',
        'noisy_dna': 'Noisy DNA (N)'
    }
    
    # Dataset abbreviations for second plot
    dataset_abbrev = {
        'synthetic': 'S',
        'microsoft': 'M',
        'noisy_dna': 'N'
    }

    # Bin centers for x-axis (0-10%, 10-20%, etc.)
    bin_centers = np.arange(5, 100, 10)  # 5, 15, 25, ..., 95

    # Plot 1: Overall enrichment factor
    for dataset_name, enrichments in all_enrichments.items():
        color = dataset_colors.get(dataset_name, '#AAAAAA')
        display_name = dataset_display_names.get(dataset_name, dataset_name.replace('_', ' ').title())
        ax1.plot(bin_centers, enrichments,
                marker='o', linewidth=0.5, markersize=2,
                color=color,
                label=display_name)

    ax1.axhline(y=1.0, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Relative position in sequence (\\%)', color='black')
    ax1.set_ylabel('Error enrichment', color='black')
    ax1.legend(loc='best', frameon=False, fontsize=fontsize-2)
    ax1.set_xlim(0, 100)

    # Style axis spines and ticks (lightgray box, black labels)
    for spine in ax1.spines.values():
        spine.set_color('lightgray')
    ax1.tick_params(axis='both', which='both', color='lightgray', labelcolor='black')

    # Plot 2: Error rates by type for ALL datasets
    error_type_labels = {'sub': 'Sub.', 'del': 'Del.', 'ins': 'Ins.'}
    markers = {'sub': 'o', 'del': 's', 'ins': '^'}

    # Plot in order: error type first, then datasets
    for error_type in ['sub', 'del', 'ins']:
        for dataset_name, enrichment_by_type in all_enrichments_by_type.items():
            if error_type in enrichment_by_type:
                colors = error_type_colors.get(dataset_name, {})
                color = colors.get(error_type, '#AAAAAA')

                # Label format: error_type (abbreviation)
                abbrev = dataset_abbrev.get(dataset_name, dataset_name[0].upper())
                label = f"{error_type_labels[error_type]}({abbrev})"

                ax2.plot(bin_centers, enrichment_by_type[error_type],
                        marker=markers[error_type], linewidth=0.5, markersize=2,
                        color=color,
                        label=label)

    ax2.axhline(y=1.0, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('Relative position in sequence (\\%)', color='black')
    ax2.set_ylabel('Error enrichment', color='black')

    # Create legend with 3 columns (one per dataset) and 3 rows (one per error type)
    ax2.legend(loc='upper center', frameon=False, fontsize=fontsize-2, ncol=3, handletextpad=0.1, columnspacing=0.4, handlelength=1.0,
              bbox_to_anchor=(0.5, 0.98))
    ax2.set_xlim(0, 100)

    # Style axis spines and ticks (lightgray box, black labels)
    for spine in ax2.spines.values():
        spine.set_color('lightgray')
    ax2.tick_params(axis='both', which='both', color='lightgray', labelcolor='black')

    plt.tight_layout(pad=1.5)

    # Save figure
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\n{'='*70}")
    print(f"Saved plot to: {output_path}")
    print(f"{'='*70}")

    plt.show()


# Run the analysis
if __name__ == "__main__":
    # Set your prediction output file paths
    test_files = {
        'synthetic': '',  # path to synthetic predictions_output.tsv
        'microsoft': '',  # path to microsoft predictions_output.tsv
        'noisy_dna': ''   # path to noisy_dna predictions_output.tsv
    }

    # Run text analysis
    for dataset_name, filepath in test_files.items():
        print(f"\n{'#'*70}")
        print(f"# Analyzing {dataset_name} dataset")
        print(f"{'#'*70}")
        results = analyze_positional_errors(filepath)
        if results is None:
            print(f"Skipping {dataset_name} (file not found)")
            continue

    # Create plot
    print(f"\n{'='*70}")
    print("Creating positional bias plot")
    print(f"{'='*70}")
    plot_positional_bias(test_files)