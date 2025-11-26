"""
Attention Visualization Script
"""

import os
import glob
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import Levenshtein

# 
# CONFIGURATION - CUSTOMIZE THESE SETTINGS
# 

# Main configuration
# For comparison mode, set ATTENTION_DIR_1, ATTENTION_DIR_2, and optionally ATTENTION_DIR_3
# For single mode, set only ATTENTION_DIR_1 and leave others as None
ATTENTION_DIR_1 = "<your.data.path>/TReconLM/Contaminated_Attention_output_cz3/clean/"  # path to first attention directory
ATTENTION_DIR_2 = "<your.data.path>/TReconLM/Contaminated_Attention_output_cz3/untrained_misclustering/cont_0.020/"  # path to second attention directory (None for single mode)
ATTENTION_DIR_3 = "<your.data.path>/TReconLM/Contaminated_Attention_output_cz3/trained_misclustering/cont_0.020/"  # path to third attention directory (None if only 2 needed)
CLUSTER_SIZE = 3                 # Specific cluster size(s): int, (9,10), [9,10], or None for all
OUTPUT_DIR = "./plots"       # Where to save plots
NUM_SAMPLES = 2                            # Random samples per cluster (None = all)
SAVE_PLOTS = True                          # Whether to save plots as PDFs
SHOW_PLOTS = False                         # Whether to display plots inline
FIGSIZE = (12, 1.25)                          # Figure size for comparison plots (width scales with number of subplots)
SUBPLOT_SPACING = 0.3                  # Horizontal space between subplots (lower = closer together)
RANDOM_SEED = 1                           # For reproducible random sampling
# File type modes for each directory (set to None to auto-detect based on directory contents)
MISCLUSTERED_MODE_1 = False               # Directory 1: False for .pt files, True for .npz files
MISCLUSTERED_MODE_2 = True                # Directory 2: False for .pt files, True for .npz files
MISCLUSTERED_MODE_3 = True                # Directory 3: False for .pt files, True for .npz files

# Style configuration
ATTENTION_FONTSIZE = 17
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': ATTENTION_FONTSIZE,
    'figure.dpi': 300,
    'text.usetex': True,
})

# Colormap configuration
def truncate_colormap(cmap, minval=0.1, maxval=1.0, n=100):
    """Truncate colormap to skip lightest colors"""
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval},{maxval})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

# Truncate PuBu to skip lightest colors (0.0–0.2)
trunc_cmap = truncate_colormap(cm.PuBu, minval=0.05, maxval=1.0)

# 
# CORE FUNCTIONS
# 

def find_attention_files(attention_dir, cluster_size=None, misclustered_mode=False):
    """
    Find attention files, optionally filtered by cluster size.

    Args:
        attention_dir: Base directory containing attention files
        cluster_size: Optional cluster size filter (int, list/tuple of ints, or None)
        misclustered_mode: If True, search for contaminated attention files (.npz)

    Returns:
        Dict mapping cluster_size to list of file paths
    """
    files_by_cluster = {}

    if not os.path.exists(attention_dir):
        print(f"Attention directory not found: {attention_dir}")
        return files_by_cluster

    # Look for cluster subdirectories
    cluster_dirs = glob.glob(os.path.join(attention_dir, "cluster_size_*"))

    for cluster_dir in cluster_dirs:
        # Extract cluster size from directory name
        cluster_num = int(os.path.basename(cluster_dir).split('_')[-1])

        # Skip if filtering by cluster size
        if cluster_size is not None:
            # Handle multiple cluster sizes (list/tuple) or single cluster size (int)
            if isinstance(cluster_size, (list, tuple)):
                if cluster_num not in cluster_size:
                    continue
            else:
                if cluster_num != cluster_size:
                    continue

        # Find attention files in this cluster directory
        if misclustered_mode:
            # Look for contaminated attention files (.npz)
            files = glob.glob(os.path.join(cluster_dir, "contaminated_*.npz"))
        else:
            # Look for regular attention files (.pt)
            files = glob.glob(os.path.join(cluster_dir, "attention_sample_*.pt"))

        if files:
            files_by_cluster[cluster_num] = sorted(files)

    return files_by_cluster

def load_attention_data(filepath):
    """
    Load attention data from saved file.

    Supports both .pt (regular attention) and .npz (contaminated attention) files.
    """
    try:
        if filepath.endswith('.npz'):
            # Load contaminated attention data
            npz_data = np.load(filepath, allow_pickle=True)

            # Extract contamination annotation
            contamination_annotation = npz_data.get('contamination_annotation', None)
            if contamination_annotation is not None:
                contamination_annotation = contamination_annotation.item()  # Convert from numpy array to dict

            # Convert to format compatible with plot_attention_matrix
            attention_matrix = npz_data['attention_matrix']  # [num_generated_tokens, seq_len]
            read_boundaries = npz_data['read_boundaries'].tolist()

            # Reconstruct token sequence from read sequences if available
            if 'token_sequence' in npz_data:
                token_sequence = npz_data['token_sequence'].item()
            elif 'input_sequence' in npz_data:
                token_sequence = npz_data['input_sequence'].item()
            else:
                token_sequence = ""

            # Convert attention matrix to tensor format expected by plotting function
            # The plotting function expects normalized attention in shape [num_tokens, seq_len]
            normalized_attention = torch.tensor(attention_matrix, dtype=torch.float32)

            data = {
                'normalized_attention': normalized_attention,
                'token_sequence': token_sequence,
                'read_boundaries': read_boundaries,
                'cluster_size': npz_data.get('cluster_size', None),
                'example_idx': npz_data.get('example_idx', None),
                'ground_truth': npz_data.get('ground_truth', '').item() if 'ground_truth' in npz_data else None,
                'prediction': npz_data.get('prediction', '').item() if 'prediction' in npz_data else None,
                'contaminated_positions': contamination_annotation['contaminated_read_indices'] if contamination_annotation else None,
                'contamination_info': contamination_annotation
            }

            return data
        else:
            # Load regular .pt attention data
            data = torch.load(filepath, map_location='cpu')
            # Add contaminated_positions field if not present
            if 'contaminated_positions' not in data:
                data['contaminated_positions'] = None
            return data

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def plot_attention_matrix(data, output_path=None, show_plot=True, figsize=(6.3, 1.3), ax=None, subplot_mode=False, subplot_idx=0):
    """
    Plot attention matrix visualization.

    Args:
        data: Dictionary containing attention data
        output_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        figsize: Figure size tuple
        ax: Matplotlib axis to plot on (for subplot mode)
        subplot_mode: Whether we're in subplot mode (affects figure creation)
        subplot_idx: Index of subplot (0-based) - used to show ylabel only on first plot
    """
    attn = data['normalized_attention'].numpy()
    token_sequence = data['token_sequence']
    cluster_size = data.get('cluster_size', 'unknown')
    example_idx = data.get('example_idx', 'unknown')

    # Calculate normalized Levenshtein distance if we have GT and prediction
    normalized_lev = None
    if 'ground_truth' in data and 'prediction' in data:
        gt = data['ground_truth']
        pred = data['prediction']
        if gt and pred:
            # Use Levenshtein package to compute distance
            lev_dist = Levenshtein.distance(pred, gt)
            # Normalize by max length
            max_len = max(len(gt), len(pred))
            if max_len > 0:
                normalized_lev = lev_dist / max_len

    # Remove all padding tokens (#) from the sequence
    if ':' in token_sequence:
        # Get the input part (before colon)
        input_part = token_sequence.split(':')[0]

        # Create mapping from original positions to cleaned positions
        clean_sequence = ""
        original_to_clean_map = []  # Maps original position -> clean position
        clean_pos = 0

        for i, char in enumerate(input_part):
            if char != '#':  # Keep everything except padding tokens
                clean_sequence += char
                original_to_clean_map.append(clean_pos)
                clean_pos += 1
            else:
                original_to_clean_map.append(-1)  # Mark padding positions

        # Debug output (commented out - uncomment if needed)
        # print(f"\n=== DEBUG ===")
        # print(f"Original input: {input_part[:100]}...")
        # print(f"Cleaned input: {clean_sequence[:100]}...")
        # print(f"Original length: {len(input_part)}, Cleaned length: {len(clean_sequence)}")

        # Now we need to slice the attention matrix to only include non-padding positions
        # Filter attention matrix to remove padding columns
        non_padding_indices = [i for i, pos in enumerate(original_to_clean_map) if pos != -1]
        attn = attn[:, non_padding_indices]  # Keep only non-padding columns

        # Calculate read boundaries in the CLEANED sequence
        separator_positions = [i for i, char in enumerate(clean_sequence) if char == '|']

        read_boundaries = []
        start = 0

        for sep_pos in separator_positions:
            if sep_pos > start:
                read_boundaries.append((start, sep_pos - 1))
                start = sep_pos + 1

        # Add the last read (from last separator to end)
        if start < len(clean_sequence):
            read_boundaries.append((start, len(clean_sequence) - 1))

        # Extract read end positions for red lines (positions just before |)
        read_ends = [pos - 1 for pos in separator_positions]

        # Debug output (commented out - uncomment if needed)
        # print(f"Clean sequence separators at: {separator_positions}")
        # print(f"Read boundaries (clean): {read_boundaries}")
        # print(f"Red line positions: {read_ends}")
    else:
        # Fallback if no colon found
        read_ends = data.get('read_ends', [])
        read_boundaries = [(0, len(token_sequence)-1)]

    # Create the plot or use provided axis
    if not subplot_mode:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    # Apply min-max normalization for better contrast
    attn_min = attn.min()
    attn_max = attn.max()
    if attn_max > attn_min:
        attn_normalized = (attn - attn_min) / (attn_max - attn_min)
    else:
        attn_normalized = attn  # Avoid division by zero if all values are the same

    # Plot attention heatmap with truncated colormap
    im = ax.imshow(attn_normalized, cmap=trunc_cmap, aspect='auto', interpolation='nearest')

    # Create colorbar with custom styling (only in single mode)
    if not subplot_mode:
        cbar = plt.colorbar(im, label='Attention score', ax=ax)
        cbar.outline.set_edgecolor('lightgrey')  # Grey frame around colorbar

        # Set colorbar ticks: 0.0, 0.5, 1.0
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(['0.0', '0.5', '1.0'])
        cbar.ax.tick_params(color='lightgrey')  # Grey tick marks for colorbar (not labels)

    # Draw red dashed vertical lines at separator positions (between reads)
    # The lines should be drawn between the last character of a read and the separator
    for i, end_pos in enumerate(read_ends):
        # Draw line between the end of the read and the separator
        ax.axvline(x=end_pos + 0.5, color='red', linestyle='--', linewidth=0.7)

    # Axis labels with bold math formatting (black labels)
    ax.set_xlabel('Prompt $\\mathbf{p}$', color='black')
    # Only show ylabel on first subplot
    if subplot_idx == 0:
        ax.set_ylabel('Predicted $\\mathbf{\\hat{x}}$', color='black')

    # Style the plot - set all spines to lightgrey
    for spine in ax.spines.values():
        spine.set_edgecolor('lightgrey')
    ax.tick_params(color='lightgrey', which='both')  # Light grey tick marks only (not labels)

    # Set y-axis ticks: 0, middle, and max length
    y_max = attn_normalized.shape[0]
    if y_max > 2:
        y_middle = y_max // 2
        ax.set_yticks([0, y_middle, y_max-1])
        ax.set_yticklabels(['0', str(y_middle), str(y_max)])
    else:
        # For very short sequences, just use start and end
        ax.set_yticks([0, y_max-1])
        ax.set_yticklabels(['0', str(y_max)])

    # Custom x-axis ticks: at 0 and at read boundaries only
    tick_positions = []
    tick_labels = []

    if read_boundaries:
        # Add tick at position 0 (beginning)
        tick_positions.append(0)
        tick_labels.append('0')

        # Add ticks at red line positions (end of each read, except the last one)
        for end in read_ends:
            tick_positions.append(end + 0.5)
            tick_labels.append(f'{end + 1}')

        # Add tick at the very end of the sequence
        if read_boundaries:
            last_end = read_boundaries[-1][1]  # End of last read
            tick_positions.append(last_end)
            tick_labels.append(f'{last_end + 1}')

        # Sort ticks by position to ensure proper order
        sorted_ticks = sorted(zip(tick_positions, tick_labels))
        tick_positions = [pos for pos, _ in sorted_ticks]
        tick_labels = [label for _, label in sorted_ticks]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

    # Add read labels above the plot (y₁, y₂, y₃, etc.)
    # Contaminated reads (misclustered) are shown in red
    if read_boundaries:
        # Get the current axis limits to position labels correctly above the plot
        ylim = ax.get_ylim()
        y_pos = ylim[1] + (ylim[1] - ylim[0]) * 0.05  # Position above the plot

        # Get contaminated positions from data
        contaminated_positions = data.get('contaminated_positions', None)

        # Use calculated read boundaries for perfect alignment
        for i, (start, end) in enumerate(read_boundaries):
            mid_pos = (start + end) / 2  # True center of each read

            # Check if this read is contaminated (misclustered)
            is_contaminated = contaminated_positions is not None and i in contaminated_positions
            label_color = 'red' if is_contaminated else 'black'

            ax.text(mid_pos, y_pos, f'$y_{{{i+1}}}$',
                    ha='center', va='bottom', fontweight='bold', fontsize=ATTENTION_FONTSIZE, color=label_color)

    # Add normalized Levenshtein distance above the plot
    if normalized_lev is not None:
        # Place it in the title position but as text
        ax.set_title(f'$d_L$={normalized_lev:.3f}',
                     fontsize=ATTENTION_FONTSIZE-1, fontweight='bold', pad=25)
    else:
        # Add a note that Levenshtein is not available
        ax.set_title('(Levenshtein distance not available)',
                     fontsize=ATTENTION_FONTSIZE-2, fontweight='normal', pad=25, color='gray')

    # Only handle figure-level operations in single mode
    if not subplot_mode:
        plt.tight_layout()

        # Save plot if requested
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"    Saved: {output_path}")

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

def plot_comparison(data_list, output_path=None, show_plot=True, figsize=(12, 4)):
    """
    Plot multiple attention matrices side by side for comparison.

    Args:
        data_list: List of attention data dictionaries (2 or 3 items)
        output_path: Path to save the comparison plot
        show_plot: Whether to display the plot
        figsize: Figure size for the combined plot
    """
    num_plots = len(data_list)

    # Create figure with N subplots, with space for colorbar
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)

    # Ensure axes is always a list even if num_plots=1
    if num_plots == 1:
        axes = [axes]

    # Plot each attention matrix
    for idx, (ax, data) in enumerate(zip(axes, data_list)):
        plot_attention_matrix(data, ax=ax, subplot_mode=True, subplot_idx=idx)

    # Get the image from the last axis for colorbar
    im = axes[-1].images[0] if axes[-1].images else axes[0].images[0]

    # Adjust subplot spacing - reduce horizontal space between plots
    plt.subplots_adjust(wspace=SUBPLOT_SPACING)  # Reduce horizontal spacing (default is ~0.2)

    # Get the position of the last subplot to align colorbar
    pos_last = axes[-1].get_position()

    # Add a shared colorbar on the right, matching the height of the plots
    cbar_width =  0.009  # Width of colorbar
    cbar_padding = 0.02  # Space between plot and colorbar
    cbar_ax = fig.add_axes([pos_last.x1 + cbar_padding, pos_last.y0, cbar_width, pos_last.height])

    cbar = fig.colorbar(im, cax=cbar_ax, label='Attention score')
    cbar.outline.set_edgecolor('lightgrey')  # Grey frame around colorbar

    # Set colorbar ticks: 0.0, 0.5, 1.0
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(['0.0', '0.5', '1.0'])
    cbar.ax.tick_params(color='lightgrey')  # Grey tick marks for colorbar (not labels)

    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison: {output_path}")

    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def main():
    """Main visualization function - does everything in one place!"""

    print("Complete attention visualization")
    print("=" * 60)

    # Set random seed for reproducible results
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Check if we're in comparison mode
    comparison_mode = ATTENTION_DIR_2 is not None
    num_dirs = sum([ATTENTION_DIR_1 is not None, ATTENTION_DIR_2 is not None, ATTENTION_DIR_3 is not None])

    # Print configuration
    print("Configuration:")
    if comparison_mode:
        print(f"  Attention directory 1: {ATTENTION_DIR_1}")
        print(f"  Attention directory 2: {ATTENTION_DIR_2}")
        if ATTENTION_DIR_3:
            print(f"  Attention directory 3: {ATTENTION_DIR_3}")
        print(f"  Mode: comparison ({num_dirs} directories)")
    else:
        print(f"  Attention directory: {ATTENTION_DIR_1}")
        print(f"  Mode: single")

    cluster_filter_text = 'All' if CLUSTER_SIZE is None else (
        f"{CLUSTER_SIZE}" if isinstance(CLUSTER_SIZE, (list, tuple)) else str(CLUSTER_SIZE)
    )
    print(f"  Cluster size filter: {cluster_filter_text}")
    print(f"  Misclustered modes: Dir1={MISCLUSTERED_MODE_1}, Dir2={MISCLUSTERED_MODE_2}, Dir3={MISCLUSTERED_MODE_3}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Samples per cluster: {NUM_SAMPLES if NUM_SAMPLES else 'All'}")
    print(f"  Save plots: {SAVE_PLOTS}")
    print(f"  Show plots: {SHOW_PLOTS}")
    print(f"  Figure size: {FIGSIZE}")
    print()

    # Create output directory if saving
    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Created output directory: {OUTPUT_DIR}")

    # Find attention files for all directories
    print("Searching for attention files...")
    files_by_cluster_1 = find_attention_files(ATTENTION_DIR_1, CLUSTER_SIZE, MISCLUSTERED_MODE_1)

    # Find attention files for second and third directories if in comparison mode
    files_by_cluster_2 = None
    files_by_cluster_3 = None
    if comparison_mode:
        files_by_cluster_2 = find_attention_files(ATTENTION_DIR_2, CLUSTER_SIZE, MISCLUSTERED_MODE_2)
        if ATTENTION_DIR_3:
            files_by_cluster_3 = find_attention_files(ATTENTION_DIR_3, CLUSTER_SIZE, MISCLUSTERED_MODE_3)

    if not files_by_cluster_1:
        print(f"No attention files found in {ATTENTION_DIR_1}")
        if CLUSTER_SIZE:
            print(f"   (searched for cluster size {CLUSTER_SIZE})")
        return

    if comparison_mode and not files_by_cluster_2:
        print(f"No attention files found in {ATTENTION_DIR_2}")
        return

    # Show found files
    print("Found attention files:")
    if comparison_mode:
        print("  Directory 1:")
        total_files_1 = 0
        for cluster_num in sorted(files_by_cluster_1.keys()):
            num_files = len(files_by_cluster_1[cluster_num])
            total_files_1 += num_files
            print(f"    Cluster size {cluster_num}: {num_files} files")
        print(f"    Total: {total_files_1} files")

        print("  Directory 2:")
        total_files_2 = 0
        for cluster_num in sorted(files_by_cluster_2.keys()): # keys isfilenames
            num_files = len(files_by_cluster_2[cluster_num])
            total_files_2 += num_files
            print(f"    Cluster size {cluster_num}: {num_files} files")
        print(f"    Total: {total_files_2} files")

        # Find common cluster sizes
        common_clusters = set(files_by_cluster_1.keys()) & set(files_by_cluster_2.keys())
        print(f"\n  Common cluster sizes: {sorted(common_clusters)}")
    else:
        total_files = 0
        for cluster_num in sorted(files_by_cluster_1.keys()):
            num_files = len(files_by_cluster_1[cluster_num])
            total_files += num_files
            print(f"   Cluster size {cluster_num}: {num_files} files")
        print(f"   Total: {total_files} files")

    # Sample files if requested - use same example indices across directories
    if NUM_SAMPLES:
        print(f"\nSampling {NUM_SAMPLES} files per cluster size...")
        clusters_to_process = common_clusters if comparison_mode else files_by_cluster_1.keys()

        # Define pattern extraction function (same as used in plotting)
        def get_base_pattern(filepath):
            """Extract example index from filename for matching across different file types"""
            filename = os.path.basename(filepath)
            if filename.startswith('contaminated_ex'):
                example_idx = filename.split('_')[1][2:]  # Remove 'ex' prefix
                return f"ex{example_idx}"
            elif filename.startswith('attention_sample_'):
                example_idx = os.path.splitext(filename)[0].split('_')[-1]
                return f"ex{example_idx}"
            else:
                return os.path.splitext(filename)[0]

        for cluster_num in clusters_to_process:
            files_1 = sorted(files_by_cluster_1[cluster_num])  # Sort for consistency

            if comparison_mode:
                files_2 = sorted(files_by_cluster_2[cluster_num])
                files_3 = sorted(files_by_cluster_3[cluster_num]) if files_by_cluster_3 and cluster_num in files_by_cluster_3 else None

                # Map pattern -> filepath for each directory
                patterns_1 = {get_base_pattern(f): f for f in files_1}
                patterns_2 = {get_base_pattern(f): f for f in files_2}

                # Find common patterns across all directories
                if files_3:
                    patterns_3 = {get_base_pattern(f): f for f in files_3}
                    common_patterns = set(patterns_1.keys()) & set(patterns_2.keys()) & set(patterns_3.keys())
                else:
                    common_patterns = set(patterns_1.keys()) & set(patterns_2.keys())

                common_patterns = sorted(common_patterns)

                if len(common_patterns) > NUM_SAMPLES:
                    # Sample from common patterns
                    sampled_patterns = random.sample(common_patterns, NUM_SAMPLES)
                    files_by_cluster_1[cluster_num] = [patterns_1[p] for p in sampled_patterns]
                    files_by_cluster_2[cluster_num] = [patterns_2[p] for p in sampled_patterns]
                    if files_3:
                        files_by_cluster_3[cluster_num] = [patterns_3[p] for p in sampled_patterns]
                    print(f"   Cluster {cluster_num}: {len(common_patterns)} common files → {NUM_SAMPLES} files")
                else:
                    # Use all common patterns
                    files_by_cluster_1[cluster_num] = [patterns_1[p] for p in common_patterns]
                    files_by_cluster_2[cluster_num] = [patterns_2[p] for p in common_patterns]
                    if files_3:
                        files_by_cluster_3[cluster_num] = [patterns_3[p] for p in common_patterns]
                    print(f"   Cluster {cluster_num}: Using all {len(common_patterns)} common files")
            else:
                if len(files_1) > NUM_SAMPLES:
                    files_by_cluster_1[cluster_num] = random.sample(files_1, NUM_SAMPLES)
                    print(f"   Cluster {cluster_num}: {len(files_1)} → {NUM_SAMPLES} files")

    # Generate plots
    print(f"\nGenerating attention plots...")
    plot_count = 0

    if comparison_mode:
        # Comparison mode: plot side-by-side
        for cluster_num in sorted(common_clusters):
            files_1 = files_by_cluster_1[cluster_num]
            files_2 = files_by_cluster_2[cluster_num]
            files_3 = files_by_cluster_3[cluster_num] if files_by_cluster_3 and cluster_num in files_by_cluster_3 else None

            # Find common filenames across all directories by extracting example index
            def get_base_pattern(filepath):
                """Extract example index from filename for matching across different file types"""
                filename = os.path.basename(filepath)

                # For contaminated files: "contaminated_ex1000_cs3_cont1of3_rate0.020.npz"
                if filename.startswith('contaminated_ex'):
                    # Extract the example number (ex1000 -> 1000)
                    example_idx = filename.split('_')[1][2:]  # Remove 'ex' prefix
                    return f"ex{example_idx}"

                # For regular attention files: "attention_sample_1000.pt"
                elif filename.startswith('attention_sample_'):
                    # Extract the example number
                    example_idx = os.path.splitext(filename)[0].split('_')[-1]
                    return f"ex{example_idx}"

                # Fallback to original filename without extension
                else:
                    return os.path.splitext(filename)[0]

            # Map base_pattern -> full_path for each directory
            filenames_1 = {get_base_pattern(f): f for f in files_1}
            filenames_2 = {get_base_pattern(f): f for f in files_2}

            if files_3:
                filenames_3 = {get_base_pattern(f): f for f in files_3}
                common_patterns = set(filenames_1.keys()) & set(filenames_2.keys()) & set(filenames_3.keys())
            else:
                common_patterns = set(filenames_1.keys()) & set(filenames_2.keys())

            common_patterns = sorted(common_patterns)  # Sort for consistent ordering

            print(f"\n--- Processing Cluster Size {cluster_num} ({len(common_patterns)} common files) ---")

            for i, base_pattern in enumerate(common_patterns):
                # Build file paths for this common pattern across all directories
                file_paths = [filenames_1[base_pattern], filenames_2[base_pattern]]
                if files_3:
                    file_paths.append(filenames_3[base_pattern])

                # Load all attention data files
                data_list = []
                all_loaded = True

                for filepath in file_paths:
                    data = load_attention_data(filepath)
                    if data is None:
                        print(f"  Failed to load: {os.path.basename(filepath)}")
                        all_loaded = False
                        break
                    data_list.append(data)

                if not all_loaded:
                    continue

                # Generate output path if saving
                output_path = None
                if SAVE_PLOTS:
                    # Handle both .pt and .npz files
                    filename_out = f"{base_pattern}_comparison.pdf"
                    output_path = os.path.join(OUTPUT_DIR, f"cluster_{cluster_num}_{filename_out}")

                # Print info
                print(f"Example {data_list[0].get('example_idx', 'unknown')} | Pattern: {base_pattern} ({i+1}/{len(common_patterns)})")

                # Plot comparison
                plot_comparison(
                    data_list=data_list,
                    output_path=output_path,
                    show_plot=SHOW_PLOTS,
                    figsize=FIGSIZE
                )

                plot_count += 1

                # Add spacing between plots
                if SHOW_PLOTS and i < len(common_patterns) - 1:
                    print("\\n" + "-"*60 + "\\n")

    else:
        # Single mode: plot individual files
        for cluster_num in sorted(files_by_cluster_1.keys()):
            files = files_by_cluster_1[cluster_num]
            print(f"\n--- Processing Cluster Size {cluster_num} ({len(files)} files) ---")

            for i, filepath in enumerate(files):
                # Load attention data
                data = load_attention_data(filepath)
                if data is None:
                    print(f"Failed to load: {os.path.basename(filepath)}")
                    continue

                # Generate output path if saving
                output_path = None
                if SAVE_PLOTS:
                    # Handle both .pt and .npz files
                    base_filename = os.path.splitext(os.path.basename(filepath))[0]
                    filename = f"{base_filename}.pdf"
                    output_path = os.path.join(OUTPUT_DIR, f"cluster_{cluster_num}_{filename}")

                # Print the input sequence and results from the file
                print(f"  Example {data.get('example_idx', 'unknown')} (file {i+1}/{len(files)})")
                if 'token_sequence' in data:
                    input_seq = data['token_sequence'].split(':')[0] if ':' in data['token_sequence'] else data['token_sequence']
                    print(f"     Input sequence: {input_seq}")

                # Check for ground truth and prediction
                if 'ground_truth' in data and 'prediction' in data:
                    gt = data['ground_truth']
                    pred = data['prediction']
                    print(f"     Ground truth:   {gt}")
                    print(f"     Prediction:     {pred}")
                    # Calculate normalized Levenshtein
                    if gt and pred:
                        lev_dist = Levenshtein.distance(pred, gt)
                        max_len = max(len(gt), len(pred))
                        norm_lev = lev_dist / max_len if max_len > 0 else 0
                        print(f"     Levenshtein:    {lev_dist} (normalized: {norm_lev:.3f})")
                    else:
                        print(f"     Warning: GT or prediction is empty, cannot compute Levenshtein distance")
                else:
                    print(f"     Warning: No GT/prediction data - file may be from older run without these fields")

                plot_attention_matrix(
                    data=data,
                    output_path=output_path,
                    show_plot=SHOW_PLOTS,
                    figsize=FIGSIZE
                )

                plot_count += 1

                # Add spacing between plots for better readability
                if SHOW_PLOTS and i < len(files) - 1:
                    print("\\n" + "-"*60 + "\\n")

    # Summary
    print(f"\nGenerated {plot_count} attention plots")
    if SAVE_PLOTS:
        print(f"Plots saved to: {OUTPUT_DIR}")

    print("\nAttention visualization complete!")



# 
# ENTRY POINT
# 

if __name__ == "__main__":
    # Try to run in notebook first, then as script
    main()