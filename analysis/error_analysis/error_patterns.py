# Run from repo root: cd /path/to/TReconLM && python -m analysis.error_analysis.error_patterns
import csv
import re
from collections import defaultdict
import numpy as np
from scipy import stats
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from analysis.helpers import get_edit_operations


def extract_context(sequence, pos, window_size):
    """Extract context window around position."""
    half_window = window_size // 2
    start = pos - half_window
    end = pos + half_window + 1

    if start < 0 or end > len(sequence):
        return None

    return sequence[start:end]


def analyze_pattern_errors(filepath, window_sizes=[3, 5, 7], min_count=20):
    """
    Analyze error patterns with statistical hypothesis testing.
    
    Args:
        filepath: Path to predictions TSV file
        window_sizes: List of k-mer lengths to analyze
        min_count: Minimum pattern occurrences for testing
    """
    results = {}
    
    for window_size in window_sizes:
        print(f"\nAnalyzing with {window_size}-mer patterns...")
        
        # Track pattern occurrences and errors
        pattern_counts = defaultdict(int)
        pattern_errors = defaultdict(int)
        pattern_error_types = defaultdict(lambda: {'sub': 0, 'del': 0, 'ins': 0})
        total_positions = 0
        total_errors = 0
        total_error_types = {'sub': 0, 'del': 0, 'ins': 0}
        
        with open(filepath) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                gt = row["ground_truth"]
                pred = row["prediction"]
                
                operations = get_edit_operations(gt, pred)
                
                # Count all possible patterns in ground truth
                for i in range(len(gt)):
                    context = extract_context(gt, i, window_size)
                    if context:
                        pattern_counts[context] += 1
                        total_positions += 1
                
                # Count patterns where errors occur
                for op in operations:
                    op_type = op[0]
                    pos = op[1]
                    context = extract_context(gt, pos, window_size)
                    if context:
                        pattern_errors[context] += 1
                        pattern_error_types[context][op_type] += 1
                        total_errors += 1
                        total_error_types[op_type] += 1
        
        # Calculate overall error rate
        r_overall = total_errors / total_positions if total_positions > 0 else 0
        
        # Track all patterns and filtering
        n_total_patterns = len(pattern_counts)
        patterns_to_test = []
        patterns_filtered = []
        p_values = []
        
        for pattern in pattern_counts:
            N_pattern = pattern_counts[pattern]
            E_pattern = pattern_errors[pattern]
            r_pattern = E_pattern / N_pattern
            enrichment = r_pattern / r_overall if r_overall > 0 else 0
            
            # Filter by minimum count
            if N_pattern < min_count:
                patterns_filtered.append({
                    'pattern': pattern,
                    'N_pattern': N_pattern,
                    'E_pattern': E_pattern,
                    'enrichment': enrichment
                })
                continue
            
            # Binomial test: two-sided
            result = binomtest(
                k=E_pattern,
                n=N_pattern,
                p=r_overall,
                alternative='two-sided'
            )
            
            patterns_to_test.append({
                'pattern': pattern,
                'N_pattern': N_pattern,
                'E_pattern': E_pattern,
                'r_pattern': r_pattern,
                'enrichment': enrichment,
                'error_types': dict(pattern_error_types[pattern])
            })
            p_values.append(result.pvalue)
        
        # Multiple testing correction (Benjamini-Hochberg)
        if len(p_values) > 0:
            reject, p_adjusted, _, _ = multipletests(
                p_values,
                method='fdr_bh',
                alpha=0.05
            )
            
            # Add adjusted p-values and significance to results
            for i, pattern_info in enumerate(patterns_to_test):
                pattern_info['p_value'] = p_values[i]
                pattern_info['p_adjusted'] = p_adjusted[i]
                pattern_info['significant'] = reject[i]
        
        # Sort by enrichment
        significant_patterns = [p for p in patterns_to_test if p['significant']]
        significant_patterns.sort(key=lambda x: x['enrichment'], reverse=True)
        
        # Sort filtered patterns by enrichment (for reporting)
        patterns_filtered.sort(key=lambda x: x['enrichment'], reverse=True)
        
        # Store results
        results[window_size] = {
            'total_positions': total_positions,
            'total_errors': total_errors,
            'r_overall': r_overall,
            'n_total_patterns': n_total_patterns,
            'n_patterns_tested': len(patterns_to_test),
            'n_patterns_filtered': len(patterns_filtered),
            'n_significant': len(significant_patterns),
            'significant_patterns': significant_patterns,
            'all_tested_patterns': patterns_to_test,
            'filtered_patterns': patterns_filtered[:10]  # Top 10 filtered
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Results for {window_size}-mer patterns")
        print(f"{'='*60}")

        print(f"\nOverall statistics:")
        print(f"  Total positions: {total_positions:,}")
        print(f"  Total errors: {total_errors:,}")
        print(f"  Overall error rate (r_overall): {r_overall:.3%}")

        print(f"\nPattern filtering:")
        print(f"  Total unique patterns observed: {n_total_patterns}")
        print(f"  Patterns with N >= {min_count} (tested): {len(patterns_to_test)} ({100*len(patterns_to_test)/n_total_patterns:.1f}%)")
        print(f"  Patterns with N < {min_count} (filtered): {len(patterns_filtered)} ({100*len(patterns_filtered)/n_total_patterns:.1f}%)")
        print(f"  Significant patterns (FDR < 0.05): {len(significant_patterns)}")

        # Error type breakdown
        print(f"\nError type breakdown:")
        if total_errors > 0:
            for error_type in ['sub', 'del', 'ins']:
                count = total_error_types[error_type]
                rate = count / total_errors
                print(f"  {error_type.capitalize():12s}: {count:,} ({rate:.1%})")

        # Top 10 hardest (highest enrichment), including ties at rank 10
        print(f"\nTop hardest patterns (highest error enrichment, min 10 + ties):")
        print(f"  {'Rank':<5} {'Pattern':<{window_size+2}} {'Enrich':<8} {'p_adj':<10} {'N':<8} {'Errors':<8}")
        print(f"  {'-'*5} {'-'*(window_size+2)} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

        cutoff_enrich = significant_patterns[9]['enrichment'] if len(significant_patterns) >= 10 else None
        for i, p in enumerate(significant_patterns):
            if i >= 10 and (cutoff_enrich is None or p['enrichment'] < cutoff_enrich):
                break
            print(f"  {i+1:<5} '{p['pattern']}'  {p['enrichment']:>6.2f}x  {p['p_adjusted']:>8.4f}  "
                  f"{p['N_pattern']:>6}  {p['E_pattern']:>6}")

        # Pattern class enrichment analysis
        # Classify each k-mer by its repeat structure, e.g. for 3-mers:
        #   XXX (all same), XXY (first two same), XYX (ends same), XYY (last two same), XYZ (all different)
        def get_pattern_class(pattern):
            """Map a k-mer to its abstract class like XXY, XYZ, etc."""
            mapping = {}
            next_label = 0
            labels = []
            for ch in pattern:
                if ch not in mapping:
                    mapping[ch] = next_label
                    next_label += 1
                labels.append(mapping[ch])
            return ''.join(chr(ord('X') + l) for l in labels)

        # Group tested patterns by class
        class_patterns = defaultdict(list)
        for p in patterns_to_test:
            cls = get_pattern_class(p['pattern'])
            class_patterns[cls].append(p)

        print(f"\nPattern class enrichment:")
        print(f"  {'Class':<10} {'#Types':<8} {'N':>10} {'Errors':>10} {'Rate':>10} {'Enrich':>8}")
        print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

        class_stats = []
        for cls in sorted(class_patterns.keys()):
            pats = class_patterns[cls]
            cls_N = sum(p['N_pattern'] for p in pats)
            cls_E = sum(p['E_pattern'] for p in pats)
            cls_rate = cls_E / cls_N if cls_N > 0 else 0
            cls_enrich = cls_rate / r_overall if r_overall > 0 else 0
            class_stats.append((cls, len(pats), cls_N, cls_E, cls_rate, cls_enrich))

        # Sort by enrichment descending
        class_stats.sort(key=lambda x: x[5], reverse=True)
        for cls, n_types, cls_N, cls_E, cls_rate, cls_enrich in class_stats:
            print(f"  {cls:<10} {n_types:<8} {cls_N:>10,} {cls_E:>10,} {cls_rate:>10.3%} {cls_enrich:>7.2f}x")
    
    return results


def generate_latex_table_with_stats(results_dict):
    """Generate LaTeX table with statistical significance."""
    latex_code = []

    latex_code.append("\\begin{table}[t]")
    latex_code.append("\\centering")
    latex_code.append("\\caption{Top 5 sequence patterns with significantly higher reconstruction error rates than expected under uniform error distribution. Enrichment indicates how much more difficult each pattern is to reconstruct relative to the overall error rate.}")
    latex_code.append("\\vspace{1em}")
    latex_code.append("\\label{tab:error_patterns}")
    latex_code.append("\\small")
    latex_code.append("\\renewcommand{\\arraystretch}{1.2}")
    latex_code.append("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}c|c|c|c|c|c|c|c|c}")
    latex_code.append("\\hline")

    # Header row, just the length categories
    latex_code.append("\\multicolumn{3}{c|}{\\textbf{Length 3}} & \\multicolumn{3}{c|}{\\textbf{Length 5}} & \\multicolumn{3}{c}{\\textbf{Length 7}} \\\\")
    latex_code.append("\\hline")

    # Column headers
    latex_code.append("Pat & $N_{\\text{pattern}}$ & Enr & Pat & $N_{\\text{pattern}}$ & Enr & Pat & $N_{\\text{pattern}}$ & Enr \\\\")
    latex_code.append("\\hline")

    # Get the first dataset
    datasets = list(results_dict.keys())
    if len(datasets) > 0:
        dataset = datasets[0]
        dataset_results = results_dict[dataset]

        # Data rows 
        for rank in range(1, 6):
            row = []

            for window_size in [3, 5, 7]:
                if window_size in dataset_results:
                    patterns = dataset_results[window_size]['significant_patterns']
                    if rank <= len(patterns):
                        p = patterns[rank-1]
                        row.extend([
                            f"\\texttt{{{p['pattern']}}}",
                            f"{p['N_pattern']}",
                            f"{p['enrichment']:.2f}x"
                        ])
                    else:
                        row.extend(["-", "-", "-"])
                else:
                    row.extend(["-", "-", "-"])

            latex_code.append(" & ".join(row) + " \\\\")

    latex_code.append("\\hline")
    latex_code.append("\\end{tabular*}")
    latex_code.append("\\end{table}")

    return "\n".join(latex_code)


if __name__ == "__main__":
    # Set your prediction output file path
    test_files = {
        'synthetic': '/mnt/TReconLM/pred_gt/synthetic_L110/predictions_output.tsv',  # path to predictions_output.tsv (e.g., '/mnt/.../synthetic_L110/predictions_output.tsv')
    }

    all_results = {}

    for dataset_name, filepath in test_files.items():
        print(f"\n{'#'*60}")
        print(f"Analyzing {dataset_name} dataset")
        print(f"{'#'*60}")
        results = analyze_pattern_errors(filepath, window_sizes=[3, 5, 7], min_count=20)
        all_results[dataset_name] = results

    # Generate LaTeX table
    print(f"\n{'#'*60}")
    print("LaTeX table code")
    print(f"{'#'*60}")
    latex_table = generate_latex_table_with_stats(all_results)
    print(latex_table)