#!/bin/bash
# Run inference with multiple shuffle seeds to evaluate input-ordering robustness.
# Parses failure rate and Levenshtein distance from each run, then reports mean ± std.
#
# Usage:
#   bash analysis/error_analysis/run_permutation_robustness.sh NUM_SEEDS [HYDRA_ARGS...]
#   bash analysis/error_analysis/run_permutation_robustness.sh --aggregate NUM_SEEDS
#
# Arguments:
#   NUM_SEEDS       Number of random read-orderings to evaluate (e.g. 20).
#   HYDRA_ARGS      Any extra Hydra overrides forwarded to src.inference
#                   (experiment config, checkpoint path, wandb toggles, etc.).
#   --aggregate     Skip inference, just re-parse existing logs in shuffle_robustness_results/.
#
# Examples:
#   bash analysis/error_analysis/run_permutation_robustness.sh 20 exps=ids_110
#   bash analysis/error_analysis/run_permutation_robustness.sh 20 exps=ids_110 wandb.wandb_log=false
#   bash analysis/error_analysis/run_permutation_robustness.sh 10 exps=ids_110 model.checkpoint_path=/path/to/model.pt
#   bash analysis/error_analysis/run_permutation_robustness.sh --aggregate 20

set -e

# cd to repo root (parent of scripts/)
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PYTHONPATH"

RESULTS_DIR="shuffle_robustness_results"
mkdir -p "$RESULTS_DIR"

# Check for --aggregate flag
if [ "$1" = "--aggregate" ]; then
    NUM_SEEDS="${2:-20}"
    echo "Aggregating existing results for $NUM_SEEDS seeds..."
else
    NUM_SEEDS="${1:-20}"
    shift 1 2>/dev/null || true
    HYDRA_ARGS="$@"

    echo "=============================================="
    echo "Input-Ordering Robustness Evaluation"
    echo "=============================================="
    echo "Seeds:      1 to $NUM_SEEDS"
    echo "Hydra args: $HYDRA_ARGS"
    echo "Results:    $RESULTS_DIR/"
    echo "=============================================="

    # Run inference for each seed
    for seed in $(seq 1 "$NUM_SEEDS"); do
        echo ""
        echo "[Seed $seed/$NUM_SEEDS] Running inference..."
        LOG_FILE="$RESULTS_DIR/seed_${seed}.log"

        python -m src.inference \
            $HYDRA_ARGS \
            model.sampling.shuffle_reads_seed="$seed" \
            2>&1 | tee "$LOG_FILE"

        echo "[Seed $seed/$NUM_SEEDS] Done. Log saved to $LOG_FILE"
    done

    echo ""
    echo "=============================================="
    echo "All $NUM_SEEDS runs completed. Aggregating..."
    echo "=============================================="
fi

# Aggregate results with Python
python3 - "$RESULTS_DIR" "$NUM_SEEDS" <<'PYEOF'
import sys, re, os
from collections import defaultdict
import numpy as np

results_dir = sys.argv[1]
num_seeds = int(sys.argv[2])

failure_rates = []
levenshtein_means = []
# Per-cluster-size: {N: {'failure_rates': [...], 'lev_means': [...]}}
per_N = defaultdict(lambda: {'failure_rates': [], 'lev_means': []})

for seed in range(1, num_seeds + 1):
    log_file = os.path.join(results_dir, f"seed_{seed}.log")
    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found, skipping")
        continue

    with open(log_file, 'r') as f:
        content = f.read()

    # Parse overall line: "  Overall: success=46.40%, failure=53.60%, HAM=0.174±0.226, LEV=0.0341±0.0508"
    overall_match = re.search(
        r'Overall:.*?failure=([\d.]+)%.*?LEV=([\d.]+)±',
        content
    )
    if overall_match:
        failure_rates.append(float(overall_match.group(1)))
        levenshtein_means.append(float(overall_match.group(2)))
    else:
        # Fallback: try "Failure rate (cropped): X.XX%" format
        fr_match = re.search(r'Failure rate \(cropped\):\s*([\d.]+)%', content)
        lev_match = re.search(r'Levenshtein \(cropped\):\s*([\d.]+)', content)
        if fr_match:
            failure_rates.append(float(fr_match.group(1)))
        if lev_match:
            levenshtein_means.append(float(lev_match.group(1)))

    # Parse per-cluster-size lines:
    #   N= 2: count= 5000, success=95.00%, failure=5.00%, HAM=0.050±0.218, LEV=0.0045±0.0200
    for m in re.finditer(
        r'N=\s*(\d+):\s*count=\s*\d+,\s*success=[\d.]+%,\s*failure=([\d.]+)%,\s*'
        r'HAM=[\d.]+±[\d.]+,\s*LEV=([\d.]+)±[\d.]+',
        content
    ):
        N = int(m.group(1))
        fr = float(m.group(2))
        lev = float(m.group(3))
        per_N[N]['failure_rates'].append(fr)
        per_N[N]['lev_means'].append(lev)

if not failure_rates:
    print("ERROR: Could not parse any results. Check log files in", results_dir)
    sys.exit(1)

failure_rates = np.array(failure_rates)
levenshtein_means = np.array(levenshtein_means)

print()
print("=" * 70)
print("SHUFFLE ROBUSTNESS RESULTS")
print("=" * 70)
print(f"Number of seeds: {len(failure_rates)}")

# Per-cluster-size results
if per_N:
    print()
    print("Per cluster size (mean ± std across runs):")
    print(f"{'N':>4}  {'Failure Rate':>20}  {'Levenshtein':>20}")
    print("-" * 48)
    for N in sorted(per_N.keys()):
        fr_arr = np.array(per_N[N]['failure_rates'])
        lev_arr = np.array(per_N[N]['lev_means'])
        print(f"{N:>4}  {fr_arr.mean():>8.2f}% ± {fr_arr.std():<7.2f}%  {lev_arr.mean():>8.4f} ± {lev_arr.std():<8.4f}")

# Overall results
print()
print(f"{'Overall':>4}  {failure_rates.mean():>8.2f}% ± {failure_rates.std():<7.2f}%  {levenshtein_means.mean():>8.4f} ± {levenshtein_means.std():<8.4f}")
print()
print("=" * 70)
print()
print("Per-seed overall results:")
print(f"{'Seed':>6}  {'Failure Rate':>14}  {'Levenshtein':>12}")
print("-" * 36)
for i in range(len(failure_rates)):
    print(f"{i+1:>6}  {failure_rates[i]:>13.2f}%  {levenshtein_means[i]:>12.4f}")

# Markdown table
if per_N:
    sorted_Ns = sorted(per_N.keys())
    cols = [f"N={N}" for N in sorted_Ns] + ["Overall"]

    # Build markdown table
    header = "| Metric | " + " | ".join(cols) + " |"
    separator = "|--------|" + "|".join(["-------"] * len(cols)) + "|"

    fr_cells = []
    for N in sorted_Ns:
        fr_arr = np.array(per_N[N]['failure_rates'])
        fr_cells.append(f"{fr_arr.mean():.2f} ± {fr_arr.std():.2f}")
    fr_cells.append(f"{failure_rates.mean():.2f} ± {failure_rates.std():.2f}")
    fr_row = "| Failure Rate (%) | " + " | ".join(fr_cells) + " |"

    lev_cells = []
    for N in sorted_Ns:
        lev_arr = np.array(per_N[N]['lev_means'])
        lev_cells.append(f"{lev_arr.mean():.4f} ± {lev_arr.std():.4f}")
    lev_cells.append(f"{levenshtein_means.mean():.4f} ± {levenshtein_means.std():.4f}")
    lev_row = "| Levenshtein | " + " | ".join(lev_cells) + " |"

    table = "\n".join([header, separator, fr_row, lev_row])

    # Print to terminal
    print()
    print()
    print("MARKDOWN TABLE (copy-paste into paper):")
    print()
    print(table)

    # Save to file
    table_file = os.path.join(results_dir, "robustness_table.md")
    with open(table_file, 'w') as f:
        f.write("# Input-Ordering Robustness Results\n\n")
        f.write(f"Seeds: {len(failure_rates)}\n\n")
        f.write(table + "\n")
    print()
    print(f"Table saved to {table_file}")
PYEOF
