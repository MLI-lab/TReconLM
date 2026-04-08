"""
Convert dnaformer binned data files to the standard format used by TReconLM:
  - reads.txt: clusters separated by '==============================='
  - ground_truth.txt: one GT per line

Also removes the first 12 characters (index) from all sequences and
subclusters large clusters (>max_reads) using create_subclusters.
"""

import os
import sys
import argparse

# Add repo root to path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.helper_functions import create_subclusters

INDEX_LEN = 12


def convert_binned_to_standard(input_path, output_dir, max_reads=10):
    """Parse binned format (GT / *** / reads / blank lines), subcluster, and write reads.txt + ground_truth.txt."""
    with open(input_path, "r") as f:
        lines = f.readlines()

    clusters = []  # list of (gt, [reads])
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        # Skip blank lines
        if line == "":
            i += 1
            continue
        # This should be a GT line, next line should be ***
        gt = line[INDEX_LEN:]
        i += 1
        if i < len(lines) and lines[i].strip().startswith("*"):
            i += 1  # skip *** separator
        # Collect reads until blank line or EOF
        reads = []
        while i < len(lines):
            rline = lines[i].rstrip("\n")
            if rline == "":
                i += 1
                break
            reads.append(rline[INDEX_LEN:])
            i += 1
        if reads:
            clusters.append((gt, reads))

    raw_count = len(clusters)

    # Subcluster: split large clusters into 2-max_reads sized subclusters
    all_reads = [reads for _, reads in clusters]
    all_gts = [gt for gt, _ in clusters]
    examples, cluster_sizes = create_subclusters(all_reads, all_gts, max_reads=max_reads)

    # Parse examples back into (gt, [reads]) pairs
    subclustered = []
    for example in examples:
        reads_str, gt = example.rsplit(':', 1)
        reads = reads_str.split('|')
        subclustered.append((gt, reads))

    os.makedirs(output_dir, exist_ok=True)

    # Write reads.txt
    reads_path = os.path.join(output_dir, "reads.txt")
    with open(reads_path, "w") as f:
        for idx, (gt, reads) in enumerate(subclustered):
            for read in reads:
                f.write(read + "\n")
            if idx < len(subclustered) - 1:
                f.write("===============================\n")

    # Write ground_truth.txt
    gt_path = os.path.join(output_dir, "ground_truth.txt")
    with open(gt_path, "w") as f:
        for gt, _ in subclustered:
            f.write(gt + "\n")

    print(f"Converted {raw_count} clusters from {os.path.basename(input_path)}")
    print(f"  -> {len(subclustered)} subclusters (max_reads={max_reads})")
    print(f"  -> {reads_path}")
    print(f"  -> {gt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert dnaformer binned data to standard reads.txt + ground_truth.txt format."
    )
    parser.add_argument("input_file", nargs="?", help="Path to a single input .txt file (optional)")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "without_index_data"),
        help="Base output directory (default: ./without_index_data)",
    )
    parser.add_argument(
        "--max_reads", type=int, default=10,
        help="Maximum reads per subcluster (default: 10)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all .txt files in the dnaformer directory",
    )
    args = parser.parse_args()

    if args.all:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        txt_files = sorted(f for f in os.listdir(script_dir) if f.endswith(".txt"))
        for fname in txt_files:
            input_path = os.path.join(script_dir, fname)
            # Each file gets its own subdirectory
            name = os.path.splitext(fname)[0]
            out_dir = os.path.join(args.output_dir, name)
            convert_binned_to_standard(input_path, out_dir, max_reads=args.max_reads)
    elif args.input_file:
        name = os.path.splitext(os.path.basename(args.input_file))[0]
        out_dir = os.path.join(args.output_dir, name)
        convert_binned_to_standard(args.input_file, out_dir, max_reads=args.max_reads)
    else:
        parser.error("Provide an input file or use --all")
