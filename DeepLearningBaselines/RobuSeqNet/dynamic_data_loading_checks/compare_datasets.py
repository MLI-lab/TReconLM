import os

def load_reads(reads_path):
    with open(reads_path, 'r') as f:
        lines = f.read().splitlines()

    clusters = []
    current_cluster = []
    for line in lines:
        if line.strip() == "=" * 30:
            if current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
        else:
            current_cluster.append(line.strip())
    if current_cluster:
        clusters.append(current_cluster)

    return clusters

def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def compare_datasets(generated_dir, fixed_dir):
    reads_dynamic = load_reads(os.path.join(generated_dir, 'reads_dynamic.txt'))
    labels_dynamic = load_labels(os.path.join(generated_dir, 'labels_dynamic.txt'))

    reads_fixed = load_reads(os.path.join(fixed_dir, 'reads.txt'))
    labels_fixed = load_labels(os.path.join(fixed_dir, 'reference.txt'))

    # Compare number of samples
    if len(reads_dynamic) != len(reads_fixed):
        print(f"Number of clusters mismatch: {len(reads_dynamic)} vs {len(reads_fixed)}")
        return

    if len(labels_dynamic) != len(labels_fixed):
        print(f"Number of labels mismatch: {len(labels_dynamic)} vs {len(labels_fixed)}")
        return

    print(f"Number of clusters and labels match: {len(reads_dynamic)} samples.")

    # Compare each sample
    for idx, (r_dyn, r_fix) in enumerate(zip(reads_dynamic, reads_fixed)):
        if r_dyn != r_fix:
            print(f"Mismatch in reads at sample {idx}")
            print(f"Dynamic reads: {r_dyn}")
            print(f"Fixed reads:   {r_fix}")
            return

    for idx, (l_dyn, l_fix) in enumerate(zip(labels_dynamic, labels_fixed)):
        if l_dyn != l_fix:
            print(f"Mismatch in labels at sample {idx}")
            print(f"Dynamic label: {l_dyn}")
            print(f"Fixed label:   {l_fix}")
            return

    print("All clusters and labels match.")

if __name__ == "__main__":
    generated_dir = "/workspaces/TReconLM/Multi-Read-Reconstruction/examples/generated_data"
    fixed_dir = "/workspaces/TReconLM/Multi-Read-Reconstruction/examples/data"
    compare_datasets(generated_dir, fixed_dir)
