# Combined plotting script for RobuSeqNet, TReconLM, and gpt-4o Mini
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import wandb
from tqdm import tqdm
import re
from Levenshtein import distance as levenshtein_distance



# Add TReconLM root to path
sys.path.insert(0, "/workspaces/TReconLM")
from src.utils.hamming_distance import hamming_distance_postprocessed

# --- Settings ---
fontsize = 9
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{type1cm}',
    'font.size': fontsize,
})

save_dir = "./plots"
download_dir = "./downloaded_artifact"
entity = "<your.wandb.entity>"
project_wandb = "TRACE_RECONSTRUCTION"
gpt_project = "GPTMini"  # New project for GPT models
os.makedirs(save_dir, exist_ok=True)

# --- Visual Config ---
color_dict = {
    "TReconLM": "#6699CC",
    "RobuSeqNet": "#F8DB57",
    "gpt-4o mini 0-shot": "#CC76D2",
    "gpt-4o mini 3-shot": "#DA493E",
    "gpt-4o mini 5-shot": "#F09236",
    "Avg. noisy read": "#aaaaaa",
    "Best noisy read": "#aaaaaa",
}
marker_dict = {
    "TReconLM": "o",
    "RobuSeqNet": "P",
    "gpt-4o mini 0-shot": "^",
    "gpt-4o mini 3-shot": "s",
    "gpt-4o mini 5-shot": "D",
}
label_dict = {
    "TReconLM": r"\textsc{TReconLM}",
    "RobuSeqNet": r"\textsc{RobuSeqNet}",
    "gpt-4o mini 0-shot": r"\textsc{gpt-4o Mini} 0-shot",
    "gpt-4o mini 3-shot": r"\textsc{gpt-4o Mini} 3-shot",
    "gpt-4o mini 5-shot": r"\textsc{gpt-4o Mini} 5-shot",
    "Avg. noisy read": None,
    "Best noisy read": None,
}

# --- Fallback gpt Metrics (if W&B fetch fails) ---
normalize = lambda d: {k: v / 60.0 for k, v in d.items()}
gpt_data_fallback = {
    "gpt-4o mini 0-shot": normalize({2: 12.748, 5: 12.136, 10: 10.948}),
    "gpt-4o mini 3-shot": normalize({2: 11.28, 5: 8.672, 10: 6.54}),
    "gpt-4o mini 5-shot": normalize({2: 11.168, 5: 8.75, 10: 6.088}),
}

# --- Functions ---
def filter_invalid(vals):
    return [v if v is not None and not np.isnan(v) else np.nan for v in vals]

def compute_noisy_baselines_from_artifact(artifact_name):
    api = wandb.Api()
    art = api.artifact(f"{entity}/{project_wandb}/{artifact_name}", type="dataset")
    d = art.download(download_dir)
    with open(os.path.join(d, "reads.txt")) as f:
        reads = [l.strip() for l in f]
    with open(os.path.join(d, "ground_truth.txt")) as f:
        gts = [l.strip() for l in f]
    clusters, cur = [], []
    for line in reads:
        if line.startswith("="):
            if cur: clusters.append(cur)
            cur = []
        else:
            cur.append(line)
    if cur: clusters.append(cur)
    assert len(clusters) == len(gts)
    data = [(c, len(c), gt) for c, gt in zip(clusters, gts)]

    g_ld, m_ld = defaultdict(list), defaultdict(list)
    for reads, n, gt in tqdm(data):
        if not 2 <= n <= 10: continue
        L = len(gt)
        ls = [levenshtein_distance(gt, r) / L for r in reads]
        g_ld[n].append(np.mean(ls))
        m_ld[n].append(np.min(ls))

    return {
        "Avg. noisy read": {n: np.mean(g_ld[n]) for n in g_ld},
        "Best noisy read": {n: np.mean(m_ld[n]) for n in m_ld},
    }

def compute_noisy_baselines_from_artifact_map(artifact_map):
    avg_ld, min_ld = defaultdict(float), defaultdict(float)
    for n, artifact in artifact_map.items():
        out = compute_noisy_baselines_from_artifact(artifact)
        avg_ld[n] = out["Avg. noisy read"].get(n, np.nan)
        min_ld[n] = out["Best noisy read"].get(n, np.nan)
    return {
        "Avg. noisy read": avg_ld,
        "Best noisy read": min_ld
    }

def load_treconlm_and_robu_metrics():
    api = wandb.Api()
    runs = api.runs(f"{entity}/RobuSeqNet")
    metrics = defaultdict(lambda: defaultdict(dict))
    for run in runs:
        if run.state != "finished":
            continue
        name = run.name
        if "RobuSeq" in name:
            algo = "RobuSeqNet"
        elif "pretr" in name:
            algo = "TReconLM"
        else:
            continue

        for k, v in run.summary.items():
            if isinstance(v, (int, float)) and "N=" in k:
                match = re.match(r"(?:test/)?(avg_levenshtein|levenshtein_mean)_N=(\d+)", k)
                if match:
                    metric, n = match.groups()
                    n = int(n)
                    if 2 <= n <= 10:
                        if metric == "levenshtein_mean":
                            metric = "avg_levenshtein"
                        if algo == "RobuSeqNet":
                            v = v / 110
                        metrics[algo][metric][n] = v
    return metrics

def load_gpt_metrics_from_wandb():
    """
    Load GPT model metrics from W&B GPTMini project.
    Falls back to hardcoded values if loading fails.
    """
    api = wandb.Api()
    gpt_metrics = defaultdict(lambda: defaultdict(dict))

    try:
        print(f"Loading GPT metrics from W&B project: {entity}/{gpt_project}")
        runs = api.runs(f"{entity}/{gpt_project}")

        for run in runs:
            if run.state != "finished":
                continue

            name = run.name
            print(f"  Found run: {name}")

            # Parse model and shot count from run name or config
            # Expected format: "gpt-4o-mini_5shot" or similar
            model = None
            n_shots = None

            # Try to extract from run name
            if "gpt-4o-mini" in name.lower() or "gpt-4o mini" in name.lower():
                model = "gpt-4o-mini"

                # Try to find shot count
                shot_match = re.search(r'(\d+)[-_]?shot', name.lower())
                if shot_match:
                    n_shots = int(shot_match.group(1))
                elif "0shot" in name.lower() or "0-shot" in name.lower():
                    n_shots = 0

            # Try to extract from config
            if model is None or n_shots is None:
                config = run.config
                if 'model' in config:
                    model = config['model']
                if 'n_shots' in config:
                    n_shots = config['n_shots']

            if model and n_shots is not None:
                # Create key: "gpt-4o mini 0-shot", "gpt-4o mini 3-shot", etc.
                model_key = f"{model.replace('-', ' ')} {n_shots}-shot"

                # Extract metrics
                for k, v in run.summary.items():
                    if isinstance(v, (int, float)) and "N=" in k:
                        match = re.match(r"avg_levenshtein_N=(\d+)", k)
                        if match:
                            n = int(match.group(1))
                            if 2 <= n <= 10:
                                gpt_metrics[model_key]["avg_levenshtein"][n] = v
                                print(f"    Loaded {model_key}: N={n}, d_L={v:.4f}")

        # Convert to regular dict
        result = {}
        for model_key in gpt_metrics:
            if "avg_levenshtein" in gpt_metrics[model_key]:
                result[model_key] = dict(gpt_metrics[model_key]["avg_levenshtein"])

        if result:
            print(f"Successfully loaded {len(result)} GPT model configurations from W&B")
            return result
        else:
            print("No GPT metrics found in W&B, using fallback values")
            return gpt_data_fallback

    except Exception as e:
        print(f"Error loading GPT metrics from W&B: {e}")
        print("Using fallback hardcoded values")
        return gpt_data_fallback

def plot_combined(metrics, gpt_data, noisy_robu, noisy_gpt, plot_noisy_baselines=False):
    fig, axs = plt.subplots(1, 2, figsize=(6, 1.3), dpi=300, gridspec_kw={'wspace': 0.4})
    Ns_left = list(range(2, 11))
    Ns_right = [2, 5, 10]

    # --- Left subplot ---
    ax = axs[0]
    for model in ["RobuSeqNet", "TReconLM"]:
        ys = [metrics[model]["avg_levenshtein"].get(n, np.nan) for n in Ns_left]
        ax.plot(
            Ns_left, ys,
            label=label_dict[model],
            color=color_dict[model],
            marker=marker_dict[model],
            linewidth=0.5,
            markersize=2
        )

    if plot_noisy_baselines:
        for kind in ["Avg. noisy read", "Best noisy read"]:
            ys = [noisy_robu[kind].get(n, np.nan) for n in Ns_left]
            ax.plot(
                Ns_left, ys,
                color=color_dict[kind],
                linestyle=":" if "Avg" in kind else "--",
                linewidth=0.5
            )

    ax.set_xticks(Ns_left)
    ax.set_xlim(min(Ns_left) - 0.3, max(Ns_left) + 0.3)
    ax.set_xlabel(r"Cluster size $N$")
    ax.set_ylabel(r"$d_L$")
    ax.set_yticks([0.01, 0.1, 0.2, 0.3])
    ax.set_yticklabels(["0.01", "0.1", "0.2", "0.3"])

    # --- Right subplot ---
    ax = axs[1]
    for model in gpt_data:
        ys = [gpt_data[model].get(n, np.nan) for n in Ns_right]
        ax.plot(
            Ns_right, ys,
            label=label_dict.get(model, model),  # Use model name if not in label_dict
            color=color_dict.get(model, "#000000"),
            marker=marker_dict.get(model, "o"),
            linewidth=0.5,
            markersize=2
        )

    kinds = ["TReconLM"]
    if plot_noisy_baselines:
        kinds += ["Avg. noisy read", "Best noisy read"]

    for kind in kinds:
        ys = [
            metrics[kind]["avg_levenshtein"].get(n, np.nan)
            if kind == "TReconLM"
            else noisy_gpt[kind].get(n, np.nan)
            for n in Ns_right
        ]
        linestyle = "-" if kind == "TReconLM" else ":" if "Avg" in kind else "--"
        label = label_dict.get(kind, None)
        marker = marker_dict.get(kind, None)
        ax.plot(
            Ns_right, ys,
            label=label,
            color=color_dict[kind],
            linestyle=linestyle,
            marker=marker,
            linewidth=0.5,
            markersize=2 if marker else 0
        )

    ax.set_xticks(Ns_right)
    ax.set_xlim(min(Ns_right) - 0.3, max(Ns_right) + 0.3)
    ax.set_xlabel(r"Cluster size $N$")
    ax.set_ylabel(r"$d_L$")
    ax.set_ylim(0.0005, 0.3)
    ax.set_yticks([0.01, 0.1, 0.2, 0.3])
    ax.set_yticklabels(["0.01", "0.1", "0.2", "0.3"])

    # --- Style tweaks ---
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_color('lightgray')
        ax.tick_params(axis='both', which='both', color='lightgray', labelcolor='black')

    # --- Custom legend ---
    handles_labels = []
    for key in ["RobuSeqNet", "TReconLM"] + list(gpt_data.keys()):
        label = label_dict.get(key, None)
        color = color_dict.get(key, None)
        marker = marker_dict.get(key, None)
        if label and color and marker:
            h, = axs[1].plot([], [], label=label, color=color, marker=marker, linewidth=0.5, markersize=2)
            handles_labels.append((h, label))

    handles, labels = zip(*handles_labels)
    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.17),
        fontsize=8,
        handletextpad=0.3,
        columnspacing=0.6,
        handlelength=1.0,
        borderaxespad=0.2,
        frameon=False
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "combined_levenshtein_plot.pdf"), bbox_inches='tight')
    plt.show()

# --- Main ---
if __name__ == "__main__":
    artifact_robu = "test_dataset_seed34721_gl110_bs1500_ds50000:latest"
    artifact_map_gpt = {
        2: "GPTMini_cz2_seed34721_gl60:latest",
        5: "GPTMini_cz5_seed34721_gl60:latest",
        10: "GPTMini_cz10_seed34721_gl60:latest",
    }

    # Load metrics
    print("Loading TReconLM and RobuSeqNet metrics...")
    metrics = load_treconlm_and_robu_metrics()

    print("\nLoading GPT metrics from W&B...")
    gpt_data = load_gpt_metrics_from_wandb()

    print("\nComputing noisy baselines...")
    noisy_robu = compute_noisy_baselines_from_artifact(artifact_robu)
    noisy_gpt = compute_noisy_baselines_from_artifact_map(artifact_map_gpt)

    print("\nGenerating plot...")
    plot_combined(metrics, gpt_data, noisy_robu, noisy_gpt)
    print("Done!")
