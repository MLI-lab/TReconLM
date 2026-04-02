# Combined plotting script for RobuSeqNet, TReconLM, gpt-4o Mini, and gpt-4o Mini CoT
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
project_gpt_mini = "GPTMini"  # WandB project containing GPT-4o mini and GPT-5 runs
os.makedirs(save_dir, exist_ok=True)

# Gray color for legend elements (darker than lightgray for better visibility)
GRAY_COLOR = "#777777"

# --- Visual Config ---
# Colors for few-shot prompting strategies
color_dict = {
    "TReconLM": "#6699CC",
    "RobuSeqNet": "#F8DB57",
    "0-shot": "#CC76D2",  # Purple
    "3-shot": "#DA493E",  # Red
    "5-shot": "#F09236",  # Orange
    "Avg. noisy read": "#aaaaaa",
    "Best noisy read": "#aaaaaa",
}

# Markers for different models
marker_dict = {
    "TReconLM": "o",
    "RobuSeqNet": "P",
    "gpt-4o-mini": "D",  # Diamond
    "gpt-5": "s",  # Square (for future use)
}

# Line styles
linestyle_dict = {
    "standard": "-",  # Solid
    "cot": "--",  # Dashed
}

# Labels for legend
label_dict = {
    "TReconLM": r"\textsc{TReconLM}",
    "RobuSeqNet": r"\textsc{RobuSeqNet}",
    "Avg. noisy read": None,
    "Best noisy read": None,
}

# --- gpt Metrics ---
normalize = lambda d: {k: v / 60.0 for k, v in d.items()}
gpt_data = {
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

def load_gpt_cot_metrics(project_name):
    """
    Load GPT-4o mini CoT metrics from WandB.

    Args:
        project_name: WandB project name containing CoT runs

    Returns:
        Dictionary with structure: {model: {shot_type: {cluster_size: levenshtein}}}
        Note: Values are already normalized in WandB
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project_name}")
    cot_metrics = defaultdict(lambda: defaultdict(dict))

    print(f"  Searching for CoT runs in project: {project_name}")

    for run in runs:
        if run.state != "finished":
            continue

        name = run.name.lower()

        # Identify model type
        if "gpt-4o-mini" in name or "gpt-4o mini" in name:
            model = "gpt-4o-mini"
        elif "gpt-5" in name:
            model = "gpt-5"
        else:
            continue

        # Check if this is a CoT run
        if "thinking" not in name:
            continue

        # Identify shot type (be specific to avoid matching numbers in gl60, cs2, etc.)
        if "zero" in name or "_0shot" in name or name.endswith("_0") or "_0_" in name:
            shot_type = "0-shot-cot"
        elif "three" in name or "_3shot" in name or name.endswith("_3") or "_3_" in name:
            shot_type = "3-shot-cot"
        elif "five" in name or "_5shot" in name or name.endswith("_5") or "_5_" in name:
            shot_type = "5-shot-cot"
        else:
            continue

        print(f"  Found CoT run: {run.name} -> {model}, {shot_type}")

        # Extract metrics for different cluster sizes
        extracted_metrics = []
        for k, v in run.summary.items():
            if isinstance(v, (int, float)) and "avg_levenshtein_N=" in k:
                match = re.match(r"avg_levenshtein_N=(\d+)", k)
                if match:
                    n = int(match.group(1))
                    if 2 <= n <= 10:
                        # Values are already normalized in WandB
                        cot_metrics[model][shot_type][n] = v
                        extracted_metrics.append(f"N={n}: {v:.4f}")

        if extracted_metrics:
            print(f"    Metrics: {', '.join(extracted_metrics)}")

    if not cot_metrics:
        print("  No CoT runs found")
    else:
        print(f"  Loaded CoT metrics for: {dict(cot_metrics)}")

    return cot_metrics

def load_gpt5_metrics(project_name):
    """
    Load GPT-5 metrics from WandB, handling potentially split runs.

    Args:
        project_name: WandB project name containing GPT-5 runs

    Returns:
        Dictionary with structure: {model: {shot_type: {cluster_size: levenshtein}}}
        Note: Values are already normalized in WandB
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project_name}")
    gpt5_metrics = defaultdict(lambda: defaultdict(dict))

    print(f"  Searching for GPT-5 runs in project: {project_name}")

    for run in runs:
        if run.state != "finished":
            continue

        name = run.name.lower()

        # Only process GPT-5 runs
        if "gpt-5" not in name and "gpt5" not in name:
            continue

        model = "gpt-5"

        # Identify shot type and CoT (be specific to avoid matching numbers in gl60, cs2, etc.)
        is_cot = "thinking" in name
        if "zero" in name or "_0shot" in name or name.endswith("_0") or "_0_" in name:
            shot_type = "0-shot-cot" if is_cot else "0-shot"
        elif "three" in name or "_3shot" in name or name.endswith("_3") or "_3_" in name:
            shot_type = "3-shot-cot" if is_cot else "3-shot"
        elif "five" in name or "_5shot" in name or name.endswith("_5") or "_5_" in name:
            shot_type = "5-shot-cot" if is_cot else "5-shot"
        else:
            continue

        print(f"  Found GPT-5 run: {run.name} -> {model}, {shot_type}")

        # Extract metrics for different cluster sizes
        extracted_metrics = []
        for k, v in run.summary.items():
            if isinstance(v, (int, float)) and "avg_levenshtein_N=" in k:
                match = re.match(r"avg_levenshtein_N=(\d+)", k)
                if match:
                    n = int(match.group(1))
                    if 2 <= n <= 10:
                        # Values are already normalized in WandB
                        gpt5_metrics[model][shot_type][n] = v
                        extracted_metrics.append(f"N={n}: {v:.4f}")

        if extracted_metrics:
            print(f"    Metrics: {', '.join(extracted_metrics)}")

    if not gpt5_metrics:
        print("  No GPT-5 runs found")
    else:
        print(f"  Loaded GPT-5 metrics for: {dict(gpt5_metrics)}")

    return gpt5_metrics

def plot_combined(metrics, gpt_data, gpt_cot_data, gpt5_data, noisy_robu, noisy_gpt, plot_noisy_baselines=False):
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

    # Plot standard GPT-4o mini runs
    for model_key, data in gpt_data.items():
        # Extract shot type from model_key (e.g., "gpt-4o mini 0-shot" -> "0-shot")
        shot_type = model_key.split()[-1]  # Gets "0-shot", "3-shot", "5-shot"

        ys = [data.get(n, np.nan) for n in Ns_right]
        ax.plot(
            Ns_right, ys,
            color=color_dict[shot_type],
            marker=marker_dict["gpt-4o-mini"],
            linestyle=linestyle_dict["standard"],
            linewidth=0.5,
            markersize=2
        )

    # Plot GPT-4o mini CoT runs
    if gpt_cot_data and "gpt-4o-mini" in gpt_cot_data:
        for shot_type, data in gpt_cot_data["gpt-4o-mini"].items():
            # Extract base shot type (e.g., "0-shot-cot" -> "0-shot")
            base_shot = shot_type.replace("-cot", "")

            ys = [data.get(n, np.nan) for n in Ns_right]
            ax.plot(
                Ns_right, ys,
                color=color_dict[base_shot],
                marker=marker_dict["gpt-4o-mini"],
                linestyle=linestyle_dict["cot"],
                linewidth=0.5,
                markersize=2
            )

    # Plot GPT-5 runs (if available)
    if gpt5_data and "gpt-5" in gpt5_data:
        for shot_type, data in gpt5_data["gpt-5"].items():
            # Determine if CoT
            is_cot = "-cot" in shot_type
            base_shot = shot_type.replace("-cot", "") if is_cot else shot_type

            ys = [data.get(n, np.nan) for n in Ns_right]
            ax.plot(
                Ns_right, ys,
                color=color_dict[base_shot],
                marker=marker_dict["gpt-5"],
                linestyle=linestyle_dict["cot"] if is_cot else linestyle_dict["standard"],
                linewidth=0.5,
                markersize=2
            )

    # Plot TReconLM on right subplot
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
            color=color_dict.get(kind, "#6699CC"),
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
            spine.set_color('lightgray')  # Keep frame lightgray
        ax.tick_params(axis='both', which='both', color='lightgray', labelcolor='black')

    # --- Custom two-section legend ---
    # Section 1: Baseline models (with their colors and markers)
    model_handles = []

    # Add RobuSeqNet and TReconLM
    h, = axs[1].plot([], [], marker=marker_dict["RobuSeqNet"], color=color_dict["RobuSeqNet"],
                    linestyle='-', linewidth=0.5, markersize=2, label=label_dict["RobuSeqNet"])
    model_handles.append(h)

    h, = axs[1].plot([], [], marker=marker_dict["TReconLM"], color=color_dict["TReconLM"],
                    linestyle='-', linewidth=0.5, markersize=2, label=label_dict["TReconLM"])
    model_handles.append(h)

    # Add GPT models (with gray markers)
    if any("gpt-4o" in key for key in gpt_data.keys()) or (gpt_cot_data and "gpt-4o-mini" in gpt_cot_data):
        h, = axs[1].plot([], [], marker=marker_dict["gpt-4o-mini"], color=GRAY_COLOR,
                        linestyle='none', markersize=2, label=r"\textsc{GPT-4o mini}")
        model_handles.append(h)

    if gpt5_data and "gpt-5" in gpt5_data:
        h, = axs[1].plot([], [], marker=marker_dict["gpt-5"], color=GRAY_COLOR,
                        linestyle='none', markersize=2, label=r"\textsc{GPT-5}")
        model_handles.append(h)

    # Section 2: Prompting strategies (colored lines and dashed gray for CoT)
    strategy_handles = []

    # Add colored lines for 0/3/5-shot
    for shot_type in ["0-shot", "3-shot", "5-shot"]:
        h, = axs[1].plot([], [], color=color_dict[shot_type], linestyle='-',
                        linewidth=0.5, label=shot_type)
        strategy_handles.append(h)

    # Add gray dashed line for CoT
    h, = axs[1].plot([], [], color=GRAY_COLOR, linestyle='--',
                    linewidth=0.5, label=r"CoT")
    strategy_handles.append(h)

    # Combine handles for legend
    all_handles = model_handles + strategy_handles

    # Create legend
    fig.legend(
        handles=all_handles,
        loc='upper center',
        ncol=len(all_handles),
        bbox_to_anchor=(0.5, 1.17),
        fontsize=8,
        handletextpad=0.3,
        columnspacing=0.6,
        handlelength=1.0,
        borderaxespad=0.2,
        frameon=False
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "openai_comparison.pdf"), bbox_inches='tight')
    plt.show()

# --- Main ---
if __name__ == "__main__":
    artifact_robu = "test_dataset_seed34721_gl110_bs1500_ds50000:latest"
    artifact_map_gpt = {
        2: "GPTMini_cz2_seed34721_gl60:latest",
        5: "GPTMini_cz5_seed34721_gl60:latest",
        10: "GPTMini_cz10_seed34721_gl60:latest",
    }

    print("Loading TReconLM and RobuSeqNet metrics...")
    metrics = load_treconlm_and_robu_metrics()

    print("Loading GPT-4o mini CoT metrics...")
    gpt_cot_data = load_gpt_cot_metrics(project_gpt_mini)

    print("Loading GPT-5 metrics...")
    gpt5_data = load_gpt5_metrics(project_gpt_mini)

    print("Computing noisy baselines...")
    noisy_robu = compute_noisy_baselines_from_artifact(artifact_robu)
    noisy_gpt = compute_noisy_baselines_from_artifact_map(artifact_map_gpt)

    print("Generating plot...")
    plot_combined(metrics, gpt_data, gpt_cot_data, gpt5_data, noisy_robu, noisy_gpt)
    print(f"Plot saved to {os.path.join(save_dir, 'openai_comparison.pdf')}")
