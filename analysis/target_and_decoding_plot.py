import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FuncFormatter
from collections import defaultdict
from tqdm import tqdm
import wandb
from Levenshtein import distance as levenshtein_distance
import matplotlib as mpl

# Add TReconLM root to path
sys.path.insert(0, "/workspaces/TReconLM")
from src.utils.hamming_distance import hamming_distance_postprocessed

# Settings 
FONTSIZE = 7.7
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{type1cm}',
    'font.size': FONTSIZE,
})

ENTITY = "<your.wandb.entity>"
PROJECT_ARTIFACT = "TRACE_RECONSTRUCTION"
PROJECT_WANDB = "Baselines"
PROJECT_BEAM = "BeamSearch"
PROJECT_REPRODUCE = "Reproduce"
PROJECT_MAJORITY = "MajorityVoting"
SAVE_DIR = "./plots"
DOWNLOAD_DIR = "./downloaded_artifact"
os.makedirs(SAVE_DIR, exist_ok=True)

# Colors and markers
color_dict = {
    "CPRED": "#6699CC",
    "NESTED": "#5ab4ac",
    "MSA": "#9D9AC5",
    "MUSCLE": "#F287BD",
    "Avg. noisy read": "#AAAAAA",
    "Best noisy read": "#AAAAAA",
    "TReconLM": "#4477AA",          # base color
    "CRED (beam=2)": "#9BB8E8",     # light blue
    "CRED (beam=4)": "#5C85C5",     # medium blue
    "CRED (beam=8)": "#9C8380",     # light pink/rose
    "MAJORITY": "#EFC8D5",          # pastel red (more opacity)
}
marker_dict = {
    "CPRED": "o",
    "NESTED": "^",
    "MSA": "v",
    "MUSCLE": "X",
    "TReconLM": "s",
    "CRED (beam=2)": "o",
    "CRED (beam=4)": "o",
    "CRED (beam=8)": "2",
    "MAJORITY": "4",
}
# Legend label mapping
label_map = {
    "MAJORITY": "+MV",
    "CRED (beam=2)": "+beam=2",
    "CRED (beam=4)": "+beam=4",
    "CRED (beam=8)": "+beam=8",
}

def get_algo_label(algo, dataset_tag):
    if dataset_tag == "microsoft" and algo == "TReconLM":
        return "TReconLM (p.)"
    return algo

def ensure_ticks(ax, axis="y", log=False):
    tgt = ax.xaxis if axis == "x" else ax.yaxis
    loc = MaxNLocator(min_n_ticks=2)
    tgt.set_major_locator(loc)
    if log:
        ax.set_yscale("log")
        tgt.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    else:
        fmt = ScalarFormatter(useMathText=False, useOffset=False)
        fmt.set_powerlimits((-3, 3))
        tgt.set_major_formatter(fmt)

def load_dataset_from_artifact(artifact_name):
    wandb.login()  # uses env token if available; safe to call repeatedly
    api = wandb.Api()
    art = api.artifact(f"{ENTITY}/{PROJECT_ARTIFACT}/{artifact_name}", type="dataset")
    d = art.download(DOWNLOAD_DIR)
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
    return [(c, len(c), gt) for c, gt in zip(clusters, gts)]

def safe_mean(x): return np.mean(x) if x else np.nan

def compute_noisy_baselines_from_artifact(artifact_name):
    data = load_dataset_from_artifact(artifact_name)
    g_hd, g_ld = defaultdict(list), defaultdict(list)
    m_hd, m_ld = defaultdict(list), defaultdict(list)
    for reads, n, gt in tqdm(data, desc=f"[Noisy baselines] {artifact_name}"):
        if not 2 <= n <= 10: continue
        L = len(gt)
        hs = [hamming_distance_postprocessed(gt, r) for r in reads]
        ls = [levenshtein_distance(gt, r) / L for r in reads]
        g_hd[n].append(np.mean(hs)); g_ld[n].append(np.mean(ls))
        m_hd[n].append(np.min(hs)); m_ld[n].append(np.min(ls))
    def stats(d): return (
        {n: safe_mean(d[n]) for n in range(2, 11)},
        {n: np.std(d[n]) for n in range(2, 11)},
    )
    avg_hd, std_hd = stats(g_hd)
    min_hd, _ = stats(m_hd)
    avg_ld, std_ld = stats(g_ld)
    min_ld, _ = stats(m_ld)
    return min_hd, avg_hd, std_hd, min_ld, avg_ld, std_ld

def load_metrics(min_hd, avg_hd, std_hd, min_ld, avg_ld, std_ld, filter_name):
    """Loads baseline runs from the 'Baselines' project filtered by name."""
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT_WANDB}")
    mets = defaultdict(lambda: defaultdict(dict))
    matched = [r for r in runs if r.state == "finished" and filter_name in r.name and "sweep" not in r.name]
    for run in matched:
        name = run.name
        if "MUSCLE" in name: algo = "MUSCLE"
        elif "nested" in name: algo = "NESTED"
        elif "MSA" in name: algo = "MSA"
        else: continue
        for k, v in run.summary.items():
            if isinstance(v, (int, float)) and "N=" in k:
                metric, n = k.split("_N=")
                try: mets[algo][metric][int(n)] = v
                except: continue
    # Attach noisy-read baselines
    mets["Best noisy read"]["avg_hamming"] = min_hd
    mets["Avg. noisy read"]["avg_hamming"] = avg_hd
    mets["Avg. noisy read"]["std_hamming"] = std_hd
    mets["Best noisy read"]["avg_levenshtein"] = min_ld
    mets["Avg. noisy read"]["avg_levenshtein"] = avg_ld
    mets["Avg. noisy read"]["std_levenshtein"] = std_ld
    return mets

def load_beams_by_tag(tags=("beam2", "beam4")):
    """
    Search BeamSearch project for finished runs that include each tag in `tags`.
    Returns dict: { 'beam2': mets, 'beam4': mets, 'beam6': mets },
    where each mets maps metric -> {N: value}.
    """
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT_BEAM}")
    out = {}
    for t in tags:
        # pick the most recent finished run with this tag
        candidates = [r for r in runs if r.state == "finished" and (r.tags and t in r.tags)]
        if not candidates:
            continue
        # choose the latest by created_at
        candidates.sort(key=lambda r: r.created_at, reverse=True)
        run = candidates[0]
        mets = defaultdict(dict)
        for k, v in run.summary.items():
            if isinstance(v, (int, float)) and "_N=" in k:
                metric, n = k.split("_N=")
                try:
                    mets[metric][int(n)] = v
                except:
                    pass
        out[t] = mets
    return out

def load_run_by_id(project, run_id):
    """Load a single run by W&B run ID and return metric dicts keyed by N."""
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{project}/{run_id}")
    mets = defaultdict(dict)
    for k, v in run.summary.items():
        if isinstance(v, (int, float)) and "_N=" in k:
            metric, n = k.split("_N=")
            mets[metric][int(n)] = v
    return mets

def load_run_by_name(project, run_name):
    """Load a single run by W&B run name and return metric dicts keyed by N."""
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{project}")
    matched = [r for r in runs if r.name == run_name and r.state == "finished"]
    if not matched:
        print(f"Warning: No finished run found with name '{run_name}' in {project}")
        return {}
    run = matched[0]  # take the first match
    mets = defaultdict(dict)
    for k, v in run.summary.items():
        if isinstance(v, (int, float)) and "_N=" in k:
            metric, n = k.split("_N=")
            mets[metric][int(n)] = v
    return mets

def plot_all_metrics(metrics, dataset_tag, show_std=False, plot_noisy_baselines=False):
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.1), dpi=300, gridspec_kw={'wspace': 0.4})
    Ns = list(range(2, 11))
    bar_width = 0.09
    x = np.array(Ns)

    # Define plotting order (exclude noisy references from this list)
    algo_order = [k for k in metrics if k not in ("Avg. noisy read", "Best noisy read")]

    # Line plot: average Levenshtein
    ax0 = axs[0]
    for algo in algo_order:
        if "avg_levenshtein" not in metrics[algo]:
            continue
        ys = [metrics[algo]["avg_levenshtein"].get(n, np.nan) for n in Ns]
        col = color_dict.get(algo)
        mk = marker_dict.get(algo, "o")

        if "beam" in algo or algo == "MAJORITY":
            # pick marker based on beam number in label
            if "beam=2" in algo:
                mk = "1"
            elif "beam=4" in algo:
                mk = "3"
            elif "beam=8" in algo:
                mk = "2"
            else:
                mk = marker_dict.get(algo, "o")  # fallback to marker_dict
            
            ax0.plot(
                Ns, ys,
                label=label_map.get(algo, algo),
                color=col,
                marker=mk,
                linestyle="--",
                linewidth=0.5,
                markersize=4
            )
        else:
            # Standard runs: solid line
            ax0.plot(
                Ns, ys,
                label=label_map.get(algo, algo),
                color=col,
                marker=mk,
                linestyle="-",
                linewidth=0.5,
                markersize=2
            )

    ax0.set_xlim(min(Ns) - 0.3, max(Ns) + 0.3)
    ax0.set_xticks(Ns)
    ax0.set_xlabel(r"Cluster size $N$")
    ax0.set_ylabel(r"$d_L$")
    ax0.set_yscale("log")
    ax0.set_yticks([0.001, 0.01, 0.1])
    ax0.set_yticklabels(["0.001", "0.01", "0.1"])

    # Bar plot: failure rate 
    ax1 = axs[1]
    preferred_order = ["MUSCLE", "MSA", "NESTED", "CPRED",
                       "CRED (beam=2)", "CRED (beam=4)", "CRED (beam=8)",
                       "MAJORITY", "TReconLM"]

    for i, n in enumerate(Ns):
        failures = [
            (algo, 1 - metrics[algo].get("success_rate", {}).get(n, 0))
            for algo in preferred_order
            if algo in metrics and n in metrics[algo].get("success_rate", {})
        ]
        for j, (algo, fail_rate) in enumerate(failures):
            offset = (j - (len(failures) - 1) / 2) * bar_width

            # hatch beams as dashed fill
            if "beam" in algo:
                hatch='//////////'
                facecolor = "white"   # white fill
                edgecolor = color_dict.get(algo)
                lw = 0.1
            else:
                hatch = None
                facecolor = color_dict.get(algo)
                edgecolor = color_dict.get(algo)
                lw = 0.1

            ax1.bar(
                x[i] + offset,
                fail_rate,
                width=bar_width,
                color=facecolor,
                edgecolor=edgecolor,
                hatch=hatch,
                linewidth=lw,
                zorder=2,
                label=label_map.get(algo, algo) if i == 0 else None,
            )
                

    ax1.set_xticks(x)
    ax1.set_xticklabels(Ns)
    ax1.set_xlim(min(Ns) - 0.5, max(Ns) + 0.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_xlabel(r"Cluster size $N$")
    ax1.set_ylabel("Failure rate")

    # Styling (light spines, standard ticks)
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_color('lightgray')
        ax.tick_params(axis='both', which='both', color='lightgray', labelcolor='black')

    # Legend (unique labels across both subplots)
    handles0, labels0 = axs[0].get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles0, labels0):
        if l not in unique:
            unique[l] = h
    handles, labels = list(unique.values()), list(unique.keys())

    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.12),
        fontsize=6.5,
        handletextpad=0.3,
        columnspacing=0.6,
        handlelength=1.5,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(SAVE_DIR, "targets.pdf"), bbox_inches='tight')
    plt.show()
    plt.close(fig)

def main():
    show_std = False
    configs = [
        ("gl110", "test_dataset_seed34721_gl110_bs1500_ds50000:latest"),
    ]
    for tag, artifact in configs:
        # Noisy read references from artifact
        min_hd, avg_hd, std_hd, min_ld, avg_ld, std_ld = compute_noisy_baselines_from_artifact(artifact)
        metrics = load_metrics(min_hd, avg_hd, std_hd, min_ld, avg_ld, std_ld, filter_name=tag)

        # CPRED from Reproduce project by run ID
        cpred_run_id = "vi8obds0"
        cpred_mets = load_run_by_id(PROJECT_REPRODUCE, cpred_run_id)
        metrics["CPRED"] = {
            "avg_levenshtein": cpred_mets.get("avg_levenshtein", {}),
            "success_rate": cpred_mets.get("success_rate", {}),
        }

        # Beam runs from BeamSearch by tags beam2/beam4
        beams = load_beams_by_tag(tags=("beam2", "beam4"))
        tag_to_label = {"beam2": "CRED (beam=2)", "beam4": "CRED (beam=4)"}
        for t, mets in beams.items():
            if t not in tag_to_label:
                continue
            label = tag_to_label[t]
            metrics[label] = {
                "avg_levenshtein": mets.get("avg_levenshtein", {}),
                "success_rate": mets.get("success_rate", {}),
            }

        # Beam=8 from BeamSearch project
        beam8_run_name = "TReconLM_inference_20251117_051512_beam8"
        beam8_mets = load_run_by_name(PROJECT_BEAM, beam8_run_name)
        if beam8_mets:
            metrics["CRED (beam=8)"] = {
                "avg_levenshtein": beam8_mets.get("avg_levenshtein", {}),
                "success_rate": beam8_mets.get("success_rate", {}),
            }

        # Majority voting from MajorityVoting project
        majority_run_name = "TReconLM_inference_20251116_224316_majority_first_prediction_final"
        majority_mets = load_run_by_name(PROJECT_MAJORITY, majority_run_name)
        if majority_mets:
            # Convert failure_rate to success_rate for consistency with other methods
            failure_rates = majority_mets.get("majority_voting_voted_failure_rate", {})
            success_rates = {n: 1 - fr for n, fr in failure_rates.items()}
            metrics["MAJORITY"] = {
                "avg_levenshtein": majority_mets.get("majority_voting_voted_levenshtein", {}),
                "success_rate": success_rates,
            }

        plot_all_metrics(metrics, dataset_tag=tag, show_std=show_std)

if __name__ == "__main__":
    main()
