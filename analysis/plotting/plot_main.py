# Run from repo root: cd /path/to/TReconLM && python -m analysis.plotting.plot_main
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from collections import defaultdict
from tqdm import tqdm
import wandb
from Levenshtein import distance as levenshtein_distance

# Add TReconLM root to path
sys.path.insert(0, "/workspaces/TReconLM")
from analysis.plot_config import (
    setup_latex_plots, ALGO_COLORS as color_dict, ALGO_MARKERS as marker_dict,
    ALGO_LABELS, WANDB_ENTITY as ENTITY, SAVE_DIR, DOWNLOAD_DIR,
)
from src.utils.hamming_distance import hamming_distance_postprocessed


# Font size per dataset: synthetic (L=60/110/180) use 10.3, real datasets use 11.2
FONTSIZE_BY_DATASET = {
    "gl60": 10.3,
    "gl110": 10.3,
    "gl180": 10.3,
    "microsoft": 11.2,
    "noisy": 11.2,
    "chandak": 11.2,
}
DEFAULT_FONTSIZE = 10.3

LINEWIDTH = 0.6
SHADE_ALPHA = 0.25
ERRORBAR_LINEWIDTH = 0.4
ERRORBAR_CAPSIZE = 1.5
ERRORBAR_ALPHA = 1

setup_latex_plots(fontsize=DEFAULT_FONTSIZE)

# W&B config 
PROJECT_ARTIFACT = "TRACE_RECONSTRUCTION"
PROJECT_WANDB_BASELINES = "Baselines"    
PROJECT_WANDB_REPRODUCE = "Reproduce"     # TReconLM pretrained (synthetic)
PROJECT_WANDB_MIC = "FinetuneMicrosoft"   # Microsoft finetuned + pretrained
PROJECT_WANDB_NOISY = "FinetuneNoisyDNA"  # Noisy finetuned + pretrained
PROJECT_WANDB_CHANDAK = "chandak"         # Chandak real DNA dataset

os.makedirs(SAVE_DIR, exist_ok=True)

# Colors / markers 

DATASET_TO_NT = {
    "gl60":  "60nt",
    "gl110": "110nt",
    "gl180": "180nt",
}

# Helpers 
def safe_mean(x):
    return np.mean(x) if len(x) > 0 else np.nan

def load_dataset_from_artifact(artifact_name):
    wandb.login()
    api = wandb.Api()
    art = api.artifact(f"{ENTITY}/{PROJECT_ARTIFACT}/{artifact_name}", type="dataset")
    d = art.download(DOWNLOAD_DIR)
    with open(os.path.join(d, "reads.txt")) as f:
        reads = [l.strip() for l in f]
    with open(os.path.join(d, "ground_truth.txt")) as f:
        gts = [l.strip() for l in f if l.strip()]
    clusters, cur = [], []
    for line in reads:
        if line.startswith("="):
            if cur: clusters.append(cur)
            cur = []
        elif line:
            cur.append(line)
    if cur: clusters.append(cur)
    assert len(clusters) == len(gts)
    return [(c, len(c), gt) for c, gt in zip(clusters, gts)]

def compute_noisy_baselines_from_artifact(artifact_name):
    data = load_dataset_from_artifact(artifact_name)
    g_ld, m_ld = defaultdict(list), defaultdict(list)
    for reads, n, gt in tqdm(data, desc=f"[Noisy baselines] {artifact_name}"):
        if not 2 <= n <= 10:
            continue
        L = len(gt)
        ls = [levenshtein_distance(gt, r) / L for r in reads]
        g_ld[n].append(np.mean(ls))
        m_ld[n].append(np.min(ls))
    def stats(d):
        return (
            {n: safe_mean(d[n]) for n in range(2, 11)},
            {n: np.std(d[n])   for n in range(2, 11)},
        )
    avg_ld, std_ld = stats(g_ld)
    min_ld, _      = stats(m_ld)
    return min_ld, avg_ld, std_ld

def _collect_numeric_metrics_from_run_summary(run_summary):
    """
      Matches any of:
        avg_levenshtein_N=5
        avg_levenshtein_cropped_N=5  
        success_rate_N=5
        success_rate_cropped_N=5      
    Returns: dict like {'avg_levenshtein': {5: ...}, 'success_rate': {5: ...}}
    """
    out = defaultdict(dict)
    for k, v in run_summary.items():
        if not isinstance(v, (int, float)):
            continue
        m = re.match(r"^(avg_levenshtein|success_rate)(?:_cropped)?_N=(\d+)$", k)
        if m:
            base, n = m.group(1), int(m.group(2))
            out[base][n] = float(v)
    return out

# Baselines loader
def load_metrics_baselines(avg_ld, min_ld, filter_name):
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT_WANDB_BASELINES}")
    mets = defaultdict(lambda: defaultdict(dict))
    matched = [r for r in runs if r.state == "finished" and filter_name in r.name and "sweep" not in r.name]
    used_runs = defaultdict(list)
    for run in matched:
        name = run.name
        if "TReconLM" in name:   # exclude any TReconLM from Baselines
            continue
        if "MUSCLE" in name: algo = "MUSCLE"
        elif "VS" in name: algo = "VS"
        elif "BMALA" in name: algo = "BMALA"
        elif "TrellisBMA" in name: algo = "TrellisBMA"
        elif "Iterative" in name or "ITR" in name: algo = "ITR"
        elif "Robseqnet" in name or "RobuSeqNet" in name: algo = "RobuSeqNet"
        elif "DNAformer" in name: algo = "DNAformer"
        else:
            continue
        used_runs[algo].append(run.name)
        numeric = _collect_numeric_metrics_from_run_summary(run.summary)
        for metric, d in numeric.items():
            for n, val in d.items():
                mets[algo].setdefault(metric, {})[n] = val

    # Optional noisy references (if you want to draw them later)
    mets["Best noisy read"]["avg_levenshtein"] = min_ld
    mets["Avg. noisy read"]["avg_levenshtein"] = avg_ld

    print(f"\n Baselines W&B runs for dataset '{filter_name}' (NO TReconLM from Baselines) ")
    for algo, run_names in used_runs.items():
        print(f"{algo}:")
        for rname in run_names:
            print(f"  - {rname}")
    print()
    return mets

# Chandak loaders
def load_metrics_chandak_baselines():
    """Load baseline algorithms from the chandak project (no TReconLM)."""
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT_WANDB_CHANDAK}")
    metrics = defaultdict(lambda: defaultdict(dict))

    # Map run names to algorithm labels 
    run_mapping = {
        "muscle_local_data": "MUSCLE",
        "vs_local_data": "VS",
        "trellisbma_local_data": "TrellisBMA",
        "bmala_local_data": "BMALA",
        "itr_local_data": "ITR",
    }

    used_runs = []
    for run in runs:
        if run.state != "finished":
            continue

        # Check if this run matches any mappings
        algo_label = run_mapping.get(run.name)
        if not algo_label:
            continue

        used_runs.append((algo_label, run.name))

        # Extract metrics using the helper
        numeric = _collect_numeric_metrics_from_run_summary(run.summary)
        for metric, d in numeric.items():
            for n, val in d.items():
                metrics[algo_label].setdefault(metric, {})[n] = val

    print(f"\n Chandak baseline W&B runs ")
    for algo, run_name in used_runs:
        print(f"{algo}: {run_name}")
    print()

    return metrics

def load_treconlm_chandak():
    """
    Load TReconLM runs from chandak project.
    Returns (finetuned_mets, pretrained_mets) in the same format as other finetune loaders.
    """
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT_WANDB_CHANDAK}")

    finetuned_run_name = "TReconLM_inference_20251119_115632_finetuned"
    pretrained_run_name = "TReconLM_inference_20251118_123018_117nt_chandak_pretr"

    finetune_mets = {"avg_levenshtein": {"mean": {}, "std": {}}, "success_rate": {"mean": {}, "std": {}}}
    pretrained_mets = {"avg_levenshtein": {"mean": {}, "std": {}}, "success_rate": {"mean": {}, "std": {}}}

    for run in runs:
        if run.state != "finished":
            continue

        numeric = _collect_numeric_metrics_from_run_summary(run.summary)

        if run.name == finetuned_run_name:
            print(f"\n Chandak finetuned TReconLM ")
            print(f"  - {run.name}")
            for metric in ("avg_levenshtein", "success_rate"):
                for n, val in numeric.get(metric, {}).items():
                    finetune_mets[metric]["mean"][n] = val
                    finetune_mets[metric]["std"][n] = 0.0  # Single run, no std

        elif run.name == pretrained_run_name:
            print(f"\n Chandak pretrained TReconLM ")
            print(f"  - {run.name}")
            for metric in ("avg_levenshtein", "success_rate"):
                for n, val in numeric.get(metric, {}).items():
                    pretrained_mets[metric]["mean"][n] = val
                    pretrained_mets[metric]["std"][n] = 0.0  # Single run, no std

    return finetune_mets, pretrained_mets

# Reproduce loader (pretrained TReconLM aggregated across seeds)
def load_treconlm_reproduce_mean_std_by_nt(nt_str):
    """
    Match runs in Reproduce project by name substring:
      'TReconLM_inference_*_final_{nt_str}_reproduce' with optional '_seed\\d+'
    Aggregate mean / std across all matched runs (pretrained only).
    """
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT_WANDB_REPRODUCE}")

    name_sub = f"final_{nt_str}_reproduce"
    name_regex = re.compile(rf"^TReconLM_inference_\d+_final_{re.escape(nt_str)}_reproduce(_seed\d+)?$")

    candidates = []
    for r in runs:
        if r.state != "finished":
            continue
        if name_sub not in r.name:
            continue
        candidates.append(r)

    print(f"\n TReconLM Reproduce (pretrained) matches for {nt_str} ")
    for r in candidates:
        print("  -", r.name, "tags:", r.tags)

    per_metric_vals = {
        "avg_levenshtein": defaultdict(list),
        "success_rate": defaultdict(list),
    }
    for run in candidates:
        numeric = _collect_numeric_metrics_from_run_summary(run.summary)
        for metric in ("avg_levenshtein", "success_rate"):
            for n, val in numeric.get(metric, {}).items():
                per_metric_vals[metric][n].append(val)

    out = {}
    for metric in ("avg_levenshtein", "success_rate"):
        mean_d, std_d = {}, {}
        for n in range(2, 10+1):
            arr = np.array(per_metric_vals[metric].get(n, []), dtype=float)
            if arr.size == 0:
                mean_d[n] = np.nan
                std_d[n]  = 0.0
            else:
                mean_d[n] = float(np.mean(arr))
                std_d[n]  = float(np.std(arr))
        out[metric] = {"mean": mean_d, "std": std_d}
    return out

# Finetune loaders (MICROSOFT / NOISY)
def _aggregate_runs_by_tags(project, needed_tags):
    """
    From a project, pick all finished runs that contain ANY of the 'needed_tags'
    (we group tags by logical group outside), then aggregate mean/std.
    Returns dict: metric -> {"mean": {N:..}, "std": {N:..}}
    """
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{project}")
    selected = [r for r in runs if r.state == "finished" and r.tags and any(t in r.tags for t in needed_tags)]
    print(f"\n[{project}] matched runs for tags {needed_tags}:")
    for r in selected:
        print("  -", r.name, "tags:", r.tags)

    per_metric_vals = {
        "avg_levenshtein": defaultdict(list),
        "success_rate": defaultdict(list),
    }
    for run in selected:
        numeric = _collect_numeric_metrics_from_run_summary(run.summary)
        for metric in ("avg_levenshtein", "success_rate"):
            for n, val in numeric.get(metric, {}).items():
                per_metric_vals[metric][n].append(val)

    out = {}
    for metric in ("avg_levenshtein", "success_rate"):
        mean_d, std_d = {}, {}
        for n in range(2, 11):
            arr = np.array(per_metric_vals[metric].get(n, []), dtype=float)
            if arr.size == 0:
                mean_d[n] = np.nan
                std_d[n]  = 0.0
            else:
                mean_d[n] = float(np.mean(arr))
                std_d[n]  = float(np.std(arr))
        out[metric] = {"mean": mean_d, "std": std_d}
    return out

def load_treconlm_finetune(dataset_tag):
    """
    For 'microsoft': use project FinetuneMicrosoft
      - finetuned runs: tags in {'finetune_mic', 'finetune_mic_seed1', 'finetune_mic_seed42'}
      - pretrained:     tag in {'pretr_mic'}  (no averaging; we treat as (p.))
    For 'noisy': use project FinetuneNoisyDNA
      - finetuned runs: tags in {'finetune_noisy_ratio_full', 'finetune_noisy_ratio_full_seed1', 'finetune_noisy_ratio_full_seed42'}
      - pretrained:     tag in {'pretrained_noisy'}
    Returns: (finetune_mets, pretrained_mets)
      where each is dict metric -> {"mean": {...}, "std": {...}} for finetune,
      and for pretrained we map into the same structure with std=0 (single run).
    """
    if dataset_tag == "microsoft":
        proj = PROJECT_WANDB_MIC
        finetune_tags = {"finetune_mic", "finetune_mic_seed1", "finetune_mic_seed42"}
        pretrained_tags = {"pretr_mic"}
    elif dataset_tag == "noisy":
        proj = PROJECT_WANDB_NOISY
        finetune_tags = {"finetune_noisy_ratio_full", "finetune_noisy_ratio_full_seed1", "finetune_noisy_ratio_full_seed42"}
        pretrained_tags = {"pretrained_noisy"}
    else:
        return None, None

    # Finetune: aggregate across 3 runs
    finetune_mets = _aggregate_runs_by_tags(proj, finetune_tags)

    # Pretrained: expect single (or a few) run(s); aggregate anyway (std likely 0)
    pretrained_mets = _aggregate_runs_by_tags(proj, pretrained_tags)
    return finetune_mets, pretrained_mets

# log-safe fill for log y-scale
def _fill_between_logsafe(ax, xs, mean, std, color, alpha=SHADE_ALPHA, eps=1e-8, zorder=2):
    mean = np.asarray(mean, dtype=float)
    std  = np.asarray(std, dtype=float)
    lower = np.maximum(mean - std, eps)
    upper = np.maximum(mean + std, eps)
    ax.fill_between(xs, lower, upper, alpha=alpha, color=color, zorder=zorder)

# Plotting
def plot_all_metrics(metrics, dataset_tag, treconlm=None, tre_finetune=None, tre_pretrained=None, plot_noisy_baselines=False):
    # Set fontsize for this dataset
    fontsize = FONTSIZE_BY_DATASET.get(dataset_tag, DEFAULT_FONTSIZE)
    setup_latex_plots(fontsize=fontsize)

    #  dynamic size to compensate for longer legend on microsoft/noisy
    base_w, base_h = 7, 1.37
    if dataset_tag in ("microsoft", "noisy", "chandak"):
        fig_w, fig_h = 7.7, 1.42   # wider + taller for long legends
    else:
        fig_w, fig_h = base_w, base_h

    fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=300, gridspec_kw={'wspace': 0.4})
    Ns = list(range(2, 11))
    bar_width = 0.10
    x = np.array(Ns)

    #  Left: average Levenshtein 
    ax_l = axs[0]
    algo_order = [k for k in metrics if k not in ("Avg. noisy read", "Best noisy read")]
    if dataset_tag == "microsoft":
        metrics.pop("VS", None)  # Microsoft legend without VS
        algo_order = [k for k in metrics if k not in ("Avg. noisy read", "Best noisy read")]
    for algo in algo_order:
        if "avg_levenshtein" not in metrics[algo]:
            continue
        ys = [metrics[algo]["avg_levenshtein"].get(n, np.nan) for n in Ns]
        col = color_dict.get(algo, "#888888")
        mk  = marker_dict.get(algo, "o")
        ax_l.plot(Ns, ys, label=algo, color=col, marker=mk, linestyle="-",
                  linewidth=LINEWIDTH, markersize=2)

    # Synthetic datasets: show pretrained TReconLM mean + shaded std
    if treconlm is not None:
        mean_y = [treconlm["avg_levenshtein"]["mean"].get(n, np.nan) for n in Ns]
        std_y  = [treconlm["avg_levenshtein"]["std"].get(n, 0.0) for n in Ns]
        col    = color_dict["TReconLM"]
        ax_l.plot(Ns, mean_y, label="TReconLM", color=col, marker="o",
                  linestyle="-", linewidth=LINEWIDTH, markersize=2.2, zorder=3)
        _fill_between_logsafe(ax_l, Ns, mean_y, std_y, color=col, alpha=SHADE_ALPHA, zorder=2)

    # Microsoft / Noisy: show finetuned and pretrained (dashed)
    if tre_finetune is not None and tre_pretrained is not None:
        # finetuned solid
        ft_mean = [tre_finetune["avg_levenshtein"]["mean"].get(n, np.nan) for n in Ns]
        ft_std  = [tre_finetune["avg_levenshtein"]["std"].get(n, 0.0) for n in Ns]
        col = color_dict["TReconLM"]
        ax_l.plot(Ns, ft_mean, label="TReconLM", color=col, marker="o",
                  linestyle="-", linewidth=LINEWIDTH, markersize=2.2, zorder=4)
        _fill_between_logsafe(ax_l, Ns, ft_mean, ft_std, color=col, alpha=SHADE_ALPHA, zorder=3)

        # pretrained dashed (same color)
        pt_mean = [tre_pretrained["avg_levenshtein"]["mean"].get(n, np.nan) for n in Ns]
        ax_l.plot(Ns, pt_mean, label="TReconLM (p.)", color=col, marker="o",
                  linestyle="--", linewidth=LINEWIDTH, markersize=2, zorder=3)

    # Optional noisy refs
    if plot_noisy_baselines:
        for b in ["Avg. noisy read", "Best noisy read"]:
            if "avg_levenshtein" not in metrics[b]:
                continue
            ys = [metrics[b]["avg_levenshtein"].get(n, np.nan) for n in Ns]
            ls = ":" if b == "Avg. noisy read" else "--"
            ax_l.plot(Ns, ys, linestyle=ls, color=color_dict[b], linewidth=LINEWIDTH, alpha=1.0, zorder=1)

    # Axes styling (match your previous)
    ax_l.set_xlim(min(Ns) - 0.3, max(Ns) + 0.3)
    ax_l.set_xticks(Ns)
    ax_l.set_xlabel(r"Cluster size $N$")
    ax_l.set_yscale("log")
    if dataset_tag == 'chandak':
        ax_l.set_yticks([0.02, 0.05, 0.1, 0.2, 0.3]); ax_l.set_yticklabels(["0.02","0.05","0.1","0.2","0.3"])
    elif dataset_tag == 'noisy':
        ax_l.set_yticks([0.2, 0.3, 0.4, 0.6]); ax_l.set_yticklabels(["0.2","0.3","0.4","0.6"])
    else:
        ax_l.set_yticks([0.001, 0.01, 0.1]);   ax_l.set_yticklabels(["0.001","0.01","0.1"])
    ax_l.set_ylabel(r"$d_L$")
    ax_l.yaxis.set_label_coords(-0.2, 0.5)

    #  Right: failure rate bars 
    ax_r = axs[1]

    # Use standard order (RobuSeqNet and DNAformer will be filtered out if not present)
    base_order = ["RobuSeqNet", "VS", "MUSCLE", "BMALA", "TrellisBMA", "ITR", "DNAformer"]

    # Build the expected order depending on dataset
    if tre_finetune is not None and tre_pretrained is not None:
        # Microsoft/Noisy: baselines (without DNAformer), then (p.), then DNAformer, then finetuned
        bar_order = [a for a in base_order if a != "DNAformer" and a in metrics] \
                    + ["TReconLM (p.)", "DNAformer", "TReconLM"]
    elif treconlm is not None:
        bar_order = [a for a in base_order if a in metrics] + ["TReconLM"]
    else:
        bar_order = [a for a in base_order if a in metrics]

    for i, n in enumerate(Ns):
        failures = []

        # collect failures (order doesn't matter now; we'll sort by bar_order)
        for algo in base_order:
            sr_dict = metrics.get(algo, {}).get("success_rate", {})
            if n in sr_dict:
                failures.append((algo, 1.0 - sr_dict[n], 0.0, "solid"))

        if tre_finetune is not None and tre_pretrained is not None:
            pt_sr_mean = tre_pretrained["success_rate"]["mean"].get(n, np.nan)
            if not np.isnan(pt_sr_mean):
                failures.append(("TReconLM (p.)", 1.0 - pt_sr_mean, 0.0, "hatched"))

            ft_sr_mean = tre_finetune["success_rate"]["mean"].get(n, np.nan)
            ft_sr_std  = tre_finetune["success_rate"]["std"].get(n, 0.0)
            if not np.isnan(ft_sr_mean):
                failures.append(("TReconLM", 1.0 - ft_sr_mean, ft_sr_std, "solid"))
        elif treconlm is not None:
            sr_mean = treconlm["success_rate"]["mean"].get(n, np.nan)
            sr_std  = treconlm["success_rate"]["std"].get(n, 0.0)
            if not np.isnan(sr_mean):
                failures.append(("TReconLM", 1.0 - sr_mean, sr_std, "solid"))

        # Enforce the desired order exactly
        failures_sorted = sorted(
            [f for f in failures if f[0] in bar_order],
            key=lambda t: bar_order.index(t[0])
        )

        for j, (algo, fail_rate, sr_std, style) in enumerate(failures_sorted):
            offset = (j - (len(failures_sorted) - 1) / 2) * bar_width
            x_pos = x[i] + offset
            col = color_dict.get(algo, "#888888")

            if algo == "TReconLM (p.)" and (tre_finetune is not None and tre_pretrained is not None):
                ax_r.bar(
                    x_pos, fail_rate, width=bar_width,
                    color="white", edgecolor=col, hatch="//////////", linewidth=0.1, zorder=2
                )
            else:
                ax_r.bar(
                    x_pos, fail_rate, width=bar_width,
                    color=col, edgecolor=None, linewidth=0, zorder=2
                )
                if algo == "TReconLM" and sr_std > 0:
                    ax_r.errorbar(
                        x_pos, fail_rate, yerr=sr_std,
                        fmt='none', ecolor='black', alpha=ERRORBAR_ALPHA,
                        elinewidth=ERRORBAR_LINEWIDTH,
                        capsize=ERRORBAR_CAPSIZE, capthick=ERRORBAR_LINEWIDTH,
                        zorder=3
                    )

    # Axes styling (match your previous)
    ax_r.set_xticks(x); ax_r.set_xticklabels(Ns)
    ax_r.set_xlim(min(Ns) - 0.5, max(Ns) + 0.5)
    if dataset_tag == 'noisy':
        ax_r.set_ylim(0.6, 1.05); ax_r.set_yticks([0.6, 0.8, 1.0])
    else:
        ax_r.set_ylim(0, 1.05);   ax_r.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_r.set_xlabel(r"Cluster size $N$")
    ax_r.set_ylabel("Failure rate")

    # shared styling
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_color('lightgray')
        ax.tick_params(axis='both', which='both', color='lightgray', labelcolor='black')

    #  Legend (no duplicates, fixed order) 
    uniq = {}
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and label not in uniq:
                uniq[label] = handle

    # Use standard legend order
    base_order_for_legend = ["RobuSeqNet","VS","MUSCLE","BMALA","TrellisBMA","ITR"]
    if tre_finetune is not None and tre_pretrained is not None:
        # Pretrained first, then DNAformer, then finetuned
        preferred = base_order_for_legend + ["TReconLM (p.)", "DNAformer", "TReconLM"]
    else:
        # Only one TReconLM (synthetic)
        preferred = base_order_for_legend + ["DNAformer", "TReconLM"]

    legend_order = [lbl for lbl in preferred if lbl in uniq]
    fig = axs[0].get_figure()
    fig.legend(
        [uniq[l] for l in legend_order],
        legend_order,
        loc='upper center',
        ncol=min(len(legend_order), 10),
        bbox_to_anchor=(0.5, 1.15),
        fontsize=8.5,
        handletextpad=0.3,
        columnspacing=0.6,
        handlelength=1.5,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(SAVE_DIR, f"all_metrics_{dataset_tag}.pdf")
    plt.savefig(out, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f" Saved {out}")

# Main
def main():
    configs = [
        ("gl60",  "test_dataset_seed34721_gl60_bs800_ds50000:latest"),
        ("gl110", "test_dataset_seed34721_gl110_bs1500_ds50000:latest"),
        ("gl180", "test_dataset_seed34721_gl180_bs2400_ds50000:latest"),
        ("microsoft", "Microsoft-test-20250502_132818:latest"),
        ("noisy", "File1-test-20250509_223718:latest"),
        ("chandak", None),  # No artifact needed for Chandak
    ]
    for tag, artifact in configs:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {tag}")
        print(f"{'='*60}")

        # Special handling for Chandak (no artifact, no noisy baselines)
        if tag == "chandak":
            metrics = load_metrics_chandak_baselines()
            tre = None
            tre_ft, tre_pt = load_treconlm_chandak()
        else:
            # 1) Baselines (NO TReconLM)
            min_ld, avg_ld, std_ld = compute_noisy_baselines_from_artifact(artifact)
            metrics = load_metrics_baselines(avg_ld, min_ld, filter_name=tag)

            # 2) Synthetic: TReconLM (p.) from Reproduce by nt
            tre = None
            tre_ft = None
            tre_pt = None

            if tag in DATASET_TO_NT:
                nt_str = DATASET_TO_NT[tag]
                tre = load_treconlm_reproduce_mean_std_by_nt(nt_str)

            # 3) Microsoft / Noisy: finetune + pretrained from their projects
            if tag == "microsoft":
                tre_ft, tre_pt = load_treconlm_finetune("microsoft")
            elif tag == "noisy":
                tre_ft, tre_pt = load_treconlm_finetune("noisy")

        # 4) Plot
        plot_all_metrics(
            metrics,
            dataset_tag=tag,
            treconlm=tre,
            tre_finetune=tre_ft,
            tre_pretrained=tre_pt,
            plot_noisy_baselines=False
        )





#!/usr/bin/env python3
"""
Combined plot script: sweep metrics + misclustering line plot + failure rate bars
Styled to match the scaling laws plot
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm, ticker
from matplotlib.ticker import LogLocator
from collections import defaultdict
import wandb
from Levenshtein import distance as levenshtein_distance
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Settings - matching the scaling laws plot
fontsize = 7
setup_latex_plots(fontsize=fontsize)

# config 
projects = ["Timing", "Baselines", "Inference"]
artifact_project = "TRACE_RECONSTRUCTION"
misclustering_project = "Misclustering"
misclustering_run_id = "p2t1ikjy"  # For TReconLM
save_dir = "./plots"
download_dir = "./downloaded_artifact"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(download_dir, exist_ok=True)

# Hardcoded baselines for misclustering plot (per algorithm)
misclustering_baselines = {
    "TReconLM": 0.033527454545454534, 
    "VS": 0.16626545454545455,
    "ITR": 0.06497927272727273,
    "MUSCLE": 0.09353327272727271,
    "BMALA": 0.11718872727272726,
    "TrellisBMA": 0.1578189090909091,
    "DNAformer": 0.06976618181818182,  
    "RobuSeqNet": 0.26173787416  
}

# Algorithms to plot in misclustering subplot
misclustering_algos = ["TReconLM", "ITR", "MUSCLE", "BMALA", "TrellisBMA", "VS", "DNAformer", "RobuSeqNet"]

# Algorithms that should merge data from multiple runs (instead of replacing)
MERGE_ALGOS = {"TrellisBMA"}

# colors & markers (consistent with Cell 1)
marker_dict = {
    "TReconLM": "o",
    "TReconLM (untrained)": "o",
    "TReconLM (trained)": "o",
    "ITR": "s",
    "DNAformer": "^",
    "TrellisBMA": "v",
    "BMALA": "D",
    "RobuSeqNet": "P",
    "MUSCLE": "X",
    "VS": "*",
}

def truncate_colormap(cmap, minval=0.2, maxval=1.0, n=100):
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval},{maxval})",
        cmap(np.linspace(minval, maxval, n))
    )

def parse_k(name):
    m = re.search(r"sweep[_=]?(\d+)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"_k=?(\d+)\b", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def get_algo(name):
    """Extract algorithm name from run name (case-insensitive)."""
    name_lower = name.lower()
    if "vs_algorithm" in name_lower or "vs" in name_lower.split("_"):
        return "VS"
    if "muscle" in name_lower:
        return "MUSCLE"
    if "bmala" in name_lower:
        return "BMALA"
    if "trellisbma" in name_lower:
        return "TrellisBMA"
    if "iterative_algorithm" in name_lower or "itr" in name_lower:
        return "ITR"
    if "robseqnet" in name_lower or "robuseqnet" in name_lower:
        return "RobuSeqNet"
    if "dnaformer" in name_lower:
        return "DNAformer"
    if "pretr" in name_lower or "finet" in name_lower or "treconlm" in name_lower:
        return "TReconLM"
    return None

def fetch_sweep_metrics():
    api = wandb.Api()
    runs = []
    for proj in projects:
        runs.extend(api.runs(f"{ENTITY}/{proj}", filters={"state":"finished"}, per_page=200))

    metrics = defaultdict(lambda: defaultdict(dict))
    run_names = []

    for run in runs:
        if "sweep" not in run.name.lower():
            continue
        if "fixed" in run.name.lower() or "alln" in run.name.lower():
            continue
        if "with_k" in run.tags:
            continue

        algo = get_algo(run.name)
        if not algo:
            continue

        run_names.append(run.name)
        print(f"  [Sweep] Using run: {run.name}  (algo={algo}, project={run.project})")
        k0 = parse_k(run.name)

        for key, val in run.summary.items():
            if not isinstance(val, (int, float)):
                continue

            raw_k = parse_k(key)
            k = raw_k if raw_k is not None else k0

            if k is None:
                continue

            if "avg_levenshtein" in key and "all" in key:
                metrics[algo]["avg_levenshtein"][k] = val
                continue

            if "avg_levenshtein" in key and "std" not in key:
                metrics[algo]["avg_levenshtein"].setdefault(k, val)

            if "std_levenshtein" in key:
                metrics[algo]["std_levenshtein"][k] = val

    return metrics, run_names

def compute_noisy_baselines():
    api = wandb.Api()
    avg_ld = defaultdict(list)
    min_ld = defaultdict(list)
    for k in range(11):
        art = api.artifact(
            f"{ENTITY}/{artifact_project}/sweep{k}_seed{34721+k}_gl110_bs1500_ds5000:latest",
            type="dataset"
        )
        d = art.download(download_dir)
        reads = open(f"{d}/reads.txt").read().splitlines()
        gts   = open(f"{d}/ground_truth.txt").read().splitlines()
        clusters,cur = [],[]
        for line in reads:
            if line.startswith("="):
                if cur: clusters.append(cur)
                cur=[]
            else:
                cur.append(line)
        if cur: clusters.append(cur)
        for cl,gt in zip(clusters,gts):
            L = len(gt)
            ls = [levenshtein_distance(gt,r)/L for r in cl]
            avg_ld[k].append(np.mean(ls))
            min_ld[k].append(np.min(ls))
    return {
        "Avg. noisy read":  {"avg_levenshtein": {k:np.mean(avg_ld[k]) for k in avg_ld}},
        "Best noisy read": {"avg_levenshtein": {k:np.mean(min_ld[k]) for k in min_ld}}
    }

def fetch_fixedN_failure():
    api = wandb.Api()
    runs = []
    for proj in projects:
        runs.extend(api.runs(f"{ENTITY}/{proj}", filters={"state":"finished"}, per_page=200))

    failure = defaultdict(dict)
    for run in runs:
        m = re.search(r"fixed[Nn][ _=]?(\d+)", run.name)
        if m:
            N = int(m.group(1))
        elif re.search(r"allN", run.name, flags=re.IGNORECASE):
            N = "allN"
        else:
            continue
        
        print(f"  [FixedN] Using run: {run.name}  (N={N}, project={run.project})")
        for key, val in run.summary.items():
            if not isinstance(val, (float, int)) or "success_rate" not in key:
                continue

            k = parse_k(key)
            if k is None:
                continue

            failure[N][k] = 1.0 - float(val)

    for N in sorted(failure, key=lambda x: (isinstance(x, str), x)):
        ks_found = sorted(failure[N].keys())
        print(f"  [FixedN] N={N}: failure rates for k={ks_found}")
    return failure

def which_algo(name_lower):
    """Map run name to canonical algorithm name (case-insensitive)."""
    if any(k in name_lower for k in ["treconlm", "pretr", "finet"]): 
        return "TReconLM"
    if "robseqnet" in name_lower or "robseq" in name_lower:
        return "RobuSeqNet"
    if "trellisbma" in name_lower or "trellis" in name_lower:
        return "TrellisBMA"
    if "bmala" in name_lower:
        return "BMALA"
    if "muscle" in name_lower:
        return "MUSCLE"
    if "itr" in name_lower or "iterative" in name_lower:
        return "ITR"
    if "dnaformer" in name_lower:
        return "DNAformer"
    if "vs" in name_lower:
        return "VS"
    return None

def fetch_all_misclustering_data(entity, project):
    """Fetch misclustering data for all algorithms with explicit per-group rules:
    
      - TReconLM: misclustering_<cond>_<rate>_mean_levenshtein
      - DNAformer, RobuSeqNet: cont_<rate>_mean_levenshtein_all
      - All other algorithms: cont_<rate>_mean_levenshtein
      
    Accepts failed runs for TrellisBMA, finished runs for others.
    For algorithms in MERGE_ALGOS (e.g., TrellisBMA), data from multiple runs
    is merged to combine different contamination rates.
    """
    api = wandb.Api()
    
    runs_finished = api.runs(f"{entity}/{project}", filters={"state": "finished"})
    runs_failed = api.runs(f"{entity}/{project}", filters={"state": "failed"})
    
    # Regex patterns
    trecon_pattern = re.compile(r"^misclustering_[^_]+_([0-9]*\.?[0-9]+)_mean_levenshtein$")
    mean_pattern = re.compile(r"^cont_([0-9]*\.?[0-9]+)_mean_levenshtein$")
    mean_all_pattern = re.compile(r"^cont_([0-9]*\.?[0-9]+)_mean_levenshtein_all$")

    algo_data = {}

    def maybe_update(bucket, new_points, chosen_keys, run_name, algo):
        if not new_points:
            return
        
        should_merge = algo in MERGE_ALGOS
        
        if bucket in algo_data:
            if should_merge:
                existing_data = algo_data[bucket]['data']
                existing_keys = algo_data[bucket]['keys_used']
                source_runs = algo_data[bucket].get('source_runs', [algo_data[bucket]['source_run']])
                
                new_rates = set(new_points.keys()) - set(existing_data.keys())
                if new_rates:
                    for rate in new_rates:
                        existing_data[rate] = new_points[rate]
                        existing_keys[rate] = chosen_keys[rate]
                    source_runs.append(run_name)
                    algo_data[bucket]['source_runs'] = source_runs
                    algo_data[bucket]['source_run'] = ', '.join(source_runs)
                    print(f"  Merged {len(new_rates)} new rates into {bucket} from {run_name} (total: {len(existing_data)} points)")
            else:
                if len(new_points) > len(algo_data[bucket]['data']):
                    algo_data[bucket] = {
                        'data': new_points,
                        'source_run': run_name,
                        'keys_used': chosen_keys
                    }
                    print(f"  Updated {bucket} with more complete data from {run_name} ({len(new_points)} points)")
        else:
            algo_data[bucket] = {
                'data': new_points,
                'source_run': run_name,
                'keys_used': chosen_keys
            }
            print(f"  Found {len(new_points)} data points for {bucket} in run {run_name}")

    # Process all runs, combining finished and failed for TrellisBMA
    all_runs = list(runs_finished)
    for run in runs_failed:
        algo = which_algo(run.name.lower())
        if algo == "TrellisBMA":
            all_runs.append(run)
            print(f"\nIncluding failed TrellisBMA run: {run.name}")

    for run in all_runs:
        algo = which_algo(run.name.lower())
        if not algo or algo not in misclustering_algos:
            continue

        summary = dict(run.summary)
        data_points = {}
        chosen_keys = {}

        print(f"\nProcessing run: {run.name}  (algo={algo}, state={run.state})")

        for key, value in summary.items():
            if not isinstance(value, (int, float)):
                continue

            if algo == "TReconLM":
                m = trecon_pattern.match(key)
                if m:
                    rate = float(m.group(1))
                    data_points[rate] = float(value)
                    chosen_keys[rate] = key

            elif algo in ["DNAformer", "RobuSeqNet"]:
                m = mean_all_pattern.match(key)
                if m:
                    rate = float(m.group(1))
                    data_points[rate] = float(value)
                    chosen_keys[rate] = key

            else:
                m = mean_pattern.match(key)
                if m:
                    rate = float(m.group(1))
                    data_points[rate] = float(value)
                    chosen_keys[rate] = key

        if not data_points:
            print("  No valid metrics found for this run.")
            continue

        for r, keyname in sorted(chosen_keys.items()):
            print(f"  - {keyname}: {data_points[r]:.6f}")

        if algo == "TReconLM":
            tags_lower = {t.lower() for t in (run.tags or [])}
            has_trained = "trained" in tags_lower
            has_untrained = "untrained" in tags_lower

            if has_trained and has_untrained:
                print(f"  Skipping TReconLM run with both tags (ambiguous): {run.name}")
                continue
            elif has_trained:
                bucket = "TReconLM (trained)"
            elif has_untrained:
                bucket = "TReconLM (untrained)"
            else:
                print(f"  Skipping untagged TReconLM run: {run.name}")
                continue
        else:
            bucket = algo

        maybe_update(bucket, data_points, chosen_keys, run.name, algo)

    return algo_data

def is_trained_bucket(label: str) -> bool:
    return label.strip().lower().endswith("(trained)")

def is_untrained_bucket(label: str) -> bool:
    return label.strip().lower().endswith("(untrained)")

def create_combined_figure(show_heatmap_numbers=False, plot_style="line"):
    """Create the combined 1x3 figure with consistent styling"""
    print("Fetching sweep metrics...")
    metrics, _ = fetch_sweep_metrics()
    
    print("Computing noisy baselines...")
    noisy_baselines = compute_noisy_baselines()
    metrics.update(noisy_baselines)
    
    print("Fetching fixed-N failure data...")
    fixedN_data = fetch_fixedN_failure()
    
    print("Fetching misclustering data...")
    misc_algo_data = fetch_all_misclustering_data(ENTITY, misclustering_project)
    
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.3), dpi=300, gridspec_kw={'wspace': 0.5})
    ax1, ax2, ax3 = axes
    
    ks = list(range(11))
    
    # subplot 1: Average Levenshtein
    for algo, vals in metrics.items():
        is_noisy = algo in ["Avg. noisy read", "Best noisy read"]
        if is_noisy:
            continue
        ys = [vals["avg_levenshtein"].get(k, np.nan) for k in ks]
        
        ax1.plot(ks, ys,
                 color=color_dict.get(algo, "k"),
                 marker=marker_dict.get(algo, None),
                 linestyle='-',
                 linewidth=0.3,
                 markersize=1.5)
    
    ax1.set_xlabel(r"$k$")
    ax1.set_ylabel(r"$d_L$")
    ax1.set_xticks([0, 2, 4, 6, 8, 10])
    ax1.set_xticklabels(['0', '2', '4', '6', '8', '10'])
    ax1.set_yticks(np.linspace(0.1, 0.5, 3))
    ax1.set_ylim((0, 0.53))
    ax1.grid(False)
    
    # subplot 2: Misclustering Line Plot
    if misc_algo_data:
        to_plot = misclustering_algos + ["TReconLM (untrained)", "TReconLM (trained)"]
        for algo in to_plot:
            if algo not in misc_algo_data:
                continue

            entry = misc_algo_data[algo]
            data = entry['data']
            source_run = entry['source_run']
            
            sorted_rates = sorted(data.keys())
            x_vals = sorted_rates
            y_vals = [data[rate] for rate in sorted_rates]
            
            base_key = "TReconLM" if algo.startswith("TReconLM") else algo
            baseline = misclustering_baselines.get(base_key, 0.0)
            y_vals_relative = [val - baseline for val in y_vals]
            
            linestyle = '--' if is_trained_bucket(algo) else '-'
            
            print(f"[Misclustering] Plotting '{algo}' from run '{source_run}'  (linestyle: {linestyle})")
            for rate, abs_v, rel_v in zip(x_vals, y_vals, y_vals_relative):
                print(f"  cont={rate:.3f}  dL={abs_v:.6f}  dL_increase={rel_v:.6f}")

            this_label = None if is_trained_bucket(algo) else "TReconLM"
            ax2.plot(x_vals, y_vals_relative,
                     color=color_dict.get(algo, color_dict.get(base_key, "k")),
                     marker=marker_dict.get(algo, marker_dict.get(base_key, 'o')),
                     linestyle=linestyle,
                     linewidth=0.3,
                     markersize=1.5,
                     label=this_label)
        
        ax2.set_xlabel(r'$p_m$')
        ax2.set_ylabel(r'$d_L$ increase')
        ax2.set_xlim(0.01, 0.21)
        ax2.set_xticks([0.05, 0.10, 0.15, 0.20])
        ax2.set_xticklabels(['0.05', '0.10', '0.15', '0.20'])
        ax2.grid(False)

    # subplot 3: Failure Rate as Bars or Lines
    Ns = sorted([n for n in fixedN_data if isinstance(n, int)])
    cmap = truncate_colormap(cm.PuBu, minval=0.3, maxval=0.9)
    
    if plot_style == "bar":
        bar_width = 0.13
        x = np.array(ks)
        
        for i, k in enumerate(ks):
            failures = []
            if "allN" in fixedN_data and k in fixedN_data["allN"]:
                failures.append(("TReconLM", fixedN_data["allN"][k], "TReconLM"))
            for N in Ns:
                if k in fixedN_data[N]:
                    failures.append((f"N={N}", fixedN_data[N][k], N))
            for j, (label, fail_rate, style) in enumerate(failures):
                offset = (j - (len(failures) - 1) / 2) * bar_width
                x_pos = x[i] + offset
                if style == "TReconLM":
                    ax3.bar(x_pos, fail_rate, width=bar_width,
                            color="#6699CC", edgecolor=None, linewidth=0, zorder=2)
                else:
                    N_val = style
                    norm = (N_val - min(Ns)) / (max(Ns) - min(Ns)) if max(Ns) != min(Ns) else 0
                    color = cmap(1 - norm)
                    ax3.bar(x_pos, fail_rate, width=bar_width,
                            color=color, edgecolor=None, linewidth=0, zorder=2)
    else:
        if "allN" in fixedN_data:
            y_vals = [fixedN_data["allN"].get(k, np.nan) for k in ks]
            ax3.plot(ks, y_vals, 
                     color="#6699CC", 
                     linestyle='-',
                     linewidth=0.3,
                     marker='o',
                     markersize=1.5,
                     label="TReconLM",
                     zorder=3)
        for N in Ns:
            if N in fixedN_data:
                y_vals = [fixedN_data[N].get(k, np.nan) for k in ks]
                norm = (N - min(Ns)) / (max(Ns) - min(Ns)) if max(Ns) != min(Ns) else 0
                color = cmap(1 - norm)
                ax3.plot(ks, y_vals,
                         color=color,
                         linestyle='-',
                         linewidth=0.3,
                         marker='s',
                         markersize=1.2,
                         label=f"N={N}",
                         zorder=2)
    
    if Ns:
        norm = plt.Normalize(vmin=max(Ns), vmax=min(Ns))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        pos = ax3.get_position()
        cbar_left = pos.x0 + pos.width * 0.93
        cbar_bottom = pos.y0 + pos.height * 0.03
        cbar_width = 0.008
        cbar_height = pos.height * 0.25
        cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_ticks([10, 50])
        cbar.set_ticklabels(['N=50', 'N=10'])
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        if len(Ns) > 2:
            minor_ticks = [N for N in Ns if N not in [10, 50]]
            cbar.ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            cbar.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        cbar.ax.tick_params(axis='y', which='both', color='lightgray', labelcolor='black', 
                            size=1.5, width=0.4, pad=1, direction='out', labelsize=fontsize-2)
        cbar.outline.set_visible(False)
        yticklabels = cbar.ax.get_yticklabels()
        if len(yticklabels) >= 2:
            yticklabels[0].set_verticalalignment('bottom')
            yticklabels[-1].set_verticalalignment('top')
    
    ax3.set_xlabel(r"$k$")
    ax3.set_ylabel("Failure rate")
    ax3.set_xticks([0, 2, 4, 6, 8, 10])
    ax3.set_xticklabels(['0', '2', '4', '6', '8', '10'])
    ax3.set_yticks([0.20, 0.60, 1.00])
    ax3.set_ylim((0, 1.05))
    ax3.set_xlim((-0.5, 10.5))
    ax3.grid(False)
    
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color('lightgray')
        ax.tick_params(axis='both', which='both', color='lightgray', labelcolor='black')
    
    # Global legend (consistent with Cell 1)
    preferred = ["RobuSeqNet", "VS", "MUSCLE", "BMALA", "TrellisBMA", "ITR", "DNAformer", "TReconLM"]
    handles, labels = [], []
    for algo in preferred:
        h = Line2D([0], [0],
                   color=color_dict[algo],
                   marker=marker_dict[algo],
                   linestyle='-',
                   linewidth=0.5,
                   markersize=2)
        handles.append(h)
        labels.append(algo)
    
    fig.legend(handles, labels,
               loc="upper center",
               ncol=len(handles),
               frameon=False,
               bbox_to_anchor=(0.5, 1.1),
               fontsize=fontsize-0.8,
               handletextpad=0.3,
               columnspacing=0.6,
               handlelength=1.5)
    
    plt.tight_layout(w_pad=0.0)
    output_path = os.path.join(save_dir, f"combined_figure_{plot_style}.pdf")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    # Per-dataset plots (L=60, L=110, L=180, Microsoft, NoisyDNA, Chandak)
    main()
    # Combined scaling + misclustering figure
    create_combined_figure(show_heatmap_numbers=False, plot_style="line")