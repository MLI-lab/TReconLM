"""
Shared configuration and utilities for all plotting scripts.

Contains:
- W&B configuration (entity, projects)
- Algorithm colors and markers
- LaTeX/matplotlib settings
- Metric extraction functions for different logging formats
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# 
# W&B Configuration
# 

WANDB_ENTITY = "<your.wandb.entity>"

# Project names
WANDB_PROJECTS = {
    "artifact": "TRACE_RECONSTRUCTION",
    "baselines": "Baselines",
    "reproduce": "Reproduce",
    "finetune_microsoft": "FinetuneMicrosoft",
    "finetune_noisy": "FinetuneNoisyDNA",
    "chandak": "chandak",
    "beam_search": "BeamSearch",
    "majority_voting": "MajorityVoting",
    "timing": "Timing",
    "inference": "Inference",
    "misclustering": "Misclustering",
    "robuseqnet": "RobuSeqNet",
}

# Default directories
SAVE_DIR = "./plots"
DOWNLOAD_DIR = "./downloaded_artifact"


# 
# Algorithm Visual Config
# 

# Unified color scheme for all algorithms
ALGO_COLORS = {
    # Main methods
    "TReconLM": "#6699CC",
    "TReconLM (p.)": "#6699CC",      # pretrained variant
    "TReconLM (f.)": "#6699CC",      # finetuned variant
    "TReconLM (untrained)": "#6699CC",
    "TReconLM (trained)": "#6699CC",
    "CPRED": "#6699CC",              # same as TReconLM

    # Baselines
    "ITR": "#66CCEE",
    "itr": "#66CCEE",
    "DNAformer": "#9CAF88",
    "TrellisBMA": "#CCBB44",
    "BMALA": "#EE6677",
    "bmala": "#EE6677",
    "RobuSeqNet": "#F8DB57",
    "MUSCLE": "#F287BD",
    "muscle": "#F287BD",
    "VS": "#FFAA00",

    # Ablations
    "NESTED": "#5ab4ac",
    "MSA": "#9D9AC5",
    "MAJORITY": "#EFC8D5",
    "CRED (beam=2)": "#9BB8E8",
    "CRED (beam=4)": "#5C85C5",
    "CRED (beam=6)": "#9C8380",
    "CRED (beam=8)": "#9C8380",

    # GPT baselines
    "gpt-4o mini 0-shot": "#CC76D2",
    "gpt-4o mini 3-shot": "#DA493E",
    "gpt-4o mini 5-shot": "#F09236",

    # Noisy references
    "Avg. noisy read": "#AAAAAA",
    "Best noisy read": "#AAAAAA",
}

# Unified marker scheme
ALGO_MARKERS = {
    "TReconLM": "o",
    "TReconLM (p.)": "o",
    "TReconLM (f.)": "o",
    "TReconLM (untrained)": "o",
    "TReconLM (trained)": "o",
    "CPRED": "o",

    "ITR": "s",
    "itr": "s",
    "DNAformer": "^",
    "TrellisBMA": "v",
    "BMALA": "D",
    "bmala": "D",
    "RobuSeqNet": "P",
    "MUSCLE": "X",
    "muscle": "X",
    "VS": "*",

    "NESTED": "^",
    "MSA": "v",
    "MAJORITY": "4",
    "CRED (beam=2)": "1",
    "CRED (beam=4)": "3",
    "CRED (beam=6)": "4",
    "CRED (beam=8)": "2",

    "gpt-4o mini 0-shot": "^",
    "gpt-4o mini 3-shot": "s",
    "gpt-4o mini 5-shot": "D",
}

# LaTeX labels for algorithms
ALGO_LABELS = {
    "TReconLM": r"\textsc{TReconLM}",
    "TReconLM (p.)": r"\textsc{TReconLM} (p.)",
    "TReconLM (f.)": r"\textsc{TReconLM} (f.)",
    "CPRED": r"\textsc{TReconLM}",
    "ITR": r"\textsc{ITR}",
    "itr": r"\textsc{ITR}",
    "DNAformer": r"\textsc{DNAformer}",
    "TrellisBMA": r"\textsc{TrellisBMA}",
    "BMALA": r"\textsc{BMALA}",
    "bmala": r"\textsc{BMALA}",
    "RobuSeqNet": r"\textsc{RobuSeqNet}",
    "MUSCLE": r"\textsc{MUSCLE}",
    "muscle": r"\textsc{MUSCLE}",
    "VS": r"\textsc{VS}",
    "gpt-4o mini 0-shot": r"\textsc{GPT-4o Mini} 0-shot",
    "gpt-4o mini 3-shot": r"\textsc{GPT-4o Mini} 3-shot",
    "gpt-4o mini 5-shot": r"\textsc{GPT-4o Mini} 5-shot",
}


def get_color(algo):
    """Get color for algorithm, with fallback."""
    return ALGO_COLORS.get(algo, "#888888")


def get_marker(algo):
    """Get marker for algorithm, with fallback."""
    return ALGO_MARKERS.get(algo, "o")


def get_label(algo):
    """Get LaTeX label for algorithm, with fallback to raw name."""
    return ALGO_LABELS.get(algo, algo)


# 
# Matplotlib/LaTeX Setup
# 

def setup_latex_plots(fontsize=7.7, font_family="serif", serif_font="Computer Modern Roman"):
    """
    Configure matplotlib for publication-quality LaTeX plots.

    Args:
        fontsize: Base font size
        font_family: Font family ('serif' or 'sans-serif')
        serif_font: Serif font to use ('Computer Modern Roman' or 'Times')
    """
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': font_family,
        'font.serif': [serif_font],
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{type1cm}',
        'font.size': fontsize,
    })


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# 
# Metric Extraction Functions
# 

def parse_metrics_N_format(run_summary, metrics=None):
    """
    Extract metrics from run summary using `metric_N=<cluster_size>` format.

    This format is used by: baselines, finetune runs, chandak.
    Example keys: avg_levenshtein_N=5, success_rate_cropped_N=5

    Args:
        run_summary: W&B run.summary dict
        metrics: List of metric names to extract, or None for all

    Returns:
        dict: {metric_name: {cluster_size: value}}
    """
    if metrics is None:
        metrics = ["avg_levenshtein", "success_rate"]

    out = defaultdict(dict)
    for k, v in run_summary.items():
        if not isinstance(v, (int, float)):
            continue

        # Match: metric_N=5 or metric_cropped_N=5
        m = re.match(r"^(\w+?)(?:_cropped)?_N=(\d+)$", k)
        if m:
            metric_name, n = m.group(1), int(m.group(2))
            if metric_name in metrics:
                out[metric_name][n] = float(v)

    return dict(out)


def parse_metrics_k_format(run_summary, run_name=None):
    """
    Extract metrics from run summary using `_k<cluster_size>` format.

    This format is used by: sweep/timing runs.
    The cluster size k can be in the metric key (_k5) or the run name (sweep_5).

    Args:
        run_summary: W&B run.summary dict
        run_name: Run name to extract fallback k from

    Returns:
        dict: {metric_name: {cluster_size: value}}
    """
    out = defaultdict(dict)

    # Get fallback k from run name
    k_from_name = None
    if run_name:
        m = re.search(r"sweep[_=]?(\d+)", run_name, flags=re.IGNORECASE)
        if m:
            k_from_name = int(m.group(1))

    for key, val in run_summary.items():
        if not isinstance(val, (int, float)):
            continue

        # Try to get k from the key itself
        m = re.search(r"_k=?(\d+)\b", key, flags=re.IGNORECASE)
        k = int(m.group(1)) if m else k_from_name

        if k is None:
            continue

        # Extract metric name
        if "avg_levenshtein" in key:
            # Prefer "all" variant if present
            if "all" in key:
                out["avg_levenshtein"][k] = float(val)
            else:
                out["avg_levenshtein"].setdefault(k, float(val))
        elif "success_rate" in key:
            out["success_rate"][k] = float(val)
        elif "failure_rate" in key:
            out["failure_rate"][k] = float(val)
        elif "inference_time" in key:
            out["inference_time"][k] = float(val)

    return dict(out)


def parse_metrics_split_format(run_summary):
    """
    Extract metrics by splitting on `_N=`.

    Simpler alternative to regex - just splits the key.
    Example: "avg_levenshtein_N=5" -> metric="avg_levenshtein", n=5

    Args:
        run_summary: W&B run.summary dict

    Returns:
        dict: {metric_name: {cluster_size: value}}
    """
    out = defaultdict(dict)
    for k, v in run_summary.items():
        if not isinstance(v, (int, float)):
            continue
        if "_N=" not in k:
            continue

        try:
            metric, n_str = k.rsplit("_N=", 1)
            n = int(n_str)
            # Remove _cropped suffix if present
            metric = metric.replace("_cropped", "")
            out[metric][n] = float(v)
        except (ValueError, IndexError):
            continue

    return dict(out)


# 
# Algorithm Name Parsing
# 

# Mapping from run name patterns to canonical algorithm names
ALGO_NAME_MAPPING = {
    "VS_algorithm": "VS",
    "muscle": "muscle",
    "bmala": "bmala",
    "trellisbma": "TrellisBMA",
    "Iterative_algorithm": "itr",
    "ITR": "itr",
    "Robseqnet": "RobuSeqNet",
    "DNAformer": "DNAformer",
    "pretr": "TReconLM",
    "finet": "TReconLM",
    "CPRED": "TReconLM",
}


def get_algo_from_run_name(run_name):
    """
    Extract canonical algorithm name from W&B run name.

    Args:
        run_name: W&B run name string

    Returns:
        str: Canonical algorithm name, or None if not recognized
    """
    name_lower = run_name.lower()

    # Check each pattern
    for pattern, algo in ALGO_NAME_MAPPING.items():
        if pattern.lower() in name_lower:
            return algo

    # Fallback: check if any known algo name is in the run name parts
    for part in re.split(r"[_\-]", run_name):
        if part in ALGO_NAME_MAPPING.values():
            return part

    return None


def parse_cluster_size_from_name(name):
    """
    Extract cluster size (k or N) from run/metric name.

    Handles formats:
    - sweep5, sweep_5, sweep=5
    - _k5, _k=5
    - _N=5

    Args:
        name: String to parse

    Returns:
        int: Cluster size, or None if not found
    """
    # Try sweep format
    m = re.search(r"sweep[_=]?(\d+)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Try _k format
    m = re.search(r"_k=?(\d+)\b", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Try _N= format
    m = re.search(r"_N=(\d+)", name)
    if m:
        return int(m.group(1))

    return None


# 
# Utility Functions
# 

def safe_mean(x):
    """Compute mean, returning nan for empty sequences."""
    if x is None or len(x) == 0:
        return np.nan
    return np.mean(x)


def safe_std(x):
    """Compute std, returning nan for empty sequences."""
    if x is None or len(x) == 0:
        return np.nan
    return np.std(x)
