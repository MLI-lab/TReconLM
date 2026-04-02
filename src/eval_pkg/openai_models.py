# export OPENAI_API_KEY=key

from openai import OpenAI
import re
import time
import Levenshtein
import sys
import random
import textwrap
import os
import json
import numpy as np
from collections import defaultdict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available for W&B tables. Install with: pip install pandas")

# Hardcoded few-shot examples for DNA sequence reconstruction
HARDCODED_EXAMPLES = [
    'TGTTCGGGATGGGAGACACCAGAAACCTCGGAAGTAATTCCGCGCATCTGGCCTCCGGG|TGTTTGGGATGGGAGACACCAGGACAACCGCCGGAATAATTTCCGCGCATCTGGCTCCGGG|TGTTTTGGATGGCACACCACCAAGACAACCGCGCGGAGTAATTCCGCGCATCTGGCTGCGGA|TGTTGTGGGATGGGAGACACCTAGACAACCGCGCGGAGTAATCTCCGCGCATCTGGCTCCGGG|TGTTTGGGATTGGAGACCCCGACACCGCGCGGAGTAATTCCGCGCATCTGGCTCCGGG|TGTTTGGGATGGGAGACACCAGACAACCGCGGGGAGTAATTCGCGCAGCTGGCTCCGGG|TGATGTGGGATTGGAGTCACCAGACAACCGGCGGAGTAGGTCCGCCGTGTGGCTCCTG|TGTTTGGGATGGGAGACAACAACAACCGCGCGGAGTAACTCCGCGCATCTGGCCCCGGG|TGTTTGGGCTAGAGACACCAGACAACCGCGCGGAGTAATTCCGGGCATCTGGCTCCGGG|GGTTTGGGATGGGAGACACCAGACAACCGTGCGGAGAAATCCGCGCATCTAGCTCGGG:TGTTTGGGATGGGAGACACCAGACAACCGCGCGGAGTAATTCCGCGCATCTGGCTCCGGG',
    'GGTGATGCCCGCTCGCTTGGAATGCTATACGTACTCCGTGTGGTCGTGACGGGAAGGGATG|GGTGAACCCCGCTCGCTCGTAATGGCTAGACGCAATCCGTAGGTCGTGGCGAGAAGGGATG|GGTGAGCCAGCTGCGCTCCGGAATGGCTATCGCAATCCGTAGGTCGTGACGGGAGAGCGATG|GGTGAGCCCGCTCGGCTTGGAAGTGGCATATACGCATCCGTAGGTCGTGAGCGGGTAAGGGATG|GGTGAGCTCGCTCGAGCTTTTATTCGTTATACGCTATTCGTATGTCGTGTCGGGAAGGGATG|GGCTGAGCCCGCTCGCTTGGAATCGCTCTATACACATTCCGTAGGTCGTGAGCGGGAAGGGAG|GTGTTGCTCGCTCGCTTGGAATGGCTATACGCAATCCGTAGGTCGTGACGGCAAGGGATC|GGTGAGCCTGCTCGCTTGGGATGGCTATACGCAATCAGTATGTCGTTGACGGAATGGGATG|GGTGATCTCGCTCCACTGGAATGGCTATGCGCAAACCCGTAGCTCGTGATGGGAAGGGATG|GGTGAGCTCCGCTTCGTTTGCGAATCCGGCTATAGGCAATCCGTAGCTCGTTGCCGGGACAGGGCTC:GGTGAGCCCGCTCGCTTGGAATGGCTATACGCAATCCGTAGGTCGTGACGGGAAGGGATG',
    'CCACGACATGGTACAAGAAAGGGCCATGAAACCCTCACTCCTTCGGGCTTGCAAATTGC|CCCAGGAAAATGGCAAGAAGGCCATGAAACCCTCACTCTTCCGGGCGTCCGGCAATTGC|CTCCACGGAACATGGTGCAAGAAAGTGCCATGAAACCCTCACTCCTCCGGGCGTGGCAGAATTGC|CACCAGAACATGGGTCAAGAAAGGCCATGCAGAACCTCACTCCTCCGGCGTGGCATACATTGC|CCCACGAACATGGTCCAAAGCAGGCGATGAAACCACTCACTCCTCCGGGCGTGGCAAATTGC|CCACGAACATGGTCAAGAAGGCCATAAACCTCAACTCCTCCGGGCGGTGGCATAATCTGC|CCACGGACATGGTCGAGAAAGCCATGAAACCCTCACTCCTCCGGGCGTGGAAATGC|CCCCAGAACATGGGTCAAGAAAGGCCATGAACCCCACTCCTCCGGGCGTGTCAAATGC|CCCACGAACATGGTCAGAACAGGCCATGAAACCCTCGACTCTCCGGGCGTGGCAAATTC|GCCACATACATGCGTCAATGAAAGGCCTGAAACCTCACTCCTCCGGGCGTGGGCAAATTGA:CCCACGAACATGGTCAAGAAAGGCCATGAAACCCTCACTCCTCCGGGCGTGGCAAATTGC',
    'CGCGCTTGAGTCGAGCGCCTCAACTCCAAGTCTGCTCGATTGCTCGAATTACACTGCCTC|CGGCGTATGAGTCGATGGCCTCCACTCACAAGTCTCTCATATGCTGCCATTACACTGCCTC|TCGTTAGGTCGATGGCGCTCACTCACAAGTCTGGCTCATATGCTGAATGTAACACCCCTC|CGCGTTGAGTCGAGGCCTAATACAGTCTGCTCATTTTGCTGTAATGTAACACTGCCTC|CGCGTTGAGTTCGATGGCCCACTCACAAGTCTGCTCATTAAGCTGAATAGTAACACTGCCTC|CGCGTTGAGTCGATGGCTCCACTCACAAGTCTGCTCATATGCTGAATGTAACACTGCCTC|CGAGGTGAGTCGATGGCCTCAGCTCAAAATCGCTAACATATGCTGAATGAAAACACTGCCATAC|CGCGTTAGTCTGTATGGCCTCACTCACTAAGATCTGTCATATGCTGAACGTAACAGCTGCCTC|ACGCGTTGAGTCGGTAGGCCTCACTCACAAGATTGCTCATATGCTGGATGTAACACTGCCTC|CGCGTTGAGTCGATGGCCTACTCACAAGTCTGCTCATATGCTTAATGTAACCATGCCCT:CGCGTTGAGTCGATGGCCTCACTCACAAGTCTGCTCATATGCTGAATGTAACACTGCCTC',
    'AACTCGAAGGAGAGCGCCATCCATATGGCGGAGCGTTACGTGTCACTCAAGGTTCTT|AAATCGAACGCCGAGAGCTCCATAATAGGGCAGTAGCGGAAGTATCTGTCATCTTAAGGTGCTT|GACGTTGACCATGGATGAGCGCATATCCATCGGGTTTGACGGCGTATCTCCCATTTTAAGTCTT|ACCAGTCGAACTGACGAGCGCGAATCCATAGGGCCTGACGACGTAGTTGTCTATCTTAAGGTTCTT|AAGCCGAACCGGAGAGAGCCATCCCAGGGCCTGGCGCCGATCTGCTCATCGGGTTAAGGTGTCCTT|AGGTCGAACAGAGAGCTCCATCCTTGGGGCCTGACGTGTTTCTCGGTTCATCGTCTTAGGTCATT|ATAGTCGAAGGAGCGCGCCATCCACCAGGGCGTGCGCGTATCCTGTCATCTATAAGGTTCTT|AAGTCCGAAGGGCAGTGCCACCATAAGGGCGTGACGTGTATGTGTCATCTTAAGGTGGTT|ATGTTCGAACGGAGAGCGCATCCGATAGGGCCGTGACGCGTATCTGTCATCTTAAGCGTTATT|AAGATCGGACCGGAGAGCGCCGTTCCCCAGGGCTGTGACCGTAGGATTCTGTCGTGCTTAAGGTTCAT:AAGTCGAACGGAGAGCGCCATCCATAGGGCGTGACGCGTATCTGTCATCTTAAGGTTCTT',
    'AAGCTTGAGTGATGGCCAACGGGTTAAGGCCTATATCAAAACTAGAGGCTAATAAGTC|ACTAGCTGCGGTGATGGCCAACGGCTTAAGGCCTATTCAAACCCTAGGAGCTCATAACTC|AAAGCTGCAGTGATGGCCAACGGCTTAAGGCCTATATCCAACTAGGAGCTCAATAACTC|AAAGGCGCGAGGATGGCCAACGGCGTAAGGCCTATATGAAACTAGGGCTCAATATCTC|AAAGCTTGGGGGGCCAAGGCTTAAGGCCCATATCAAACGTAGGAGCTCAATAACGC|AAAGTTGAGTGATGGCTAACGGCTTATGGCTATATCAAACCAGAGCTCAATAACTA|AAAGCCTGAGTCAGGCCAACGGCTTAAGGCCTAGATCAAACCCTAGGAGCTCAATTAATC|AAAGCTGAGTGGGCCAACGGCTTAAGGCCTATATCAAACCTATGGAGCTCATAACCC|AAAGCGTGAGTGGTCGCCAACGGCTTAAGGCATTCAAACCTAGGAGCTCAATAACTC|AAAGCTTGAGGTGGCCAACGGCTTAAGGCCTTTATCAAACCTAAGGAGCTCAATAACTC:AAAGCTTGAGTGATGGCCAACGGCTTAAGGCCTATATCAAACCTAGGAGCTCAATAACTC',
    'ACAATCGTGGGCATGAGTCCTCAGTATCCTCAGGAATAGGACAAGTTCCACGTGAACTATGGCG|ACATCGGGCAGCGCTCTTTCATACCGTAGAATATTGGCAAGTTTGCCTGTGAACTATGCG|ACATCTGTGGGCAGAGGTCCTCAGACCCTCAGGAATTGGACAAGTTTCCACGTGAAACTAGGCG|CATCGTGGCAGAGTCCTCAGCACCCTCAGGAATTGGACAAGTTTCCAGTGATTCTATGCG|ACTCAGTGGCAGGGTCCTAGACCTCAGGAGATTGGCACAACGATTCCACGTGAACTATGC|ACATCGTGGCAGAATCCTCAGACCCTCAGGAAAGTTGGACAAGTTTCCAGGTGAATCTATGCG|ACATGTGAGCAGAGTCCTCGAACCCTCGGGCCGATTGGAATAAGTTTCCAAAGTGACTATGGG|ACATCGTGGCCTAGGAGTACCTCATGACCCTCATGATTGGACAAGATTCCACGTGAATGTAGG|ACATACCGTGGCAAGTCGTCTTTCATACCCTCAGGGATTGGACATGTGTCACACCGTGATACTGTGCG|ACAGCGTGGCAGAGTCCTCAGCCCCTCAGGAATTGGACAGGTTCCACGGCCAACTATGCG:ACATCGTGGCAGAGTCCTCAGACCCTCAGGAATTGGACAAGTTTCCACGTGAACTATGCG',
    'ATGAACTACCCTCAATCATCTCAACTGCGGAAGTGGAGCCCAGCCCGACACGGGAAAG|ATGTAACATTACCCAATCTCTAGTCACTGCGGAAGTGGAGCCCAGCCCTGATACGGGAACG|ACGTAACTAACCCCAATCGTCGTCAACTGCGGAGTGAGCGCCAGCCACGTACACGCGGAACG|ATGTAACTAACCCAATCATCTTCAACTCGGAAGTGGAGCCCACCCGATGCGGGAAACG|TTGTAACTAACCACAATCAGTCTTCACCTGCGGAGCAGGGAGCCCAGCCCGACACGGGAAACG|ATGTAACTGAAGCCAATCATCTCAACTGCGGAAGTGGAGGCCCACCGACACGGGAAACAG|ATGTAACTAACACAACCATCTCAACTGGGAAGTGGAGCCCAGCCCGACACGGCACAG|ATGTTAACAAACCCAATCATCTTCAACTGGCGGAAGTTGGAGCCCAGTCCAGACACGGGAAACG|ATGTAACCTCCCCAATCGAACTTCAACTGCGGAAGTGGAGCCCAGGCCCCGAACGGGCGAAACG|ATGTACACTACACCCAATCATCTCAACATGCTGAAGTGGATGCCCAGTCCCGAGACGGGAAACG:ATGTAACTAACCCAATCATCTTCAACTGCGGAAGTGGAGCCCAGCCCGACACGGGAAACG',
    'ATGCAACGAATGCTGGCCGGATACATCAAACGATTTCAAGTTATATCCCGTTT|GCCCGACGAATGATGTCCGGTCAGCTACACGTCGTCAAGTATACCGTTAT|ATGCCCGATAATATATGGCGGACTCCACTCTACACGTCGTCAAGTTATATCCCGTTAG|TGCCCCGACGATATGCCGGCGGATACACTCTCACGATCGTCAAGTATATCCGTTAA|ATGCCCGACGCTTCTGGCCGGATACACTCAACAATCGTCACCGTTTATCCGATAA|ATGCCCGACGAATGCTGGCCGGATACACTTACACGATGTCAATGATATCCGAGTG|CGCCCACATATTTGCCGGATACACTTAACATAGTCAAGTAATCGCGTTAT|ATGCCCGAGATATGTGGCCGGCTAGACTTACACGATCTCAGTTAATCCCGTTAT|ATGCCCACGAGTATGCTGCCGGATCCTCACAAATCGTCAAGTTATATCCCGATAT|GCCAGACATAAGCTGGCCGATAAACTGTCACAAACGTCAGTTATCCCGGT:ATGCCCGACGATATGCTGGCCGGATACACTCTACACGATCGTCAAGTTATATCCCGTTAT',
    'CTACCAGGTCGAGGTAATGTGTTCGCATATCCTGACACAGGGCTGTCATGGTTGAACAA|CTACTAGGTCGAGGGAATCAGTTCGCCTGATCGTAACAGATGGGCCGTCATGGTTGAACAT|CTACTAGCCGAGGTAATATGTTCGCTTGATGCCTAACACAGAGCCGTATGGTCGAAAA|CTACTGGGTCGAGGCACTAAGTTCGCTTGAACCTAAGCACAGGGCCGTCAAGGTTGCACAA|CTACTAGGTCGAGGTAATAAGTTCGCTTGATCCTAACACAGGGCCGGCATGGTAGAACAA|CAACTAGGTCGGGACATAATTTCGCTTGATCCTAACACAGGGCCGTCATGGTTGACA|CAACTAGGTCGGTAATAAGTTCGCTGGATCCTAACACAGGGCCGTCATGACTGAACAA|ATACTAGTCGAGGAATAAGATTGTTGATCCCAACACAGGGCTCCGTGATGGTTAAACTTCA|CTACTAGGTCGAGGTAATAAGTTCGCTTGATCCTAACCCAGGGCCGACATGCGTTGAACAA|CTACTACAGGGGGGAAATAAGTTCGCTTGATCCTACAAAGGGGCGTCTGGTTGGACAT:CTACTAGGTCGAGGTAATAAGTTCGCTTGATCCTAACACAGGGCCGTCATGGTTGAACAA',
    'CACTTCGACCGTGTTACGCCGAGGGTTCGTTTCATAAAAGTAGCAAGCGTGATTATCTCATAGC|CACATATCGAACCGGTTACGCCGAGGGTTCGTGAAATAATAGTAGGAGCGTGTTAATTCAAACTGC|CACTTTCGACCGAGTTACGCCGAGGGTTCATTCAATAATTGTAGGAGCGTGTTAATTCACCTGC|CACCTTCGACCGGTTCGTCCGAGGGTATGGATCATTAATAGTAGGAGCGTGTTGATCCACTGC|CGACTTCGACCGGTTACGCCTGCAGCGTTCGTACCTAGTAGTAGGAAGCGTCTTAATTCACTGC|CAAATTCGATCGGTTACGCCGAGGGTCCGTTCTTAATAGTTAGGAAGCGTGTTGATTCACTGC|CACTTCGACCGGTTACTCACGAGGGTTGGTTCTTAATAAGAGGAGCGTGTTACATTCACTGCC|CGACCTTCGACTGGTTACGACCTGAGGGTTCGTTCACATTTAGTAGGAGCGGGTTAATTCACTGC|CACTTCGACCGGTTACGCTCAGAGGGTTCGTTCATAATCCGGTAGGAGCTTCGTTTAAGATCTCATGC|CACTCCGACCTGGTTACGCGAGGGGTCAGTTCATACATAGTAGGAGCCTGTTAATTCACTGGC:CACTTCGACCGGTTACGCCGAGGGTTCGTTCATAATAGTAGGAGCGTGTTAATTCACTGC',
    'AAACCCTTACGGGTCGAATACATCTTATCCGAGCGCCTCAAGGAGTAGCGATTCCTAC|AAACCCATAGGGTCCAAAAATATTTACCGTGCACTCCGAAAGGGAGTATCGTTGATA|AAACACTTGGGGTCGAAAAAATACTATCCGTGTACCCCAGAGGTGTAGTGTCTCATAC|AACCCTGAGGGTCGAATACTGTTTGATCCGTGCACCTCCATGAGGGTGTCGCGGTTCATGC|AAACCTTAGGGCTCGAATACATATTTACCGTGCACCTCCAGAGGAGTAGCGTTTCAA|GAATCCTTAGGGTCGACCACATATTATCCGAGCACCTCCAGAGGAGTAGGTTTCATGAC|TCACCCTTGGCGGTCGAAGCCAATTTATACGTGCAGCTGCAGAGGTCACCGTTTCATAC|ACACCCTTAGGGTCGAAAACATATTTACCGTGCACCTCCAGAGGTGTATCGTTTCATAC|AAACCCCTAGGGTCGAATACATATTTATCCGTGCACCTCCAGAGGAGTTCTTTTATAC|AAACCCGTCGGGTCGAATCCTATTTATCCGTGCACGACCAGAGGAGAATCGTTTCCGAC:AAACCCTTAGGGTCGAATACATATTTATCCGTGCACCTCCAGAGGAGTAGCGTTTCATAC',
    'GTACTTGGCGTGAATACTGCTACAGGGTCGCAGCCCCTGCTTCGTGCGCTCATTGCCATAGGAGCATATG|GTTACTGGCTGTCTTGTAAGGTACCGCAGACCCTGCTTTCGCCTTACACTATTAGGCACATG|GTAGCTTTTGCTGACTGCTAACTCGCAGCCCCGTTCTTGTCTCTCATACATAGGGAGAATG|CTATGGCTGACTGCTAAGGTCGCAGCCCTAAGCTTCCGGGCCTCATGAATCTGGTCTGAATG|TTAACTGGCTGACTAGGCTAAGGTCGCAGCCCCGCTTCGTCCTCTAACCATGGCAAGAAG|GGTACGGGCTGCACAGCTAAGGTGTCAGCCCACGTGCTTCGTGCCTAAACCTCTAGTCAGAAGTG|ATACTGGTGACTGCAAGGTCCAGGCAGCCCCCTGCTTCGTGCCTCATGATCCATAGGCATAATG|GACTGGCGGCTGCTAAGGTCGCAGGCCCCTGCTTCGTGTCCGCATACACATAGGCATGAAAGC|GTACGGCTCTGCCAGAGTCGCCAGCCCGCTGCTTCGTGCCTCATAGAATAGCAGAATG|GTAGGGTGCTGCCAAGGTCGCCGCCCCTGCTTCGTCCCCTAGTACCATAGGCAGAAT:GTACTGGCTGACTGCTAAGGTCGCAGCCCCTGCTTCGTGCCTCATACCATAGGCAGAATG',
    'GATATAGATGTTGCTCGAGAGAATACTGCACAAAGTGTACAGAAGAGATGCTGTAGGAG|GAGAAAGGATTGTACTGAGTGGTACTGTACAAGAGTAAGAAGAGATGCAAAGGTAG|ATAAAGGATCGTTAGCTCGAGTGGATACTGACAAAGAGTCAGAAGAGACGCTAATGGTAG|GAGTAAGATTGTGTGCTCGAGTGAATACCTGTACAAGAGCAAAGAGATGCTAACGGTAG|GTGAAAGGTTGTGCTCGAATTGGAATCATGTGACAAAAATTTCAGAAGAGATGGAAGGTACG|GATACGGATTGTGCTCGAGTGGATACTGGTATAGAGAAGAGAGTAATGCTAAGGTAG|ATATAGGACTGTTCCTCGAAGTGGATACTGTACAAAAATCAGAAGCGAGTAAGGTAG|GATCAGGATTGTACTCGAGTGCTACTGTACAAAGCGTCAGAGGTGCCATAGGTACG|GATAAAGGGACGTTGCCCGAGTGATACTGTCAAAGCGTAAAAGAGATGCTAGGTG|GGATCAAAGGATTGCTTGCTCGAGTGTGATACTGTACAATGATCAGAAGAGATCTAATAG:GATAAAGGATTGTTGCTCGAGTGGATACTGTACAAAGAGTCAGAAGAGATGCTAAGGTAG',
    'TGCTCGCCTCTTTGTTCCTCTCTGTGCAGCTCAACTTTTTAACAACGCTCTATAT|TGCCGCACTCTTGTTCGCTTTAGGGACGGCTCACCTTTTGGAACATAACGCGTCTAATAT|TGCTGCGCCTCTTGTTCTCTTTACGGACGTCTCAACTTTTGTAACATACGCGTGCATAT|TTGCTCGCCTCTTGTCCTTCTTTTAAGACGTCTCAACTCGTGAACATACGCGTGCTATAT|TCTCGCCTCTTGTCCTCTTTACGGCACTCAAACTTTTGGAACAACGCGTGTTATTT|GCCGCCTATGATCCTCTTGAACGACGTCACAATTTGGAAGCCATACGCGTGCTATAT|TCTTGGCACTCTGCTCTCTTTACGTGCCATTCACTTTTGGGTACATAGCCGTGCTATGA|TGCACGCCTCTTGTTCCTTTTTCGGACGGCGCAACTTTTGAAACATCTCTGGCCTTAT|TGCCTCTGTTGGTCCCTAGTTTACGACATTCAAGCTGTTGGAACATACGCGTGATATAT|TGCTCGCCCTCATTGTTCCCCTTTCAGGCGTCTCACCTTTATTGGACTATAACGCGTGGCTATAT:TGCTCGCCTCTTGTTCCTCTTTACGGACGTCTCAACTTTTGGAACATACGCGTGCTATAT'
]

def extract_estimate(completion):
    # Extract the string between *** ***
    match = re.search(r'\*\*\*(.*?)\*\*\*', completion)
    if match:
        return match.group(1)
    return ""    

def transform_cpred_example(cpred_string):
    # Split the input string into sequences and ground truth
    sequences, ground_truth = cpred_string.split(':')
    
    # Split the sequences by '|'
    sequence_list = sequences.split('|')
    
    # Format the sequences
    formatted_sequences = "\n".join([f"{i+1}. {seq}" for i, seq in enumerate(sequence_list)])
    
    # Combine formatted sequences with the ground truth
    result = f"{formatted_sequences}\nCorrect output:\n{ground_truth}"
    
    return result

def concatenate_examples(list_of_examples):

    prompt = ""
    if len(list_of_examples) == 1:
        prompt += '\nHere is an example:\n'

    else:
        prompt += '\nHere are some examples:\n'

    prompt += "\n".join(list_of_examples)
    return prompt


def load_dataset_from_wandb(artifact_name, entity, project_artifact):
    """
    Load test dataset from W&B artifact.

    Returns:
        List of (idx, reads, ground_truth) tuples
    """
    if not WANDB_AVAILABLE:
        raise ImportError("wandb is required. Install with: pip install wandb")

    wandb.login()
    api = wandb.Api()

    print(f"\nLoading artifact: {entity}/{project_artifact}/{artifact_name}:latest")
    artifact = api.artifact(f"{entity}/{project_artifact}/{artifact_name}:latest", type="dataset")

    download_dir = "./downloaded_artifact_openai"
    artifact_dir = artifact.download(download_dir)

    print(f"Downloaded to: {artifact_dir}")

    # Read raw clusters
    with open(os.path.join(artifact_dir, "reads.txt")) as f:
        reads_lines = [l.strip() for l in f]
    with open(os.path.join(artifact_dir, "ground_truth.txt")) as f:
        gt_lines = [l.strip() for l in f]

    # Parse clusters (separated by ===)
    clusters, current = [], []
    for line in reads_lines:
        if line == "===============================":
            if current:
                clusters.append(current)
                current = []
        else:
            current.append(line)
    if current:
        clusters.append(current)

    assert len(clusters) == len(gt_lines), f"Mismatch: {len(clusters)} clusters vs {len(gt_lines)} ground truths"

    print(f"Loaded {len(clusters)} examples")

    return list(zip(range(len(clusters)), clusters, gt_lines))


def sample_by_cluster_size(dataset, cluster_sizes, samples_per_cluster, seed=42):
    """
    Sample examples evenly across cluster sizes.

    Args:
        dataset: List of (idx, reads, gt) tuples
        cluster_sizes: List of cluster sizes to sample (e.g., [2, 5, 10])
        samples_per_cluster: Number of samples per cluster size
        seed: Random seed

    Returns:
        Dict mapping cluster_size -> list of (idx, reads, gt) tuples
    """
    rng = random.Random(seed)

    # Group by cluster size
    by_cluster_size = defaultdict(list)
    for idx, reads, gt in dataset:
        cluster_size = len(reads)
        by_cluster_size[cluster_size].append((idx, reads, gt))

    print(f"\nDataset composition:")
    for cs in sorted(by_cluster_size.keys()):
        print(f"  Cluster size {cs}: {len(by_cluster_size[cs])} examples")

    # Sample from each requested cluster size
    sampled = {}
    for cs in cluster_sizes:
        if cs not in by_cluster_size:
            print(f"Warning: No examples with cluster size {cs}")
            sampled[cs] = []
            continue

        available = by_cluster_size[cs]
        if len(available) <= samples_per_cluster:
            sampled[cs] = available
            print(f"  Sampled all {len(available)} examples for cluster size {cs}")
        else:
            sampled[cs] = rng.sample(available, samples_per_cluster)
            print(f"  Sampled {samples_per_cluster} examples for cluster size {cs}")

    return sampled


def run_evaluation(args):
    """
    Main evaluation function: load data, run inference, log results.
    """
    print("="*80)
    print("OPENAI MODEL EVALUATION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Cluster sizes: {args.cluster_sizes}")
    print(f"Samples per cluster: {args.samples_per_cluster}")
    print(f"Few-shot examples: {args.n_shots}")
    print(f"Chain-of-thought: {args.enable_thinking}")
    print("="*80)

    # Load dataset from W&B
    dataset = load_dataset_from_wandb(
        args.artifact_name,
        args.entity,
        args.project_artifact
    )

    # Sample by cluster size
    cluster_sizes = [int(x) for x in args.cluster_sizes.split(',')]
    sampled_data = sample_by_cluster_size(
        dataset,
        cluster_sizes,
        args.samples_per_cluster,
        args.seed
    )

    # Initialize W&B run for logging
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
            project=args.project,
            entity=args.entity,
            name=f"{args.model}_gl{args.ground_truth_length}_cs{args.cluster_sizes}",
            config={
                'model': args.model,
                'n_shots': args.n_shots,
                'enable_thinking': args.enable_thinking,
                'cluster_sizes': cluster_sizes,
                'samples_per_cluster': args.samples_per_cluster,
                'ground_truth_length': args.ground_truth_length,
                'artifact_name': args.artifact_name,
            }
        )

    # Create log file if verbose mode is enabled
    log_file = None
    if args.verbose:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"openai_verbose_{args.model}_{timestamp}.log"
        print(f"\nVerbose mode enabled. Logging to: {log_file}")
        # Write header to log file
        with open(log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("OPENAI MODEL EVALUATION - VERBOSE LOG\n")
            f.write("="*80 + "\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Few-shot examples: {args.n_shots}\n")
            f.write(f"Chain-of-thought: {args.enable_thinking}\n")
            f.write(f"Cluster sizes: {args.cluster_sizes}\n")
            f.write(f"Samples per cluster: {args.samples_per_cluster}\n")
            f.write("="*80 + "\n\n")

    # Initialize OpenAI model
    inference_params = {
        'n_shots': args.n_shots,
        'enable_thinking': args.enable_thinking,
        'model': args.model,
        'verbose': args.verbose,
        'log_file': log_file,
    }
    openai_model = OpenAIModel(inference_params)

    # Load few-shot examples (hardcoded for now, could load from file)
    if args.n_shots > 0:
        # Use the hardcoded examples from the test section
        openai_model.examples = HARDCODED_EXAMPLES[:args.n_shots]
        print(f"\nUsing {len(openai_model.examples)} few-shot examples")

    # Run evaluation for each cluster size
    all_results = []
    cluster_metrics = {}

    for cluster_size in cluster_sizes:
        print(f"\n{'='*80}")
        print(f"EVALUATING CLUSTER SIZE {cluster_size}")
        print(f"{'='*80}")

        examples = sampled_data[cluster_size]
        if not examples:
            print(f"No examples for cluster size {cluster_size}, skipping...")
            continue

        results = []
        total_time = 0

        for i, (idx, reads, gt) in enumerate(examples):
            # Convert to CPRED format
            cpred_string = '|'.join(reads) + ':' + gt

            # Run inference
            try:
                result = openai_model.inference(cpred_string)
                candidate = result['candidate_sequence']
                time_taken = result['time_taken']

                # Calculate metrics
                lev_distance = Levenshtein.distance(gt, candidate)
                normalized_lev_distance = lev_distance / len(gt) if len(gt) > 0 else 0.0
                exact_match = (candidate == gt)

                results.append({
                    'idx': idx,
                    'cluster_size': cluster_size,
                    'ground_truth': gt,
                    'prediction': candidate,
                    'levenshtein_distance': lev_distance,
                    'normalized_levenshtein_distance': normalized_lev_distance,
                    'exact_match': exact_match,
                    'time_taken': time_taken,
                })

                total_time += time_taken

                # Progress update every 50 examples
                if (i + 1) % 50 == 0:
                    avg_lev = np.mean([r['levenshtein_distance'] for r in results])
                    avg_norm_lev = np.mean([r['normalized_levenshtein_distance'] for r in results])
                    accuracy = np.mean([r['exact_match'] for r in results])
                    print(f"  Progress: {i+1}/{len(examples)} | Avg Lev: {avg_lev:.2f} | Avg Norm Lev: {avg_norm_lev:.4f} | Accuracy: {accuracy:.2%} | Avg time: {total_time/(i+1):.2f}s")

            except Exception as e:
                print(f"  Error on example {i}: {e}")
                continue

        # Calculate cluster-level metrics
        if results:
            accuracy = np.mean([r['exact_match'] for r in results])
            cluster_metrics[cluster_size] = {
                'mean_levenshtein': np.mean([r['levenshtein_distance'] for r in results]),
                'std_levenshtein': np.std([r['levenshtein_distance'] for r in results]),
                'mean_normalized_levenshtein': np.mean([r['normalized_levenshtein_distance'] for r in results]),
                'std_normalized_levenshtein': np.std([r['normalized_levenshtein_distance'] for r in results]),
                'accuracy': accuracy,
                'failure_rate': 1.0 - accuracy,
                'mean_time': np.mean([r['time_taken'] for r in results]),
                'total_time': total_time,
                'num_examples': len(results),
            }

            print(f"\nCluster size {cluster_size} results:")
            print(f"  Mean Levenshtein: {cluster_metrics[cluster_size]['mean_levenshtein']:.2f} ± {cluster_metrics[cluster_size]['std_levenshtein']:.2f}")
            print(f"  Mean Normalized Levenshtein: {cluster_metrics[cluster_size]['mean_normalized_levenshtein']:.4f} ± {cluster_metrics[cluster_size]['std_normalized_levenshtein']:.4f}")
            print(f"  Accuracy: {cluster_metrics[cluster_size]['accuracy']:.2%}")
            print(f"  Failure Rate: {cluster_metrics[cluster_size]['failure_rate']:.2%}")
            print(f"  Mean time per example: {cluster_metrics[cluster_size]['mean_time']:.2f}s")
            print(f"  Total time: {cluster_metrics[cluster_size]['total_time']:.1f}s")

            all_results.extend(results)

    # Calculate overall metrics
    overall_metrics = {}
    if all_results:
        overall_accuracy = np.mean([r['exact_match'] for r in all_results])
        overall_metrics = {
            'mean_levenshtein': np.mean([r['levenshtein_distance'] for r in all_results]),
            'std_levenshtein': np.std([r['levenshtein_distance'] for r in all_results]),
            'mean_normalized_levenshtein': np.mean([r['normalized_levenshtein_distance'] for r in all_results]),
            'std_normalized_levenshtein': np.std([r['normalized_levenshtein_distance'] for r in all_results]),
            'accuracy': overall_accuracy,
            'failure_rate': 1.0 - overall_accuracy,
            'total_examples': len(all_results),
        }

        print(f"\n{'='*80}")
        print("OVERALL RESULTS")
        print(f"{'='*80}")
        print(f"  Mean Levenshtein: {overall_metrics['mean_levenshtein']:.2f} ± {overall_metrics['std_levenshtein']:.2f}")
        print(f"  Mean Normalized Levenshtein: {overall_metrics['mean_normalized_levenshtein']:.4f} ± {overall_metrics['std_normalized_levenshtein']:.4f}")
        print(f"  Accuracy: {overall_metrics['accuracy']:.2%}")
        print(f"  Failure Rate: {overall_metrics['failure_rate']:.2%}")
        print(f"  Total examples: {overall_metrics['total_examples']}")

    # Log to W&B with keys matching the plotting script format
    if WANDB_AVAILABLE and not args.no_wandb:
        # Log cluster-level metrics in the format expected by plotting script
        # Main metric: avg_levenshtein_N=X (normalized Levenshtein distance)
        log_dict = {}
        for cs, metrics in cluster_metrics.items():
            # Key format matching plotting script: avg_levenshtein_N=2, avg_levenshtein_N=5, etc.
            log_dict[f'avg_levenshtein_N={cs}'] = metrics['mean_normalized_levenshtein']
            log_dict[f'levenshtein_mean_N={cs}'] = metrics['mean_normalized_levenshtein']  # Alternative key
            log_dict[f'accuracy_N={cs}'] = metrics['accuracy']
            log_dict[f'failure_rate_N={cs}'] = metrics['failure_rate']
            log_dict[f'mean_time_N={cs}'] = metrics['mean_time']

        # Log overall metrics
        if overall_metrics:
            log_dict['avg_levenshtein_overall'] = overall_metrics['mean_normalized_levenshtein']
            log_dict['accuracy_overall'] = overall_metrics['accuracy']
            log_dict['failure_rate_overall'] = overall_metrics['failure_rate']
            log_dict['total_examples'] = overall_metrics['total_examples']

        wandb.log(log_dict)

        # Also update run.summary so metrics are accessible via wandb.Api()
        for key, value in log_dict.items():
            wandb.run.summary[key] = value

        # Log results table
        if PANDAS_AVAILABLE:
            wandb.log({"results": wandb.Table(dataframe=pd.DataFrame(all_results))})
        else:
            print("Warning: pandas not available, skipping W&B table logging")

        wandb.finish()

    # Write summary to log file if verbose mode is enabled
    if args.verbose and log_file:
        from datetime import datetime
        with open(log_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("EVALUATION SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for cs, metrics in cluster_metrics.items():
                f.write(f"Cluster size {cs}:\n")
                f.write(f"  Mean Levenshtein: {metrics['mean_levenshtein']:.2f} ± {metrics['std_levenshtein']:.2f}\n")
                f.write(f"  Mean Normalized Levenshtein: {metrics['mean_normalized_levenshtein']:.4f} ± {metrics['std_normalized_levenshtein']:.4f}\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.2%}\n")
                f.write(f"  Failure Rate: {metrics['failure_rate']:.2%}\n")
                f.write(f"  Mean time per example: {metrics['mean_time']:.2f}s\n")
                f.write(f"  Total time: {metrics['total_time']:.1f}s\n")
                f.write(f"  Number of examples: {metrics['num_examples']}\n\n")

            if overall_metrics:
                f.write(f"Overall:\n")
                f.write(f"  Mean Levenshtein: {overall_metrics['mean_levenshtein']:.2f} ± {overall_metrics['std_levenshtein']:.2f}\n")
                f.write(f"  Mean Normalized Levenshtein: {overall_metrics['mean_normalized_levenshtein']:.4f} ± {overall_metrics['std_normalized_levenshtein']:.4f}\n")
                f.write(f"  Accuracy: {overall_metrics['accuracy']:.2%}\n")
                f.write(f"  Failure Rate: {overall_metrics['failure_rate']:.2%}\n")
                f.write(f"  Total examples: {overall_metrics['total_examples']}\n")

            f.write("="*80 + "\n")

        print(f"\nVerbose log saved to: {log_file}")

    # Save results to JSON (optional)
    if args.save_json:
        output_file = f"openai_results_{args.model}_{args.ground_truth_length}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'config': vars(args),
                'overall_metrics': overall_metrics,
                'cluster_metrics': cluster_metrics,
                'all_results': all_results,
            }, f, indent=2)
        print(f"Results saved to: {output_file}")
    print("="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


class OpenAIModel:

    def __init__(self, config):
        self.client = OpenAI()
        #self.client = None

        self.n_shots = config['n_shots']
        self.enable_thinking = config.get('enable_thinking', False)
        self.model = config.get('model', 'gpt-4o-mini')
        self.examples = []
        self.verbose = config.get('verbose', False)
        self.log_file = config.get('log_file', None)

    def inference(self, test_example):

        # process test_example
        ground_truth_sequence = test_example.split(":")[1]
        ground_truth_length = len(ground_truth_sequence)

        noisy_reads = test_example.split(":")[0].split("|")
        cluster_size = len(noisy_reads)

        noisy_reads_prompt = "\n".join([f"{i+1}. {seq}" for i, seq in enumerate(noisy_reads)])

        # hints
        if self.n_shots != 0:

            #print('self.examples')
            #print(len(self.examples))
            #print_list(self.examples)
            sampled_examples = random.sample(self.examples, self.n_shots)
            sampled_examples_prompt = []

            for i, sample_example in enumerate(sampled_examples):
                sampled_examples[i] = f"\nExample #{i+1}\n" + "Input DNA sequences:\n" + transform_cpred_example(sample_example)

            sampled_examples_prompt = concatenate_examples(sampled_examples) + "\n"

        else:
            sampled_examples_prompt = ''

        # Choose final instruction based on enable_thinking
        if self.enable_thinking:
            final_instruction = f"First think step by step about how the input traces align and which positions are reliable. Then provide an estimate of the ground truth DNA sequence consisting of {ground_truth_length} characters in the format ***estimated DNA sequence***, use three * on each side of the estimated DNA sequence."
        else:
            final_instruction = f"Provide an estimate of the ground truth DNA sequence consisting of {ground_truth_length} characters in the format ***estimated DNA sequence***, use three * on each side of the estimated DNA sequence."

        prompt = textwrap.dedent(
f"""We consider a reconstruction problem of DNA sequences. We want to reconstruct a DNA sequence consisting of {ground_truth_length} characters (either A,C,T or G) from {cluster_size} noisy DNA sequences.
These noisy DNA sequences were generated by introducing random errors (insertion, deletion, and substitution of single characters).
The task is to provide an estimate of the ground truth DNA sequence.
{sampled_examples_prompt}
Task:
Reconstruct the DNA sequence from the following noisy input sequences.
Input DNA sequences:
{noisy_reads_prompt}
{final_instruction}
""")

        #print(prompt)
        #return

        start_time = time.time()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        end_time = time.time()
        time_taken = end_time - start_time

        #print(completion.choices[0].message.content)
        output_string = completion.choices[0].message.content
        #print(output_string)
        candidate_sequence = extract_estimate(output_string)

        # Log to file if verbose mode is enabled
        if self.verbose and self.log_file:
            with open(self.log_file, 'a') as f:
                f.write("="*80 + "\n")
                f.write(f"TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"MODEL: {self.model}\n")
                f.write(f"GROUND TRUTH LENGTH: {ground_truth_length}\n")
                f.write(f"CLUSTER SIZE: {cluster_size}\n")
                f.write("="*80 + "\n\n")
                f.write("INPUT PROMPT:\n")
                f.write("-"*80 + "\n")
                f.write(prompt + "\n")
                f.write("-"*80 + "\n\n")
                f.write("OUTPUT RESPONSE:\n")
                f.write("-"*80 + "\n")
                f.write(output_string + "\n")
                f.write("-"*80 + "\n\n")
                f.write(f"EXTRACTED CANDIDATE: {candidate_sequence}\n")
                f.write(f"GROUND TRUTH: {ground_truth_sequence}\n")
                f.write(f"TIME TAKEN: {time_taken:.2f}s\n")
                f.write("\n" + "="*80 + "\n\n")

        # Calculate the Levenshtein distance
        #distance = Levenshtein.distance(ground_truth_sequence, candidate_sequence)
        #print(f"Levenshtein distance: {distance}")
        #print(ground_truth_sequence)
        #print(estimate)
        #print(len(estimate))

        return_dict = {
            'candidate_sequence': candidate_sequence,
            'time_taken': time_taken
        }
        return return_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='OpenAI model inference for DNA sequence reconstruction')

    # Mode selection
    parser.add_argument('--evaluate', action='store_true',
                        help='Run full evaluation mode (loads data from W&B, evaluates multiple cluster sizes), not just simple test.')

    # Model configuration
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='OpenAI model to use (default: gpt-4o-mini). Options: gpt-4o-mini, gpt-4o, gpt-4-turbo, o1-preview, o1-mini, or any future OpenAI model ID')
    parser.add_argument('--enable-thinking', action='store_true',
                        help='Enable chain-of-thought prompting (encourages step-by-step reasoning)')
    parser.add_argument('--n-shots', type=int, default=3,
                        help='Number of few-shot examples (default: 3)')

    # Evaluation mode arguments
    parser.add_argument('--artifact-name', type=str,
                        default='test_dataset_seed34721_gl60_bs800_ds50000',
                        help='W&B artifact name to load test data from')
    parser.add_argument('--entity', type=str, default='<your.wandb.entity>',
                        help='W&B entity (username or team name)')
    parser.add_argument('--project', type=str, default='GPTMini',
                        help='W&B project for logging evaluation results')
    parser.add_argument('--project-artifact', type=str, default='TRACE_RECONSTRUCTION',
                        help='W&B project containing the test data artifacts')
    parser.add_argument('--cluster-sizes', type=str, default='5,10',
                        help='Comma-separated cluster sizes to evaluate (e.g., "2,5,10")')
    parser.add_argument('--samples-per-cluster', type=int, default=100,
                        help='Number of examples to evaluate per cluster size')
    parser.add_argument('--ground-truth-length', type=int, default=60,
                        help='Ground truth sequence length (60, 110, or 180)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible sampling')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable W&B logging (results still saved to JSON)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging (saves prompts and responses to .log file)')
    parser.add_argument('--save-json', action='store_true',
                        help='Save results to JSON file (default: False)')

    args = parser.parse_args()

    # Run evaluation mode or simple test mode
    if args.evaluate:
        run_evaluation(args)
        sys.exit(0)

    # Simple test mode (original behavior)
    use_class = True

    # Example usage
    input_string = "AGGTTCCCTAGAAGTGATGATGGACAGCTGGAATCGCGGGCATATAATTTGTTGCCTTGGTTGCT|GGTCCACTAGAAGTGTCATGGAATGCTGTTCGGGGGCATTCATGTTGTGCTTGGTTGCACT|AGGTTCCTCGAAGGTGATATGGATGCTGTTGCGGGCATGCTACATGTTTGCACATTGAGTTGCA|AGGTCCCTAAGAAGTGTATATGGAGCTCGTTCTGCGGGATCCTAATTGGTTGTGTCCTTCAGGTTAGT|GGTCCCTAGAAGGATTGGATGCTGTTCGCGGGTATCTAATGTTGTGCCTTGGTGCAT|AGGTCGCCCAGAAGTGATATGGTCGCTGTGTCGCGGCATCTAATGTTTGTGACATCTTGATGCT|AGGTTCACCCTAGATAGTGATTGTAGTGATGCATGTTCGCGGGCATCTAATGTTGTGCCTTGGTTGCT|AGGTCCCTAGTAAGTGTATATGGCATGCGGTCGCGGGCTCTAATGTTGTGCCTTGAGTTGCT|AGCGTCCGCTAAGAAGGAATGGATGCTGATTCGCCGGGCATCTAGATGTTGTGCCTTCGGTTGCT|AGGGTCCCCCTACAGAAGTGATATGGATGACTCGCGGGCATCTAAATGGTTGTGGCCTTGTTTGCT:AGGTCCCTAGAAGTGATATGGATGCTGTTCGCGGGCATCTAATGTTGTGCCTTGGTTGCT"
    #output = transform_cpred_example(input_string)
    #print(output)
    #sys.exit()

    ground_truth_length = 60
    n_shots = args.n_shots

    examples = [
        'TGTTCGGGATGGGAGACACCAGAAACCTCGGAAGTAATTCCGCGCATCTGGCCTCCGGG|TGTTTGGGATGGGAGACACCAGGACAACCGCCGGAATAATTTCCGCGCATCTGGCTCCGGG|TGTTTTGGATGGCACACCACCAAGACAACCGCGCGGAGTAATTCCGCGCATCTGGCTGCGGA|TGTTGTGGGATGGGAGACACCTAGACAACCGCGCGGAGTAATCTCCGCGCATCTGGCTCCGGG|TGTTTGGGATTGGAGACCCCGACACCGCGCGGAGTAATTCCGCGCATCTGGCTCCGGG|TGTTTGGGATGGGAGACACCAGACAACCGCGGGGAGTAATTCGCGCAGCTGGCTCCGGG|TGATGTGGGATTGGAGTCACCAGACAACCGGCGGAGTAGGTCCGCCGTGTGGCTCCTG|TGTTTGGGATGGGAGACAACAACAACCGCGCGGAGTAACTCCGCGCATCTGGCCCCGGG|TGTTTGGGCTAGAGACACCAGACAACCGCGCGGAGTAATTCCGGGCATCTGGCTCCGGG|GGTTTGGGATGGGAGACACCAGACAACCGTGCGGAGAAATCCGCGCATCTAGCTCGGG:TGTTTGGGATGGGAGACACCAGACAACCGCGCGGAGTAATTCCGCGCATCTGGCTCCGGG',
        'GGTGATGCCCGCTCGCTTGGAATGCTATACGTACTCCGTGTGGTCGTGACGGGAAGGGATG|GGTGAACCCCGCTCGCTCGTAATGGCTAGACGCAATCCGTAGGTCGTGGCGAGAAGGGATG|GGTGAGCCAGCTGCGCTCCGGAATGGCTATCGCAATCCGTAGGTCGTGACGGGAGAGCGATG|GGTGAGCCCGCTCGGCTTGGAAGTGGCATATACGCATCCGTAGGTCGTGAGCGGGTAAGGGATG|GGTGAGCTCGCTCGAGCTTTTATTCGTTATACGCTATTCGTATGTCGTGTCGGGAAGGGATG|GGCTGAGCCCGCTCGCTTGGAATCGCTCTATACACATTCCGTAGGTCGTGAGCGGGAAGGGAG|GTGTTGCTCGCTCGCTTGGAATGGCTATACGCAATCCGTAGGTCGTGACGGCAAGGGATC|GGTGAGCCTGCTCGCTTGGGATGGCTATACGCAATCAGTATGTCGTTGACGGAATGGGATG|GGTGATCTCGCTCCACTGGAATGGCTATGCGCAAACCCGTAGCTCGTGATGGGAAGGGATG|GGTGAGCTCCGCTTCGTTTGCGAATCCGGCTATAGGCAATCCGTAGCTCGTTGCCGGGACAGGGCTC:GGTGAGCCCGCTCGCTTGGAATGGCTATACGCAATCCGTAGGTCGTGACGGGAAGGGATG',
        'CCACGACATGGTACAAGAAAGGGCCATGAAACCCTCACTCCTTCGGGCTTGCAAATTGC|CCCAGGAAAATGGCAAGAAGGCCATGAAACCCTCACTCTTCCGGGCGTCCGGCAATTGC|CTCCACGGAACATGGTGCAAGAAAGTGCCATGAAACCCTCACTCCTCCGGGCGTGGCAGAATTGC|CACCAGAACATGGGTCAAGAAAGGCCATGCAGAACCTCACTCCTCCGGCGTGGCATACATTGC|CCCACGAACATGGTCCAAAGCAGGCGATGAAACCACTCACTCCTCCGGGCGTGGCAAATTGC|CCACGAACATGGTCAAGAAGGCCATAAACCTCAACTCCTCCGGGCGGTGGCATAATCTGC|CCACGGACATGGTCGAGAAAGCCATGAAACCCTCACTCCTCCGGGCGTGGAAATGC|CCCCAGAACATGGGTCAAGAAAGGCCATGAACCCCACTCCTCCGGGCGTGTCAAATGC|CCCACGAACATGGTCAGAACAGGCCATGAAACCCTCGACTCTCCGGGCGTGGCAAATTC|GCCACATACATGCGTCAATGAAAGGCCTGAAACCTCACTCCTCCGGGCGTGGGCAAATTGA:CCCACGAACATGGTCAAGAAAGGCCATGAAACCCTCACTCCTCCGGGCGTGGCAAATTGC',
        'CGCGCTTGAGTCGAGCGCCTCAACTCCAAGTCTGCTCGATTGCTCGAATTACACTGCCTC|CGGCGTATGAGTCGATGGCCTCCACTCACAAGTCTCTCATATGCTGCCATTACACTGCCTC|TCGTTAGGTCGATGGCGCTCACTCACAAGTCTGGCTCATATGCTGAATGTAACACCCCTC|CGCGTTGAGTCGAGGCCTAATACAGTCTGCTCATTTTGCTGTAATGTAACACTGCCTC|CGCGTTGAGTTCGATGGCCCACTCACAAGTCTGCTCATTAAGCTGAATAGTAACACTGCCTC|CGCGTTGAGTCGATGGCTCCACTCACAAGTCTGCTCATATGCTGAATGTAACACTGCCTC|CGAGGTGAGTCGATGGCCTCAGCTCAAAATCGCTAACATATGCTGAATGAAAACACTGCCATAC|CGCGTTAGTCTGTATGGCCTCACTCACTAAGATCTGTCATATGCTGAACGTAACAGCTGCCTC|ACGCGTTGAGTCGGTAGGCCTCACTCACAAGATTGCTCATATGCTGGATGTAACACTGCCTC|CGCGTTGAGTCGATGGCCTACTCACAAGTCTGCTCATATGCTTAATGTAACCATGCCCT:CGCGTTGAGTCGATGGCCTCACTCACAAGTCTGCTCATATGCTGAATGTAACACTGCCTC',
        'AACTCGAAGGAGAGCGCCATCCATATGGCGGAGCGTTACGTGTCACTCAAGGTTCTT|AAATCGAACGCCGAGAGCTCCATAATAGGGCAGTAGCGGAAGTATCTGTCATCTTAAGGTGCTT|GACGTTGACCATGGATGAGCGCATATCCATCGGGTTTGACGGCGTATCTCCCATTTTAAGTCTT|ACCAGTCGAACTGACGAGCGCGAATCCATAGGGCCTGACGACGTAGTTGTCTATCTTAAGGTTCTT|AAGCCGAACCGGAGAGAGCCATCCCAGGGCCTGGCGCCGATCTGCTCATCGGGTTAAGGTGTCCTT|AGGTCGAACAGAGAGCTCCATCCTTGGGGCCTGACGTGTTTCTCGGTTCATCGTCTTAGGTCATT|ATAGTCGAAGGAGCGCGCCATCCACCAGGGCGTGCGCGTATCCTGTCATCTATAAGGTTCTT|AAGTCCGAAGGGCAGTGCCACCATAAGGGCGTGACGTGTATGTGTCATCTTAAGGTGGTT|ATGTTCGAACGGAGAGCGCATCCGATAGGGCCGTGACGCGTATCTGTCATCTTAAGCGTTATT|AAGATCGGACCGGAGAGCGCCGTTCCCCAGGGCTGTGACCGTAGGATTCTGTCGTGCTTAAGGTTCAT:AAGTCGAACGGAGAGCGCCATCCATAGGGCGTGACGCGTATCTGTCATCTTAAGGTTCTT',
        'AAGCTTGAGTGATGGCCAACGGGTTAAGGCCTATATCAAAACTAGAGGCTAATAAGTC|ACTAGCTGCGGTGATGGCCAACGGCTTAAGGCCTATTCAAACCCTAGGAGCTCATAACTC|AAAGCTGCAGTGATGGCCAACGGCTTAAGGCCTATATCCAACTAGGAGCTCAATAACTC|AAAGGCGCGAGGATGGCCAACGGCGTAAGGCCTATATGAAACTAGGGCTCAATATCTC|AAAGCTTGGGGGGCCAAGGCTTAAGGCCCATATCAAACGTAGGAGCTCAATAACGC|AAAGTTGAGTGATGGCTAACGGCTTATGGCTATATCAAACCAGAGCTCAATAACTA|AAAGCCTGAGTCAGGCCAACGGCTTAAGGCCTAGATCAAACCCTAGGAGCTCAATTAATC|AAAGCTGAGTGGGCCAACGGCTTAAGGCCTATATCAAACCTATGGAGCTCATAACCC|AAAGCGTGAGTGGTCGCCAACGGCTTAAGGCATTCAAACCTAGGAGCTCAATAACTC|AAAGCTTGAGGTGGCCAACGGCTTAAGGCCTTTATCAAACCTAAGGAGCTCAATAACTC:AAAGCTTGAGTGATGGCCAACGGCTTAAGGCCTATATCAAACCTAGGAGCTCAATAACTC',
        'ACAATCGTGGGCATGAGTCCTCAGTATCCTCAGGAATAGGACAAGTTCCACGTGAACTATGGCG|ACATCGGGCAGCGCTCTTTCATACCGTAGAATATTGGCAAGTTTGCCTGTGAACTATGCG|ACATCTGTGGGCAGAGGTCCTCAGACCCTCAGGAATTGGACAAGTTTCCACGTGAAACTAGGCG|CATCGTGGCAGAGTCCTCAGCACCCTCAGGAATTGGACAAGTTTCCAGTGATTCTATGCG|ACTCAGTGGCAGGGTCCTAGACCTCAGGAGATTGGCACAACGATTCCACGTGAACTATGC|ACATCGTGGCAGAATCCTCAGACCCTCAGGAAAGTTGGACAAGTTTCCAGGTGAATCTATGCG|ACATGTGAGCAGAGTCCTCGAACCCTCGGGCCGATTGGAATAAGTTTCCAAAGTGACTATGGG|ACATCGTGGCCTAGGAGTACCTCATGACCCTCATGATTGGACAAGATTCCACGTGAATGTAGG|ACATACCGTGGCAAGTCGTCTTTCATACCCTCAGGGATTGGACATGTGTCACACCGTGATACTGTGCG|ACAGCGTGGCAGAGTCCTCAGCCCCTCAGGAATTGGACAGGTTCCACGGCCAACTATGCG:ACATCGTGGCAGAGTCCTCAGACCCTCAGGAATTGGACAAGTTTCCACGTGAACTATGCG',
        'ATGAACTACCCTCAATCATCTCAACTGCGGAAGTGGAGCCCAGCCCGACACGGGAAAG|ATGTAACATTACCCAATCTCTAGTCACTGCGGAAGTGGAGCCCAGCCCTGATACGGGAACG|ACGTAACTAACCCCAATCGTCGTCAACTGCGGAGTGAGCGCCAGCCACGTACACGCGGAACG|ATGTAACTAACCCAATCATCTTCAACTCGGAAGTGGAGCCCACCCGATGCGGGAAACG|TTGTAACTAACCACAATCAGTCTTCACCTGCGGAGCAGGGAGCCCAGCCCGACACGGGAAACG|ATGTAACTGAAGCCAATCATCTCAACTGCGGAAGTGGAGGCCCACCGACACGGGAAACAG|ATGTAACTAACACAACCATCTCAACTGGGAAGTGGAGCCCAGCCCGACACGGCACAG|ATGTTAACAAACCCAATCATCTTCAACTGGCGGAAGTTGGAGCCCAGTCCAGACACGGGAAACG|ATGTAACCTCCCCAATCGAACTTCAACTGCGGAAGTGGAGCCCAGGCCCCGAACGGGCGAAACG|ATGTACACTACACCCAATCATCTCAACATGCTGAAGTGGATGCCCAGTCCCGAGACGGGAAACG:ATGTAACTAACCCAATCATCTTCAACTGCGGAAGTGGAGCCCAGCCCGACACGGGAAACG',
        'ATGCAACGAATGCTGGCCGGATACATCAAACGATTTCAAGTTATATCCCGTTT|GCCCGACGAATGATGTCCGGTCAGCTACACGTCGTCAAGTATACCGTTAT|ATGCCCGATAATATATGGCGGACTCCACTCTACACGTCGTCAAGTTATATCCCGTTAG|TGCCCCGACGATATGCCGGCGGATACACTCTCACGATCGTCAAGTATATCCGTTAA|ATGCCCGACGCTTCTGGCCGGATACACTCAACAATCGTCACCGTTTATCCGATAA|ATGCCCGACGAATGCTGGCCGGATACACTTACACGATGTCAATGATATCCGAGTG|CGCCCACATATTTGCCGGATACACTTAACATAGTCAAGTAATCGCGTTAT|ATGCCCGAGATATGTGGCCGGCTAGACTTACACGATCTCAGTTAATCCCGTTAT|ATGCCCACGAGTATGCTGCCGGATCCTCACAAATCGTCAAGTTATATCCCGATAT|GCCAGACATAAGCTGGCCGATAAACTGTCACAAACGTCAGTTATCCCGGT:ATGCCCGACGATATGCTGGCCGGATACACTCTACACGATCGTCAAGTTATATCCCGTTAT',
        'CTACCAGGTCGAGGTAATGTGTTCGCATATCCTGACACAGGGCTGTCATGGTTGAACAA|CTACTAGGTCGAGGGAATCAGTTCGCCTGATCGTAACAGATGGGCCGTCATGGTTGAACAT|CTACTAGCCGAGGTAATATGTTCGCTTGATGCCTAACACAGAGCCGTATGGTCGAAAA|CTACTGGGTCGAGGCACTAAGTTCGCTTGAACCTAAGCACAGGGCCGTCAAGGTTGCACAA|CTACTAGGTCGAGGTAATAAGTTCGCTTGATCCTAACACAGGGCCGGCATGGTAGAACAA|CAACTAGGTCGGGACATAATTTCGCTTGATCCTAACACAGGGCCGTCATGGTTGACA|CAACTAGGTCGGTAATAAGTTCGCTGGATCCTAACACAGGGCCGTCATGACTGAACAA|ATACTAGTCGAGGAATAAGATTGTTGATCCCAACACAGGGCTCCGTGATGGTTAAACTTCA|CTACTAGGTCGAGGTAATAAGTTCGCTTGATCCTAACCCAGGGCCGACATGCGTTGAACAA|CTACTACAGGGGGGAAATAAGTTCGCTTGATCCTACAAAGGGGCGTCTGGTTGGACAT:CTACTAGGTCGAGGTAATAAGTTCGCTTGATCCTAACACAGGGCCGTCATGGTTGAACAA',
        'CACTTCGACCGTGTTACGCCGAGGGTTCGTTTCATAAAAGTAGCAAGCGTGATTATCTCATAGC|CACATATCGAACCGGTTACGCCGAGGGTTCGTGAAATAATAGTAGGAGCGTGTTAATTCAAACTGC|CACTTTCGACCGAGTTACGCCGAGGGTTCATTCAATAATTGTAGGAGCGTGTTAATTCACCTGC|CACCTTCGACCGGTTCGTCCGAGGGTATGGATCATTAATAGTAGGAGCGTGTTGATCCACTGC|CGACTTCGACCGGTTACGCCTGCAGCGTTCGTACCTAGTAGTAGGAAGCGTCTTAATTCACTGC|CAAATTCGATCGGTTACGCCGAGGGTCCGTTCTTAATAGTTAGGAAGCGTGTTGATTCACTGC|CACTTCGACCGGTTACTCACGAGGGTTGGTTCTTAATAAGAGGAGCGTGTTACATTCACTGCC|CGACCTTCGACTGGTTACGACCTGAGGGTTCGTTCACATTTAGTAGGAGCGGGTTAATTCACTGC|CACTTCGACCGGTTACGCTCAGAGGGTTCGTTCATAATCCGGTAGGAGCTTCGTTTAAGATCTCATGC|CACTCCGACCTGGTTACGCGAGGGGTCAGTTCATACATAGTAGGAGCCTGTTAATTCACTGGC:CACTTCGACCGGTTACGCCGAGGGTTCGTTCATAATAGTAGGAGCGTGTTAATTCACTGC',
        'AAACCCTTACGGGTCGAATACATCTTATCCGAGCGCCTCAAGGAGTAGCGATTCCTAC|AAACCCATAGGGTCCAAAAATATTTACCGTGCACTCCGAAAGGGAGTATCGTTGATA|AAACACTTGGGGTCGAAAAAATACTATCCGTGTACCCCAGAGGTGTAGTGTCTCATAC|AACCCTGAGGGTCGAATACTGTTTGATCCGTGCACCTCCATGAGGGTGTCGCGGTTCATGC|AAACCTTAGGGCTCGAATACATATTTACCGTGCACCTCCAGAGGAGTAGCGTTTCAA|GAATCCTTAGGGTCGACCACATATTATCCGAGCACCTCCAGAGGAGTAGGTTTCATGAC|TCACCCTTGGCGGTCGAAGCCAATTTATACGTGCAGCTGCAGAGGTCACCGTTTCATAC|ACACCCTTAGGGTCGAAAACATATTTACCGTGCACCTCCAGAGGTGTATCGTTTCATAC|AAACCCCTAGGGTCGAATACATATTTATCCGTGCACCTCCAGAGGAGTTCTTTTATAC|AAACCCGTCGGGTCGAATCCTATTTATCCGTGCACGACCAGAGGAGAATCGTTTCCGAC:AAACCCTTAGGGTCGAATACATATTTATCCGTGCACCTCCAGAGGAGTAGCGTTTCATAC',
        'GTACTTGGCGTGAATACTGCTACAGGGTCGCAGCCCCTGCTTCGTGCGCTCATTGCCATAGGAGCATATG|GTTACTGGCTGTCTTGTAAGGTACCGCAGACCCTGCTTTCGCCTTACACTATTAGGCACATG|GTAGCTTTTGCTGACTGCTAACTCGCAGCCCCGTTCTTGTCTCTCATACATAGGGAGAATG|CTATGGCTGACTGCTAAGGTCGCAGCCCTAAGCTTCCGGGCCTCATGAATCTGGTCTGAATG|TTAACTGGCTGACTAGGCTAAGGTCGCAGCCCCGCTTCGTCCTCTAACCATGGCAAGAAG|GGTACGGGCTGCACAGCTAAGGTGTCAGCCCACGTGCTTCGTGCCTAAACCTCTAGTCAGAAGTG|ATACTGGTGACTGCAAGGTCCAGGCAGCCCCCTGCTTCGTGCCTCATGATCCATAGGCATAATG|GACTGGCGGCTGCTAAGGTCGCAGGCCCCTGCTTCGTGTCCGCATACACATAGGCATGAAAGC|GTACGGCTCTGCCAGAGTCGCCAGCCCGCTGCTTCGTGCCTCATAGAATAGCAGAATG|GTAGGGTGCTGCCAAGGTCGCCGCCCCTGCTTCGTCCCCTAGTACCATAGGCAGAAT:GTACTGGCTGACTGCTAAGGTCGCAGCCCCTGCTTCGTGCCTCATACCATAGGCAGAATG',
        'GATATAGATGTTGCTCGAGAGAATACTGCACAAAGTGTACAGAAGAGATGCTGTAGGAG|GAGAAAGGATTGTACTGAGTGGTACTGTACAAGAGTAAGAAGAGATGCAAAGGTAG|ATAAAGGATCGTTAGCTCGAGTGGATACTGACAAAGAGTCAGAAGAGACGCTAATGGTAG|GAGTAAGATTGTGTGCTCGAGTGAATACCTGTACAAGAGCAAAGAGATGCTAACGGTAG|GTGAAAGGTTGTGCTCGAATTGGAATCATGTGACAAAAATTTCAGAAGAGATGGAAGGTACG|GATACGGATTGTGCTCGAGTGGATACTGGTATAGAGAAGAGAGTAATGCTAAGGTAG|ATATAGGACTGTTCCTCGAAGTGGATACTGTACAAAAATCAGAAGCGAGTAAGGTAG|GATCAGGATTGTACTCGAGTGCTACTGTACAAAGCGTCAGAGGTGCCATAGGTACG|GATAAAGGGACGTTGCCCGAGTGATACTGTCAAAGCGTAAAAGAGATGCTAGGTG|GGATCAAAGGATTGCTTGCTCGAGTGTGATACTGTACAATGATCAGAAGAGATCTAATAG:GATAAAGGATTGTTGCTCGAGTGGATACTGTACAAAGAGTCAGAAGAGATGCTAAGGTAG',
        'TGCTCGCCTCTTTGTTCCTCTCTGTGCAGCTCAACTTTTTAACAACGCTCTATAT|TGCCGCACTCTTGTTCGCTTTAGGGACGGCTCACCTTTTGGAACATAACGCGTCTAATAT|TGCTGCGCCTCTTGTTCTCTTTACGGACGTCTCAACTTTTGTAACATACGCGTGCATAT|TTGCTCGCCTCTTGTCCTTCTTTTAAGACGTCTCAACTCGTGAACATACGCGTGCTATAT|TCTCGCCTCTTGTCCTCTTTACGGCACTCAAACTTTTGGAACAACGCGTGTTATTT|GCCGCCTATGATCCTCTTGAACGACGTCACAATTTGGAAGCCATACGCGTGCTATAT|TCTTGGCACTCTGCTCTCTTTACGTGCCATTCACTTTTGGGTACATAGCCGTGCTATGA|TGCACGCCTCTTGTTCCTTTTTCGGACGGCGCAACTTTTGAAACATCTCTGGCCTTAT|TGCCTCTGTTGGTCCCTAGTTTACGACATTCAAGCTGTTGGAACATACGCGTGATATAT|TGCTCGCCCTCATTGTTCCCCTTTCAGGCGTCTCACCTTTATTGGACTATAACGCGTGGCTATAT:TGCTCGCCTCTTGTTCCTCTTTACGGACGTCTCAACTTTTGGAACATACGCGTGCTATAT'
    ]

    if use_class:
        # Create log file if verbose mode is enabled
        log_file = None
        if args.verbose:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"openai_verbose_{args.model}_{timestamp}.log"
            print(f"Verbose mode enabled. Logging to: {log_file}")
            with open(log_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("OPENAI MODEL INFERENCE - VERBOSE LOG (TEST MODE)\n")
                f.write("="*80 + "\n")
                f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {args.model}\n")
                f.write(f"Few-shot examples: {n_shots}\n")
                f.write(f"Chain-of-thought: {args.enable_thinking}\n")
                f.write("="*80 + "\n\n")

        inference_params = {
            'n_shots': n_shots,
            'enable_thinking': args.enable_thinking,
            'model': args.model,
            'verbose': args.verbose,
            'log_file': log_file,
        }
        openai_model = OpenAIModel(inference_params)
        openai_model.examples = examples
        return_dict = openai_model.inference(input_string)

        candidate_sequence = return_dict['candidate_sequence']
        ground_truth_sequence = input_string.split(":")[1]

        print('gt:')
        print(ground_truth_sequence)
       
        #candidate_sequence = candidate_sequence[:ground_truth_length]
        print('candidate:')
        print(candidate_sequence)

        levenshtein_distance = Levenshtein.distance(ground_truth_sequence, candidate_sequence)
        print(f"Levenshtein distance: {levenshtein_distance}")

        #AGGTTCCTCGAAGGTGATGATGGACAGCTGGAATCGCGGGCATATAATTTGTTGCCTTGGTTGCT
        #AGGTCCCTAGAAGTGATATGGATGCTGTTCGCGGGCATCTAATGTTGTGCCTTGGTTGCT


    else: 
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"""We consider a trace reconstruction of DNA sequences. We want to reconstruct a DNA sequence of length {ground_truth_length} from {cluster_size} noisy DNA sequences.
                                The noisy DNA sequences are obtained by introducing insertion, deletion, and substitution errors in the ground truth sequence. The error probabilities fall within the following ranges:
                                - Insertion error probability: [0.01,0.1]
                                - Deletion error probability: [0.01,0.1]
                                - Substitution error probability: [0.01,0.1]

                                The reconstructed sequence should be as similar as possible to the ground truth sequence.

                                Here is an example: 
                                Input DNA sequences: 
                                1. AGGTTCCCTAGAAGTGATGATGGACAGCTGGAATCGCGGGCATATAATTTGTTGCCTTGGTTGCT
                                2. GGTCCACTAGAAGTGTCATGGAATGCTGTTCGGGGGCATTCATGTTGTGCTTGGTTGCACT
                                3. AGGTTCCTCGAAGGTGATATGGATGCTGTTGCGGGCATGCTACATGTTTGCACATTGAGTTGCA
                                4. AGGTCCCTAAGAAGTGTATATGGAGCTCGTTCTGCGGGATCCTAATTGGTTGTGTCCTTCAGGTTAGT
                                5. GGTCCCTAGAAGGATTGGATGCTGTTCGCGGGTATCTAATGTTGTGCCTTGGTGCAT
                                6. AGGTCGCCCAGAAGTGATATGGTCGCTGTGTCGCGGCATCTAATGTTTGTGACATCTTGATGCT
                                7. AGGTTCACCCTAGATAGTGATTGTAGTGATGCATGTTCGCGGGCATCTAATGTTGTGCCTTGGTTGCT
                                8. AGGTCCCTAGTAAGTGTATATGGCATGCGGTCGCGGGCTCTAATGTTGTGCCTTGAGTTGCT
                                9. AGCGTCCGCTAAGAAGGAATGGATGCTGATTCGCCGGGCATCTAGATGTTGTGCCTTCGGTTGCT
                                10 .AGGGTCCCCCTACAGAAGTGATATGGATGACTCGCGGGCATCTAAATGGTTGTGGCCTTGTTTGCT
                                Correct output:
                                AGGTCCCTAGAAGTGATATGGATGCTGTTCGCGGGCATCTAATGTTGTGCCTTGGTTGCT
                                
                                Task:
                                Reconstruct the DNA sequence from the following noisy input sequences. 
                                Input DNA sequences:
                                1. AGGTTCCCTAGAAGTGATGATGGACAGCTGGAATCGCGGGCATATAATTTGTTGCCTTGGTTGCT
                                2. GGTCCACTAGAAGTGTCATGGAATGCTGTTCGGGGGCATTCATGTTGTGCTTGGTTGCACT
                                3. AGGTTCCTCGAAGGTGATATGGATGCTGTTGCGGGCATGCTACATGTTTGCACATTGAGTTGCA
                                4. AGGTCCCTAAGAAGTGTATATGGAGCTCGTTCTGCGGGATCCTAATTGGTTGTGTCCTTCAGGTTAGT
                                5. GGTCCCTAGAAGGATTGGATGCTGTTCGCGGGTATCTAATGTTGTGCCTTGGTGCAT
                                6. AGGTCGCCCAGAAGTGATATGGTCGCTGTGTCGCGGCATCTAATGTTTGTGACATCTTGATGCT
                                7. AGGTTCACCCTAGATAGTGATTGTAGTGATGCATGTTCGCGGGCATCTAATGTTGTGCCTTGGTTGCT
                                8. AGGTCCCTAGTAAGTGTATATGGCATGCGGTCGCGGGCTCTAATGTTGTGCCTTGAGTTGCT
                                9. AGCGTCCGCTAAGAAGGAATGGATGCTGATTCGCCGGGCATCTAGATGTTGTGCCTTCGGTTGCT
                                10 .AGGGTCCCCCTACAGAAGTGATATGGATGACTCGCGGGCATCTAAATGGTTGTGGCCTTGTTTGCT
                                Provide an estimate of the true DNA sequence of length {ground_truth_length} in the format ***estimated DNA sequence*** - use three * on each side of the estimated DNA sequence."""
                }
            ]
        )

        print(completion.choices[0].message.content)

        output_string = completion.choices[0].message.content

        estimate = extract_estimate(output_string)

        if estimate is None:
            raise Exception("Could not extract estimate")

        gt = 'AGGTCCCTAGAAGTGATATGGATGCTGTTCGCGGGCATCTAATGTTGTGCCTTGGTTGCT'


        print(Levenshtein.distance(estimate, gt))
        #print(completion.choices[0].message)
        #print(completion)