import os
import sys
import time
import subprocess
from subprocess import DEVNULL
import argparse
import wandb
import tqdm
import concurrent.futures
import numpy as np
from collections import defaultdict
from datetime import datetime
import pickle
from Levenshtein import distance as levenshtein_distance
from src.utils.hamming_distance import hamming_distance_postprocessed
from src.utils.helper_functions import create_fasta_file, read_fasta, contaminate_trace_cluster
from src.eval_pkg.reconstruction_algorithms.trellis_reconstruction.algorithms.trellis_bma import TrellisBMAParams, compute_trellis_bma_estimation
from src.eval_pkg.reconstruction_algorithms.VSAlgorithm.mainVS import alg
from src.eval_pkg.majority_vote import majority_merge

# GLOBAL CONFIG 
ENTITY = "<your.wandb.entity>"
PROJECT_ARTIFACT = "TRACE_RECONSTRUCTION"
DOWNLOAD_DIR = "./downloaded_artifact_new"

# HELPERS
def load_dataset(artifact_name=None, local_data_dir=None):
    """
    Load dataset either from wandb artifact or local directory.

    Args:
        artifact_name: Name of wandb artifact to download (ignored if local_data_dir is provided)
        local_data_dir: Path to local directory containing reads.txt and ground_truth.txt

    Returns:
        List of tuples (index, cluster_reads, ground_truth)
    """
    if local_data_dir:
        # Use local data directory
        artifact_dir = local_data_dir
        print(f"Loading dataset from local directory: {artifact_dir}")

        # Verify required files exist
        reads_path = os.path.join(artifact_dir, "reads.txt")
        gt_path = os.path.join(artifact_dir, "ground_truth.txt")

        if not os.path.exists(reads_path):
            raise FileNotFoundError(f"reads.txt not found in {artifact_dir}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"ground_truth.txt not found in {artifact_dir}")
    else:
        # Download from wandb artifact
        if artifact_name is None:
            raise ValueError("Either artifact_name or local_data_dir must be provided")

        wandb.login()
        api = wandb.Api()
        artifact = api.artifact(f"{ENTITY}/{PROJECT_ARTIFACT}/{artifact_name}:latest", type="dataset")
        artifact_dir = artifact.download(DOWNLOAD_DIR)

    # Read raw clusters
    with open(os.path.join(artifact_dir, "reads.txt")) as f: # reads_cleaned.txt, reads.txt
        reads_lines = [l.strip() for l in f]
    with open(os.path.join(artifact_dir, "ground_truth.txt")) as f: # ground_truth.txt, ground_truth_cleaned.txt
        gt_lines = [l.strip() for l in f]

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
    assert len(clusters) == len(gt_lines)
    return list(zip(range(len(clusters)), clusters, gt_lines))


def save_checkpoint(checkpoint_path, checkpoint_data):
    """Save checkpoint to disk"""
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path):
    """Load checkpoint if exists, else return None"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return None


def generate_temp_evyat_file(reads, gt, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "evyat.txt")
    with open(path, 'w') as f:
        f.write(gt + "\n****\n")
        f.writelines(r + "\n" for r in reads)


def read_evyat(path):
    if not os.path.exists(path):
        return [], [], []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    gts, preds, dists = [], [], []
    i = 0
    while i < len(lines):
        if lines[i].startswith("Cluster Num") and i+3 < len(lines):
            gts.append(lines[i+1]); preds.append(lines[i+2])
            d = lines[i+3].split(":")[1].strip() if lines[i+3].startswith("Distance:") else -1
            dists.append(int(d))
            i += 4
        else:
            i += 1
    return gts, preds, dists

# ALGORITHM WRAPPERS 
class BMALA:
    def __init__(self, temp_dir):
        self.base   = temp_dir
        # assume cwd is /TReconLM
        self.binary = os.path.join(os.getcwd(), "src", "eval_pkg", "BMALA")
        # make sure it is executable on startup
        os.chmod(self.binary, 0o755)

    def inference(self, reads, gt, idx):
        # Add process ID to avoid conflicts with parallel workers
        folder = os.path.join(self.base, f"cluster_{os.getpid()}_{idx}")
        generate_temp_evyat_file(reads, gt, folder)
        cmd = f"{self.binary} {folder}/evyat.txt {folder} > {folder}/out.txt"
        subprocess.run(cmd, shell=True, check=True)
        _, succ, _ = read_evyat(os.path.join(folder, 'output-results-success.txt'))
        pred = succ[0] if succ else read_evyat(os.path.join(folder, 'output-results-fail.txt'))[1][0]
        subprocess.run(f"rm -rf {folder}", shell=True, check=True)
        return pred


class Iterative(BMALA):
    def __init__(self, temp_dir):
        super().__init__(temp_dir)
        # override to point at the Iterative binary
        self.binary = os.path.join(os.getcwd(), "src", "eval_pkg", "Iterative")
        os.chmod(self.binary, 0o755)

    def inference(self, reads, gt, idx):
        # Add process ID to avoid conflicts with parallel workers
        folder = os.path.join(self.base, f"cluster_{os.getpid()}_{idx}")
        generate_temp_evyat_file(reads, gt, folder)
        cmd = f"{self.binary} {folder}/evyat.txt {folder} > {folder}/out.txt"
        subprocess.run(cmd, shell=True, check=True)
        _, succ, _ = read_evyat(os.path.join(folder, 'output-results-success.txt'))
        pred = succ[0] if succ else read_evyat(os.path.join(folder, 'output-results-fail.txt'))[1][0]
        subprocess.run(f"rm -rf {folder}", shell=True, check=True)
        return pred

class MuscleAlgorithm:
    def __init__(self, temp_dir):
        self.temp   = temp_dir
        self.binary = os.path.join(os.getcwd(), "src", "eval_pkg", "muscle")
        os.chmod(self.binary, 0o755)

    # make the order (reads, gt, idx) like all the others
    def inference(self, reads, gt, idx):
        # guard against empty clusters
        assert reads, f"Cluster {idx} is empty!"

        # Add process ID to avoid conflicts with parallel workers
        inp = os.path.join(self.temp, f"in_{os.getpid()}_{idx}.fasta")
        out = os.path.join(self.temp, f"out_{os.getpid()}_{idx}.fasta")
        create_fasta_file(reads, 'obs', inp)

        # run MUSCLE and keep stderr for debugging
        result = subprocess.run(
            [self.binary, "-align", inp, "-output", out],
            stdout=DEVNULL, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MUSCLE failed for cluster {idx} (exit {result.returncode}):\n"
                f"{result.stderr}"
            )

        # check that the alignment file exists and is non-empty
        if not os.path.isfile(out) or os.path.getsize(out) == 0:
            raise FileNotFoundError(
                f"MUSCLE did not create {out} for cluster {idx}"
            )

        seqs = read_fasta(out)
        if not seqs:
            raise ValueError(
                f"MUSCLE produced an empty alignment for cluster {idx}"
            )

        pred = majority_merge(seqs, weight=0.4)

        os.remove(inp)
        os.remove(out)
        return pred



class TrellisBMAAlgorithm:
    def __init__(self, temp_dir, P_INS=0.055, P_DEL=0.055, P_SUB=0.055, k=0):
        self.base  = temp_dir
        self.k     = k
        self.P_INS = P_INS
        self.P_DEL = P_DEL
        self.P_SUB = P_SUB

    def _select_beta_parameters(self, cluster_size):
        # clamp cluster size to [2,10]
        size = max(2, min(cluster_size, 10))
        k = self.k

        # parameter sets by cluster size
        if size in [2, 3]:
            beta_b, beta_e, beta_i = 0.0, 0.1, 0.5
        elif size in [4, 5]:
            beta_b, beta_e, beta_i = 0.0, 1.0, 0.1
        elif size in [6, 7]:
            beta_b, beta_e, beta_i = 0.0, 0.5, 0.1
        elif size in [8, 9]:
            beta_b, beta_e, beta_i = 0.0, 0.5, 0.5
        else:  # size == 10
            beta_b, beta_e, beta_i = 0.0, 0.5, 0.0

        return {
            'beta_b': beta_b,
            'beta_e': beta_e,
            'beta_i': beta_i,
            'P_INS': self.P_INS, #+ k * 0.005,
            'P_DEL': self.P_DEL, # + k * 0.005,
            'P_SUB': self.P_SUB, # + k * 0.005,
        }

    def select_beta(self, cluster_size):

        params = self._select_beta_parameters(cluster_size)
        return TrellisBMAParams(**params)

    def inference(self, reads, gt, idx):

        params = self.select_beta(len(reads))
        _, pred = compute_trellis_bma_estimation(reads, gt, params)
        return pred


class VSAlgorithm:
    def __init__(self, P_SUB=0.055, k=0):
        self.cfg = {'gamma': 0.75, 'l': 5, 'r': 2, 'P_SUB': P_SUB + k * 0.005}

    def inference(self, reads, gt=None, idx=None):
        return alg(len(reads), reads, self.cfg['l'], (1+self.cfg['P_SUB'])/2, self.cfg['r'], self.cfg['gamma'], gt)

# map names to classes
ALGS = {
    'bmala': BMALA,
    'itr': Iterative,
    'muscle': MuscleAlgorithm,
    'trellisbma': TrellisBMAAlgorithm,
    'vs': VSAlgorithm,
}

ERROR_PROFILES = {
    'default':  { 'P_INS': 0.055,  'P_DEL': 0.055,  'P_SUB': 0.055 },
    'microsoft':{ 'P_INS': 0.017,  'P_DEL': 0.02,   'P_SUB': 0.022 },
    'noisy':    { 'P_INS': 0.057,  'P_DEL': 0.06,   'P_SUB': 0.026 },
    'grass':    { 'P_INS': 0.0009, 'P_DEL': 0.0099, 'P_SUB': 0.0056 },
    'chandak':  { 'P_INS': 0.035,  'P_DEL': 0.035,  'P_SUB': 0.035 },
}


def process_example(args):
    idx, reads, gt, alg_name, alg_params = args
    try:
        # Create algorithm instance inside worker to avoid pickling issues
        if alg_name == 'trellisbma':
            alg_inst = TrellisBMAAlgorithm(
                temp_dir=alg_params['temp_dir'],
                P_INS=alg_params['P_INS'],
                P_DEL=alg_params['P_DEL'],
                P_SUB=alg_params['P_SUB'],
                k=alg_params['k']
            )
        elif alg_name == 'vs':
            alg_inst = VSAlgorithm(
                P_SUB=alg_params['P_SUB'],
                k=alg_params['k']
            )
        elif alg_name == 'bmala':
            alg_inst = BMALA(alg_params['temp_dir'])
        elif alg_name == 'itr':
            alg_inst = Iterative(alg_params['temp_dir'])
        elif alg_name == 'muscle':
            alg_inst = MuscleAlgorithm(alg_params['temp_dir'])
        else:
            raise ValueError(f"Unknown algorithm: {alg_name}")

        start = time.perf_counter()
        pred = alg_inst.inference(reads, gt, idx)
        elapsed = time.perf_counter() - start
        return {
            'idx': idx,
            'ground_truth': gt,
            'reconstructed': pred,
            'hamming_distance': hamming_distance_postprocessed(gt, pred),
            'levenshtein_distance': levenshtein_distance(gt, pred)/len(gt),
            'time_taken': elapsed,
            'num_reads': len(reads),
        }
    except Exception as e:
        print(f"Error processing example {idx} with {alg_name}: {e}")
        import traceback
        traceback.print_exc()
        # Return a failed result instead of crashing
        return {
            'idx': idx,
            'ground_truth': gt,
            'reconstructed': gt,  # Use ground truth as fallback
            'hamming_distance': 1.0,
            'levenshtein_distance': 1.0,
            'time_taken': 0.0,
            'num_reads': len(reads),
        }



def run_timing_measurement(args, dataset, rates, alg_params):
    """
    Run throughput measurement by cycling through dataset for fixed time windows.

    This function measures pure algorithm throughput by repeatedly processing
    examples for a fixed duration, then reporting examples/hour with statistics.

    Args:
        args: Parsed command-line arguments
        dataset: List of (idx, reads, gt) tuples
        rates: Error profile dictionary
        alg_params: Algorithm-specific parameters
    """
    from itertools import cycle
    from collections import defaultdict
    import random

    run_duration = args.timing_duration
    num_runs = args.timing_runs
    warmup_runs = args.timing_warmup
    num_workers = args.workers
    timing_seed = getattr(args, 'timing_seed', 42)

    print(f"\n{'='*80}")
    print("TIMING MODE: THROUGHPUT MEASUREMENT")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Algorithm: {args.alg}")
    print(f"  Run duration: {run_duration / 60:.1f} minutes ({run_duration}s)")
    print(f"  Number of runs: {num_runs} ({warmup_runs} warmup + {num_runs - warmup_runs} measured)")
    print(f"  Workers (parallel processes): {num_workers}")
    print(f"  Dataset size: {len(dataset)} examples")
    print(f"")

    # Create representative subset for timing
    # Sample evenly across observation_size (cluster size) for realistic throughput
    print(f"Creating representative timing subset:")

    rng = random.Random(timing_seed)

    # Group by observation_size (cluster size = number of reads)
    by_obs_size = defaultdict(list)
    for item in dataset:
        idx, reads, gt = item
        obs_size = len(reads)  # cluster_size = observation_size
        by_obs_size[obs_size].append(item)

    print(f"  Found observation sizes: {sorted(by_obs_size.keys())}")
    for obs in sorted(by_obs_size.keys()):
        print(f"    Obs size {obs}: {len(by_obs_size[obs])} examples")

    # Sample equal number from each observation_size
    samples_per_obs = getattr(args, 'timing_samples_per_obs', 400)  # Default 400 per obs size
    timing_subset = []

    for obs in sorted(by_obs_size.keys()):
        items = by_obs_size[obs]
        if len(items) <= samples_per_obs:
            # Take all if fewer than requested
            sampled = items
        else:
            # Randomly sample
            sampled = rng.sample(items, samples_per_obs)

        timing_subset.extend(sampled)
        # Calculate total input length for each example
        lengths = [sum(len(r) for r in item[1]) for item in sampled]
        print(f"  Sampled {len(sampled)} from obs_size={obs}, total input lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

    print(f"  Total timing subset: {len(timing_subset)} examples")

    # Sort subset by total input length for efficient batching
    timing_subset.sort(key=lambda item: sum(len(r) for r in item[1]))
    print(f"  Sorted by total input length for efficient batching")
    print(f"  This subset will be cycled for all timing runs (equal representation of all obs sizes)")
    print(f"")

    # Create initial cycle (will be reset for each measured run)
    data_cycle = cycle(timing_subset)

    # Storage for results
    all_throughputs = []
    all_example_counts = []
    all_durations = []

    # Run timing windows
    for run_idx in range(num_runs):
        is_warmup = run_idx < warmup_runs
        run_label = "WARMUP" if is_warmup else f"RUN {run_idx - warmup_runs + 1}"

        # Reset cycle at start of each measured run (not warmup)
        # This ensures all measured runs see the same data for consistent results
        if not is_warmup:
            data_cycle = cycle(timing_subset)

        print(f"\n{run_label}: Starting {run_duration / 60:.1f} minute timing window...")

        run_start = time.perf_counter()
        examples_this_run = 0

        # Process examples until time limit
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as exe:
            # Queue to track submitted futures
            futures = []

            # Keep submitting until time is up
            while True:
                elapsed = time.perf_counter() - run_start
                if elapsed >= run_duration:
                    break

                # Submit a batch of work (num_workers examples)
                batch_futures = []
                for _ in range(num_workers):
                    idx, reads, gt = next(data_cycle)
                    future = exe.submit(process_example, (idx, reads, gt, args.alg, alg_params))
                    batch_futures.append(future)

                futures.extend(batch_futures)

                # Wait for this batch to complete before submitting next batch
                concurrent.futures.wait(batch_futures)
                examples_this_run += len(batch_futures)

                # Log progress every 100 examples
                if examples_this_run % 100 == 0:
                    current_elapsed = time.perf_counter() - run_start
                    current_rate = (examples_this_run / current_elapsed) * 3600
                    print(f"  [{run_label}] Progress: {examples_this_run} examples in {current_elapsed:.1f}s → {current_rate:.0f} ex/hr (current)")

        # Compute final timing for this run
        run_end = time.perf_counter()
        run_elapsed = run_end - run_start
        throughput = (examples_this_run / run_elapsed) * 3600  # examples per hour

        print(f"  [{run_label}] Completed: {examples_this_run} examples in {run_elapsed:.1f}s → {throughput:.0f} ex/hr")

        # Store results (skip warmup)
        if not is_warmup:
            all_throughputs.append(throughput)
            all_example_counts.append(examples_this_run)
            all_durations.append(run_elapsed)

    # Compute statistics
    mean_throughput = np.mean(all_throughputs)
    std_throughput = np.std(all_throughputs)
    cv_throughput = (std_throughput / mean_throughput) * 100  # CV as percentage

    total_examples = sum(all_example_counts)
    total_time = sum(all_durations)

    print(f"\n{'='*80}")
    print("TIMING RESULTS")
    print(f"{'='*80}")
    print(f"Measured runs: {len(all_throughputs)}")
    print(f"Individual throughputs: {[f'{t:.0f}' for t in all_throughputs]} ex/hr")
    print(f"")
    print(f"Mean throughput: {mean_throughput:.0f} ex/hr")
    print(f"Std throughput: {std_throughput:.0f} ex/hr")
    print(f"Coefficient of Variation (CV): {cv_throughput:.2f}%")
    print(f"")
    print(f"Total examples processed: {total_examples}")
    print(f"Total time (measured runs): {total_time / 60:.1f} minutes")
    print(f"{'='*80}")

    # Log to WandB
    wandb.log({
        'timing_mean_throughput_per_hour': mean_throughput,
        'timing_std_throughput_per_hour': std_throughput,
        'timing_cv_throughput_percent': cv_throughput,
        'timing_total_examples': total_examples,
        'timing_total_time_minutes': total_time / 60,
        'timing_num_measured_runs': len(all_throughputs),
    })

    # Log individual run throughputs
    for i, throughput in enumerate(all_throughputs):
        wandb.log({f'timing_run_{i+1}_throughput': throughput})


# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', choices=ALGS.keys(), required=True)
    parser.add_argument(
        '--artifact',
        default="test_dataset_seed34721_gl110_bs1500_ds50000",
        help="Which W&B artifact to evaluate (not used if --local-data-dir is provided)"
    )
    parser.add_argument(
        '--local-data-dir',
        type=str,
        default=None,
        help="Path to local data directory containing reads.txt and ground_truth.txt (skips wandb download)"
    )
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--project', type=str, default='Timing')

    parser.add_argument(
        '--error-profile',
        choices=['default','microsoft','noisy','grass','chandak'],
        default='default',
        help="Select insertion/deletion/substitution rates"
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='If set, run k=0..max-k instead of a single run'
    )
    parser.add_argument(
        '--max-k',
        type=int,
        default=10,
        help='When --sweep, iterate k from 0 to this (inclusive)'
    )
    parser.add_argument(
        '--subset',
        action='store_true',
        help='If set, only use the first 20% of the dataset'
    )
    parser.add_argument(
        '--misclustering',
        action='store_true',
        help='If set, run misclustering robustness experiment'
    )
    parser.add_argument(
        '--contamination-rates',
        type=str,
        default='0.12,0.15,0.18,0.2',
        help='Comma-separated contamination rates for misclustering experiment'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=365,
        help='Random seed for contamination'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of examples to sample (for faster testing)'
    )
    parser.add_argument(
        '--sampling-seed',
        type=int,
        default=42,
        help='Random seed for sampling subset of data'
    )
    parser.add_argument(
        '--timing',
        action='store_true',
        help='Enable timing mode for throughput measurement'
    )
    parser.add_argument(
        '--timing-duration',
        type=int,
        default=300,
        help='Duration of each timing run in seconds (default: 300s = 5 min)'
    )
    parser.add_argument(
        '--timing-runs',
        type=int,
        default=6,
        help='Total number of timing runs including warmup (default: 5)'
    )
    parser.add_argument(
        '--timing-warmup',
        type=int,
        default=1,
        help='Number of warmup runs to discard (default: 1)'
    )
    parser.add_argument(
        '--timing-seed',
        type=int,
        default=42,
        help='Random seed for timing subset sampling (default: 42)'
    )
    parser.add_argument(
        '--timing-samples-per-obs',
        type=int,
        default=50,
        help='Number of examples to sample per observation size for timing (default: 50)'
    )

    args = parser.parse_args()

    # pick the rates dictionary based on the flag
    rates = ERROR_PROFILES[args.error_profile]

    # if sweep, run k=0..max_k; otherwise exactly one pass (k=None)
    sweep_range = range(args.max_k + 1) if args.sweep else [None]
    base_seed   = 34721

    for k in sweep_range:
        seed = base_seed #+ (k or 0) depending if different seed for each k

        if args.local_data_dir:
            # Use local data directory (sweep mode not applicable)
            art = None
            run_name = f"{args.alg}_local_data"
        elif args.sweep:
            art = f"sweep{k}_seed{seed}_gl110_bs1500_ds5000"
            run_name = f"{args.alg}_sweep{k}_seed{seed}"
        else:
            art = args.artifact
            run_name = f"{args.alg}_{args.artifact}"

        # load the dataset (and optionally take only the first 20%)
        dataset = load_dataset(artifact_name=art, local_data_dir=args.local_data_dir)
        if args.subset:
            cut = max(1, int(0.2 * len(dataset)))
            print(f"--subset set, using first {cut} / {len(dataset)} examples")
            dataset = dataset[:cut]

        # Optional: Sample a subset of examples for faster testing
        if args.max_samples is not None and args.max_samples < len(dataset):
            print(f"\n{'='*80}")
            print(f"SAMPLING SUBSET OF DATA")
            print(f"{'='*80}")
            print(f"Total examples available: {len(dataset)}")
            print(f"Max samples requested: {args.max_samples}")
            print(f"Random seed: {args.sampling_seed}")

            rng = np.random.RandomState(args.sampling_seed)
            sampled_indices = rng.choice(len(dataset), size=args.max_samples, replace=False)
            sampled_indices = sorted(sampled_indices)
            dataset = [dataset[i] for i in sampled_indices]
            print(f"Sampled {len(dataset)} examples for inference")
            print(f"{'='*80}\n")

        # Check if timing mode is enabled
        if args.timing:
            # Prepare algorithm parameters
            if args.alg == 'trellisbma':
                alg_params = {
                    'temp_dir': DOWNLOAD_DIR,
                    'P_INS': rates['P_INS'],
                    'P_DEL': rates['P_DEL'],
                    'P_SUB': rates['P_SUB'],
                    'k': (k or 0)
                }
            elif args.alg == 'vs':
                alg_params = {
                    'P_SUB': rates['P_SUB'],
                    'k': (k or 0)
                }
            else:
                alg_params = {'temp_dir': DOWNLOAD_DIR}

            # Initialize WandB for timing run
            wandb.init(
                entity=ENTITY,
                project=args.project,
                name=f"{run_name}_timing"
            )

            # Run timing measurement
            run_timing_measurement(args, dataset, rates, alg_params)

            # Finish WandB and exit
            wandb.finish()
            return  # Exit early after timing measurement

        # If misclustering is enabled, run contamination experiment
        if args.misclustering:
            contamination_rates = [float(r.strip()) for r in args.contamination_rates.split(',')]
            print(f"Misclustering experiment enabled with rates: {contamination_rates}")

            # Create simple config object for contaminate_trace_cluster
            # Use SAME contamination distribution as DNAFormer/RobuSeqNet for fair comparison
            # Sample uniformly from [0.01, 0.1] (same as deep learning baselines)
            from types import SimpleNamespace
            cfg = SimpleNamespace()
            cfg.data = SimpleNamespace()
            cfg.data.insertion_probability_lb = 0.01
            cfg.data.insertion_probability_ub = 0.1
            cfg.data.deletion_probability_lb = 0.01
            cfg.data.deletion_probability_ub = 0.1
            cfg.data.substitution_probability_lb = 0.01
            cfg.data.substitution_probability_ub = 0.1

            print(f"  Contamination error rate sampling ranges (sampled uniformly per contaminant):")
            print(f"    INS=[{cfg.data.insertion_probability_lb:.3f}, {cfg.data.insertion_probability_ub:.3f}], "
                  f"DEL=[{cfg.data.deletion_probability_lb:.3f}, {cfg.data.deletion_probability_ub:.3f}], "
                  f"SUB=[{cfg.data.substitution_probability_lb:.3f}, {cfg.data.substitution_probability_ub:.3f}]")
            print(f"  NOTE: Baseline algorithm uses '{args.error_profile}' profile (INS={rates['P_INS']:.3f}, "
                  f"DEL={rates['P_DEL']:.3f}, SUB={rates['P_SUB']:.3f})")

            # Setup checkpoint
            checkpoint_dir = os.path.join(DOWNLOAD_DIR, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"{args.alg}_{art}_misclustering.pkl")

            # Try to load existing checkpoint
            checkpoint = load_checkpoint(checkpoint_path)

            if checkpoint:
                print(f"Resuming from checkpoint: {checkpoint_path}")
                print(f"   Completed rates: {checkpoint['completed_rates']}")
                wandb_run_id = checkpoint['wandb_run_id']
                completed_rates = checkpoint['completed_rates']
                all_results = checkpoint['results_per_rate']
            else:
                print(f"Starting fresh (no checkpoint found)")
                wandb_run_id = None
                completed_rates = []
                all_results = {}

            # Initialize wandb for misclustering experiment (resume if checkpoint exists)
            wandb.init(
                entity=ENTITY,
                project=args.project,
                name=f"{run_name}_misclustering",
                id=wandb_run_id,
                resume="allow"
            )

            # Store run ID for future checkpoints
            if wandb_run_id is None:
                wandb_run_id = wandb.run.id

            # Set random seed for reproducibility
            rng = np.random.RandomState(args.seed)

            # Run contamination experiment for each rate
            for cont_rate in contamination_rates:
                # Skip if already completed
                if cont_rate in completed_rates:
                    print(f"Skipping contamination rate {cont_rate} (already completed)")
                    continue

                print(f"\nProcessing contamination rate: {cont_rate}")

                # Contaminate the dataset
                contaminated_dataset = []
                contaminated_examples = []
                total_contaminated_traces = 0

                for idx, reads, gt in dataset:
                    # Convert reads list to list of strings if needed
                    reads_list = list(reads)

                    # Contaminate the cluster
                    contaminated_reads, contamination_info = contaminate_trace_cluster(
                        traces=reads_list,
                        ground_truth=gt,
                        contamination_rate=cont_rate,
                        baseline_error_rate=0.055,  # Fixed value (function requires it but doesn't affect contamination)
                        cfg=cfg,
                        rng=rng
                    )

                    contaminated_dataset.append((idx, contaminated_reads, gt))

                    # Track contamination
                    num_contaminated = len(contamination_info['contaminated_positions'])
                    if num_contaminated > 0:
                        contaminated_examples.append(idx)
                        total_contaminated_traces += num_contaminated

                print(f"  Contaminated {len(contaminated_examples)} examples (out of {len(dataset)})")
                print(f"  Total contaminated traces: {total_contaminated_traces}")
                print(f"  Average contaminated traces per example: {total_contaminated_traces / len(dataset):.2f}")

                # Run inference on contaminated dataset
                # Prepare algorithm parameters (not instances) to avoid pickling issues
                if args.alg == 'trellisbma':
                    alg_params = {
                        'temp_dir': DOWNLOAD_DIR,
                        'P_INS': rates['P_INS'],
                        'P_DEL': rates['P_DEL'],
                        'P_SUB': rates['P_SUB'],
                        'k': (k or 0)
                    }
                elif args.alg == 'vs':
                    alg_params = {
                        'P_SUB': rates['P_SUB'],
                        'k': (k or 0)
                    }
                else:
                    alg_params = {'temp_dir': DOWNLOAD_DIR}

                pending = []
                for idx, reads, gt in contaminated_dataset:
                    pending.append((idx, reads, gt, args.alg, alg_params))

                # Run inference
                results = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as exe:
                    for r in tqdm.tqdm(
                            exe.map(process_example, pending),
                            total=len(pending),
                            desc=f"Contamination rate {cont_rate}",
                            file=sys.stdout
                    ):
                        results.append(r)

                # Store results
                all_results[cont_rate] = results
                completed_rates.append(cont_rate)

                # Compute overall metrics
                ld_vals = [r['levenshtein_distance'] for r in results]
                h_vals = [r['hamming_distance'] for r in results]
                mean_ld = np.mean(ld_vals)
                std_ld = np.std(ld_vals)
                mean_h = np.mean(h_vals)
                std_h = np.std(h_vals)

                # Calculate overall success/failure rate (same as inference.py)
                num_successes = sum(1 for h in h_vals if h == 0)
                success_rate = num_successes / len(h_vals) if len(h_vals) > 0 else 0.0
                failure_rate = 1 - success_rate

                # Log overall metrics to WandB
                condition_name = f"cont_{cont_rate:.3f}"
                wandb.log({
                    f"{condition_name}_mean_levenshtein": mean_ld,
                    f"{condition_name}_std_levenshtein": std_ld,
                    f"{condition_name}_mean_hamming": mean_h,
                    f"{condition_name}_std_hamming": std_h,
                    f"{condition_name}_num_examples": len(results),
                    f"{condition_name}_success_rate_all": success_rate,
                    f"{condition_name}_failure_rate_all": failure_rate
                })

                print(f"  Overall: Success: {success_rate:.3f} | Failure: {failure_rate:.3f} | H: {mean_h:.2f}±{std_h:.2f} | L: {mean_ld:.4f}±{std_ld:.4f}")

                # Compute and log per-cluster-size metrics
                results_by_N = defaultdict(list)
                for r in results:
                    results_by_N[r['num_reads']].append(r)

                for N, res in results_by_N.items():
                    h_vals_N = [r['hamming_distance'] for r in res]
                    ld_vals_N = [r['levenshtein_distance'] for r in res]
                    success_rate = sum(1 for x in h_vals_N if x == 0) / len(h_vals_N)

                    wandb.log({
                        f"{condition_name}_mean_hamming_N={N}": np.mean(h_vals_N),
                        f"{condition_name}_std_hamming_N={N}": np.std(h_vals_N),
                        f"{condition_name}_mean_levenshtein_N={N}": np.mean(ld_vals_N),
                        f"{condition_name}_std_levenshtein_N={N}": np.std(ld_vals_N),
                        f"{condition_name}_success_rate_N={N}": success_rate
                    })

                    print(f"  N={N} | Success: {success_rate:.3f} | H: {np.mean(h_vals_N):.2f}±{np.std(h_vals_N):.2f} | L: {np.mean(ld_vals_N):.4f}±{np.std(ld_vals_N):.4f}")

                # Save checkpoint after each rate
                checkpoint_data = {
                    'wandb_run_id': wandb_run_id,
                    'algorithm': args.alg,
                    'artifact': art,
                    'contamination_rates': contamination_rates,
                    'completed_rates': completed_rates,
                    'results_per_rate': all_results,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                save_checkpoint(checkpoint_path, checkpoint_data)

            # All rates completed successfully
            wandb.finish()
            print("\nMisclustering experiment completed!")

            # Delete checkpoint file when fully done
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Deleted checkpoint: {checkpoint_path}")

            return  # Exit early, skip normal inference

        # initialize this run
        wandb.init(entity=ENTITY,
                   project=args.project,
                   name=run_name)

        # prepare all the (idx, reads, gt, alg_name, alg_params) tuples
        # Prepare algorithm parameters (not instances) to avoid pickling issues
        if args.alg == 'trellisbma':
            alg_params = {
                'temp_dir': DOWNLOAD_DIR,
                'P_INS': rates['P_INS'],
                'P_DEL': rates['P_DEL'],
                'P_SUB': rates['P_SUB'],
                'k': (k or 0)
            }
        elif args.alg == 'vs':
            alg_params = {
                'P_SUB': rates['P_SUB'],
                'k': (k or 0)
            }
        else:
            alg_params = {'temp_dir': DOWNLOAD_DIR}

        pending = []
        for idx, reads, gt in dataset:
            pending.append((idx, reads, gt, args.alg, alg_params))

        # run them in parallel and collect results
        results_by_N = defaultdict(list)
        time_taken_all = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as exe:
            for r in tqdm.tqdm(
                    exe.map(process_example, pending),
                    total=len(pending),
                    desc=f"Clusters k={k}",
                    file=sys.stdout
            ):
                N = r['num_reads']
                results_by_N[N].append(r)
                time_taken_all.append(r['time_taken'])

        # Log per‐N metrics to W&B
        for N, res in results_by_N.items():
            h_vals  = [r['hamming_distance']    for r in res]
            ld_vals = [r['levenshtein_distance'] for r in res]
            wandb.log({
                f"success_rate_N={N}": sum(1 for x in h_vals if x == 0) / len(h_vals),
                f"avg_hamming_N={N}":    np.mean(h_vals),
                f"std_hamming_N={N}":    np.std(h_vals),
                f"avg_levenshtein_N={N}": np.mean(ld_vals),
                f"std_levenshtein_N={N}": np.std(ld_vals),
            })

        # Aggregate across all clusters, average over all N
        all_h = []
        all_ld = []
        for res in results_by_N.values():
            all_h.extend(r['hamming_distance']    for r in res)
            all_ld.extend(r['levenshtein_distance'] for r in res)

        total = len(all_h)
        success = sum(1 for x in all_h if x == 0)

        wandb.log({
            "success_rate_all":       success / total,
            "avg_hamming_all":        np.mean(all_h),
            "std_hamming_all":        np.std(all_h),
            "avg_levenshtein_all":    np.mean(all_ld),
            "std_levenshtein_all":    np.std(all_ld),
        })

        # Timing per example
        wandb.log({
            "avg_time_per_example": np.mean(time_taken_all),
            "std_time_per_example": np.std(time_taken_all),
        })
        wandb.finish()


if __name__ == "__main__":
    main()
