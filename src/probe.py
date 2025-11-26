import os, math, time, json, pickle, random
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from datetime import datetime


from src.utils.helper_functions import filter_string
from src.eval_pkg.GPT_Inference import GPT_Inference
from src.gpt_pkg.model import GPT, GPTConfig
from src.rnn_pkg.lstm_model import LSTMConfig, LSTMConsensus  
from Levenshtein import distance as levenshtein_distance  
from tqdm import tqdm

# Small utilities

def safe_download_artifact(entity, project, artifact_name, max_retries=3):
    """Download artifact with retries - same as in inference.py"""
    for attempt in range(1, max_retries+1):
        try:
            art = wandb.use_artifact(f'{entity}/{project}/{artifact_name}:latest', type='dataset')
            return art.download()
        except (Exception) as e:
            print(f"Attempt {attempt} failed: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to download {artifact_name}")

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def split_clusters(cluster_ids: List[int], test_frac: float, val_frac: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(seed)
    ids = np.array(cluster_ids)
    rng.shuffle(ids)
    n_test = int(round(len(ids) * test_frac))
    n_val = int(round(len(ids) * val_frac))
    test_ids = ids[:n_test].tolist()
    val_ids = ids[n_test:n_test+n_val].tolist()
    train_ids = ids[n_test+n_val:].tolist()
    return train_ids, val_ids, test_ids

def base_counts_and_freqs(reads_str: str, L: int, vocab=('A','C','G','T')) -> Tuple[np.ndarray, np.ndarray]:
    """
    reads_str: 'r1|r2|...|rN'
    returns:
      counts: [L, 4] int, with L length of ground truth
      freqs:  [L, 4] float in [0,1], row-normalized by N_eff
    """
    reads = reads_str.split('|')
    V = len(vocab)
    counts = np.zeros((L, V), dtype=np.int32)
    idx = {b:i for i,b in enumerate(vocab)}
    N_eff = np.zeros(L, dtype=np.int32)
    for r in reads:
        for j,ch in enumerate(r[:L]):
            i = idx.get(ch)
            if i is not None:
                counts[j, i] += 1
                N_eff[j] += 1
    freqs = np.zeros_like(counts, dtype=np.float32)
    mask = N_eff > 0
    freqs[mask] = counts[mask] / N_eff[mask, None]
    return counts, freqs

def majority_frequency(freqs_row: np.ndarray) -> float:
    return float(np.max(freqs_row)) if freqs_row.size else 0.0

# Linear probe head

class LinearProbe(nn.Module):
    def __init__(self, d: int, out_dim: int = 4, append_logN: bool = False):
        """ d is hidden dim; out_dim is 4 for ACGT frequencies """
        super().__init__()
        self.append_logN = append_logN
        self.lin = nn.Linear(d + (1 if append_logN else 0), out_dim)

    def forward(self, h: torch.Tensor, logN: torch.Tensor = None) -> torch.Tensor:
        # h: [B, d], logN: [B] or None
        if self.append_logN:
            assert logN is not None
            x = torch.cat([h, logN[:, None]], dim=-1)
        else:
            x = h
        return self.lin(x)

# Main runner

@hydra.main(config_path="hydra/inference_config", config_name="inference_config.yaml", version_base=None)
def main(cfg: DictConfig):
    """
    Expected extra Hydra options (add these to your YAML or CLI):
      probe:
        enabled: true
        results_dir: outputs/probe_counts
        test_frac: 0.2
        val_frac: 0.1               # validation fraction of remaining data after test split
        seed: 123
        lr: 1e-3
        weight_decay: 1e-4
        epochs: 2
        batch_size_tokens: 65536    # positions per batch (not clusters)
        append_logN: false
        layer: -1                   # last layer (reserved if you later expose multiple)
        log_interval: 10            # log training loss every N steps
        eval_interval: 100          # evaluate on validation every N steps
        use_mse_loss: false         # if true, use MSE loss; if false, use cross-entropy (default)
        warmup_frac: 0.01           # fraction of total iterations for warmup (e.g., 0.01 = 1%)
        min_lr: 1e-6                # minimum learning rate for cosine decay
      model.sampling.return_hidden_states: true
      model.sampling.return_logits: false     # not needed here
    """
    print("Starting probe script...")
    print(f"Config keys: {list(cfg.keys())}")

    if not cfg.get('probe', None):
        print("ERROR: No 'probe' section found in config")
        print("Available config:")
        print(OmegaConf.to_yaml(cfg))
        return

    if not cfg.probe.get('enabled', False):
        print("ERROR: probe.enabled is not True")
        return

    print("Probe section found and enabled")
    assert cfg.probe.enabled, "Set probe.enabled=true"
    set_seed(int(cfg.probe.seed))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Setup output directory for checkpoints (similar to pretrain.py)
    checkpoint_path = cfg.probe.get('checkpoint_path', None)
    experiment = cfg.get('experiment', None)

    # Construct output directory path
    if experiment is None:
        out_dir = os.path.join(checkpoint_path, 'probe_checkpoints',
                              f"{cfg.data.sequence_type}_{cfg.data.target_type}_gt{cfg.data.ground_truth_length}")
    else:
        out_dir = os.path.join(checkpoint_path, 'probe_checkpoints',
                              f"{cfg.data.sequence_type}_{cfg.data.target_type}_gt{cfg.data.ground_truth_length}",
                              experiment)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cfg.probe.results_dir, exist_ok=True)  # Probe results directory for manifests, summaries, etc.

    # ---- Load checkpoint & vocab (same pattern as your inference) ----
    ckpt = torch.load(cfg.model.checkpoint_path, map_location="cpu", weights_only=False)
    model_args = ckpt["model_args"]
    state_dict = ckpt["model"]
    for k in list(state_dict):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

    # meta
    script_dir   = os.path.dirname(__file__)
    data_pkg_dir = os.path.join(script_dir, 'data_pkg')  # data_pkg is in src/ directory
    meta_path    = os.path.join(data_pkg_dir, f"meta_{cfg.data.sequence_type}.pkl")
    print(f"Looking for meta file at: {meta_path}")

    if not os.path.exists(meta_path):
        print(f"Meta file not found. Checking directory contents:")
        print(f"data_pkg_dir exists: {os.path.exists(data_pkg_dir)}")
        if os.path.exists(data_pkg_dir):
            print(f"Contents: {os.listdir(data_pkg_dir)}")
        raise FileNotFoundError(f"Meta file not found at {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    meta_vocab_size = len(itos)
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # Model
    model_type = model_args.get("model_type", "gpt")
    if model_type != "gpt":
        raise ValueError("Linear probe pipeline currently assumes GPT path.")
    model_args.pop("model_type", None)
    model = GPT(GPTConfig(**model_args)).to(device).eval()
    model.load_state_dict(state_dict, strict=False)

    ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    # Load dataset artifact from wandb 
    # Initialize W&B consistently with pretrain/finetune
    wandb_log = cfg.wandb.wandb_log
    wandb_project = cfg.wandb.wandb_project

    now      = datetime.now().strftime("%Y%m%d_%H%M%S")
    base     = f"Probe_inference_{now}"
    run_name = f"{base}_{cfg.experiment}" if cfg.experiment else base

    if wandb_log:
        # Create group name similar to pretrain.py
        group = f"probe_{cfg.data.sequence_type}_{cfg.data.target_type}_gt{cfg.data.ground_truth_length}_{cfg.experiment}"

        wandb.init(
            project=wandb_project,
            entity=cfg.wandb.wandb_entity,
            group=group,
            name=run_name,
            job_type='probe_training',
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=cfg.probe.results_dir
        )

    # Download dataset artifact
    entity = cfg.wandb.wandb_entity
    project = cfg.data.data_project
    dataset_key = f"{entity}/{project}/{cfg.data.artifact_name}"
    art_name = cfg.data.artifact_name

    print(f"Downloading dataset: {dataset_key}")
    art_dir = safe_download_artifact(entity, project, art_name)

    x_test = torch.load(os.path.join(art_dir, "test_x.pt"), map_location="cpu")
    gt_file = 'ground_truth_cleaned.txt' if cfg.data.get("cleaned", False) else 'ground_truth.txt'
    with open(os.path.join(art_dir, gt_file)) as f:
        gts = [l.strip() for l in f]
    assert len(x_test) == len(gts)

    # Build string inputs and cluster ids
    examples = []
    cid_to_examples_idx = {}  # Map original dataset index to position in examples list
    skipped_examples = []
    for i, (x, gt) in enumerate(zip(x_test, gts)):
        s = decode(x.tolist())
        if ':' not in s:
            skipped_examples.append((i, s[:100]))  # Store first 100 chars for debugging
            continue
        reads_part, _ = s.split(':', 1)
        N = len(reads_part.split('|'))
        examples.append((i, s, gt, N))
        cid_to_examples_idx[i] = len(examples) - 1  # Map original index i to current position
    cluster_ids = [cid for cid, _, _, _ in examples]

    print(f"Total dataset size: {len(x_test)}")
    print(f"Valid examples (with colon): {len(examples)}")
    print(f"Skipped examples: {len(skipped_examples)}")

    if len(skipped_examples) > 0:
        print("WARNING: FOUND SKIPPED EXAMPLES - this explains the flat loss bug!")
        print("  First few skipped examples:")
        for i, (orig_idx, partial_s) in enumerate(skipped_examples[:3]):
            print(f"    [{orig_idx}]: {partial_s}...")
        print("  Index mapping (first 10 valid):")
        for cid in sorted(cid_to_examples_idx.keys())[:10]:
            examples_idx = cid_to_examples_idx[cid]
            print(f"    Original cid={cid} → examples[{examples_idx}]")
        if len(cid_to_examples_idx) > 10:
            print(f"    ... and {len(cid_to_examples_idx) - 10} more")
    else:
        print("INFO: NO SKIPPED EXAMPLES - the index mapping bug is theoretical only")
        print("   The flat loss must have a different cause")
        # Still create mapping for safety, but it will be identity mapping
        print("   Index mapping is 1:1 (cid=i → examples[i])")

    # Pass 1: run frozen model once; collect hidden states & targets 
    # We stream to disk to keep memory reasonable.
    hidden_dir = os.path.join(cfg.probe.results_dir, "hidden")
    manifest_path = os.path.join(cfg.probe.results_dir, "positions_manifest.npy")
    splits_path = os.path.join(cfg.probe.results_dir, "splits.json")

    # Initialize skip_extraction variable
    skip_extraction = False

    # Split clusters (by id) - add validation set (10% of remaining after test)
    # Only create new splits if we're extracting fresh (otherwise use cached splits)
    if not skip_extraction:
        val_frac = float(cfg.probe.get('val_frac', 0.1))
        train_ids, val_ids, test_ids = split_clusters(cluster_ids, float(cfg.probe.test_frac), val_frac, int(cfg.probe.seed))

    train_set = set(train_ids); val_set = set(val_ids); test_set = set(test_ids)

    # Check if hidden states and manifest already exist
    manifest_exists = os.path.exists(manifest_path)
    splits_exist = os.path.exists(splits_path)

    if manifest_exists and splits_exist:
        print(f"\nFound existing hidden states and manifest in {cfg.probe.results_dir}")

        # Load existing manifest and splits
        pos_table_rows = np.load(manifest_path).tolist()
        with open(splits_path, 'r') as f:
            saved_splits = json.load(f)

        # Check if all expected hidden state files exist
        expected_cids = set(cid for cid, _, _ in pos_table_rows)
        missing_files = []
        for cid in expected_cids:
            hidden_file = os.path.join(hidden_dir, f"{cid}.npy")
            if not os.path.exists(hidden_file):
                missing_files.append(cid)

        if len(missing_files) == 0:
            print(f"All {len(expected_cids)} hidden state files found. Skipping extraction.")
            print(f"Loaded {len(pos_table_rows)} position entries from manifest")

            # Use saved splits
            train_ids = saved_splits["train_ids"]
            val_ids = saved_splits["val_ids"]
            test_ids = saved_splits["test_ids"]
            print(f"Using saved train/val/test splits: {len(train_ids)}/{len(val_ids)}/{len(test_ids)} clusters")

            skip_extraction = True
        else:
            print(f"Missing {len(missing_files)} hidden state files. Will re-extract all.")
            skip_extraction = False
    else:
        print(f"No existing hidden states found in {cfg.probe.results_dir}. Will extract.")
        skip_extraction = False

    if not skip_extraction:
        os.makedirs(hidden_dir, exist_ok=True)

        # Sampler params
        sampling = OmegaConf.to_container(cfg.model.sampling, resolve=True)
        sampling.update({
            "block_size": cfg.data.block_size,
            "target_type": cfg.data.target_type,
            "ground_truth_length": cfg.data.ground_truth_length,
            "greedy": cfg.model.sampling.strategy == "greedy",
            "model_type": "gpt",
            "return_hidden_states": True,
            "return_logits": False,
        })

        batch_size = int(cfg.data.batch_size)
        num_batches = math.ceil(len(examples) / batch_size)

        pos_table_rows = []  # we'll collect a small manifest per position (cluster_id, pos, N_eff, path_to_h)

        with torch.no_grad(), ctx:
            for b in tqdm(range(num_batches), desc="Extract hidden states"):
                slice_ex = examples[b*batch_size:(b+1)*batch_size]
                if not slice_ex: break

                inputs = [s for (_, s, _, _) in slice_ex]
                alignment_sizes = [N for (_, s, _, N) in slice_ex]
                out = GPT_Inference({"model": model, "ctx": ctx, "device": device, "stoi": stoi, "itos": meta['itos'], "encode": encode, "decode": decode, **sampling}).inference(inputs, alignment_size=alignment_sizes)

                # Expect: out['dec_hidden_last']: [B, L, d]
                H = out.get("dec_hidden_last", None)
                if H is None:
                    raise RuntimeError("GPT_Inference must return 'dec_hidden_last' when return_hidden_states=True")

                # For each example in batch, compute targets and save H
                for (cid, s, gt, N), h in zip(slice_ex, H):
                    reads_with_padding, _ = s.split(':', 1)
                    # CRITICAL: Remove padding tokens from reads!
                    # Padding tokens are '#' at the beginning of the sequence
                    reads = reads_with_padding.lstrip('#')
                    L = len(gt)

                    # Debug printing for first few examples
                    if cid < 3:  # Print first 3 examples
                        print(f"\n{'='*80}")
                        print(f"DEBUG: Example {cid}")
                        print(f"  Full sequence s: {s[:100]}..." if len(s) > 100 else f"  Full sequence s: {s}")
                        print(f"  Reads with padding: {reads_with_padding[:80]}..." if len(reads_with_padding) > 80 else f"  Reads with padding: {reads_with_padding}")
                        print(f"  Padding removed: {reads[:80]}..." if len(reads) > 80 else f"  Padding removed: {reads}")
                        print(f"  Ground truth: {gt}")
                        print(f"  GT length L: {L}")
                        print(f"  Hidden state shape: {h.shape}")

                    # Cut to ground-truth length just in case generation longer
                    h = h[:L].detach().float().cpu().numpy()  # [L, d]
                    np.save(os.path.join(hidden_dir, f"{cid}.npy"), h)

                    # Targets: per-position base frequencies (now computed from unpadded reads)
                    counts, freqs = base_counts_and_freqs(reads, L)

                    # Debug: Print frequency analysis for first few examples
                    if cid < 3:
                        reads_list = reads.split('|')
                        print(f"\n  All {len(reads_list)} reads in cluster (aligned vertically):")

                        # Print all reads aligned
                        max_display_len = min(80, L)  # Display first 80 positions or full length
                        for i, read in enumerate(reads_list):
                            display_read = read[:max_display_len]
                            if len(read) > max_display_len:
                                display_read += "..."
                            print(f"    Read {i:2d}: {display_read}")

                        # Print ground truth aligned with reads
                        print(f"    GT     : {gt[:max_display_len]}{'...' if len(gt) > max_display_len else ''}")
                        print(f"    {'='*90}")

                        # Print position ruler
                        ruler_line1 = "    Pos    : " + "".join([f"{i//10}" if i % 10 == 0 else " " for i in range(max_display_len)])
                        ruler_line2 = "    Pos    : " + "".join([f"{i%10}" for i in range(max_display_len)])
                        print(ruler_line1)
                        print(ruler_line2)

                        # Print frequency analysis for all positions (or first 80)
                        print(f"\n  Frequency analysis for each position:")
                        print(f"  {'Pos':<6} {'GT':<4} {'Counts [A,C,G,T]':<20} {'Freqs [A,C,G,T]':<25} {'Majority':<8} {'Bases at position'}")
                        print(f"  {'-'*100}")

                        for pos in range(min(max_display_len, L)):
                            # Get bases at this position from all reads
                            bases_at_pos = [r[pos] if pos < len(r) else '-' for r in reads_list]
                            count_str = f"[{counts[pos,0]:2d},{counts[pos,1]:2d},{counts[pos,2]:2d},{counts[pos,3]:2d}]"
                            freq_str = f"[{freqs[pos,0]:.2f},{freqs[pos,1]:.2f},{freqs[pos,2]:.2f},{freqs[pos,3]:.2f}]"
                            majority_idx = np.argmax(freqs[pos])
                            majority_base = ['A','C','G','T'][majority_idx]

                            # Show all bases or first 10 with ellipsis
                            if len(bases_at_pos) <= 10:
                                bases_str = ''.join(bases_at_pos)
                            else:
                                bases_str = ''.join(bases_at_pos[:10]) + '...'

                            correct_mark = 'Y' if gt[pos] == majority_base else 'N'
                            print(f"  {pos:<6} {gt[pos]:<4} {count_str:<20} {freq_str:<25} {majority_base}{correct_mark:<7} {bases_str}")
                    N_eff = counts.sum(axis=1)  # [L]
                    # Save a tiny manifest row per position
                    for j in range(L):
                        pos_table_rows.append((cid, j, int(N_eff[j])))

        # Persist manifest
        np.save(manifest_path, np.array(pos_table_rows, dtype=np.int64))
        # Also keep a tiny JSON for splits
        with open(splits_path, "w") as f:
            json.dump({"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}, f)

        print(f"Saved {len(pos_table_rows)} position entries to {manifest_path}")
        print(f"Saved train/val/test splits to {splits_path}")

    # Dataset class (lazy loads per cluster .npy + recomputes targets once) 
    # To avoid recomputing counts repeatedly, we’ll cache freqs per cluster in memory dict.
    freqs_cache: Dict[int, np.ndarray] = {}

    def load_cluster_tuple(cid: int):
        # CRITICAL BUG FIX: Use mapping to get correct position in examples list!
        examples_idx = cid_to_examples_idx[cid]
        _, s, gt, _ = examples[examples_idx]  # Use mapped index, not raw cid
        reads_with_padding, _ = s.split(':', 1)
        # CRITICAL: Remove padding tokens from reads!
        reads = reads_with_padding.lstrip('#')
        L = len(gt)
        if cid not in freqs_cache:
            _, freqs = base_counts_and_freqs(reads, L)
            freqs_cache[cid] = freqs.astype(np.float32)
        H = np.load(os.path.join(hidden_dir, f"{cid}.npy"))  # [L, d] - loads by original cid (correct)
        return H, freqs_cache[cid]

    # Build index lists for train/val/test positions
    train_index = [(cid, j) for (cid, j, _) in pos_table_rows if cid in train_set]
    val_index   = [(cid, j) for (cid, j, _) in pos_table_rows if cid in val_set]
    test_index  = [(cid, j) for (cid, j, _) in pos_table_rows if cid in test_set]

    # Inspect dimensionality
    tmpH = np.load(os.path.join(hidden_dir, f"{train_index[0][0]}.npy"))
    d = int(tmpH.shape[1])

    # Probe setup
    append_logN = bool(cfg.probe.append_logN)
    use_mse_loss = bool(cfg.probe.get('use_mse_loss', False))
    probe = LinearProbe(d=d, out_dim=4, append_logN=append_logN).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=float(cfg.probe.lr), weight_decay=float(cfg.probe.weight_decay))

    # Choose loss function
    if use_mse_loss:
        loss_fn = nn.MSELoss()
        print("Using MSE loss")
    else:
        loss_fn = nn.CrossEntropyLoss()
        print("Using Cross-Entropy loss")

    def make_minibatches(index_list: List[Tuple[int,int]], batch_tokens: int):
        """Yield batches roughly by positions count (tokens) without loading everything at once."""
        # simple chunking
        chunk = []
        for pair in index_list:
            chunk.append(pair)
            if len(chunk) >= batch_tokens:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    # Calculate total iterations for LR scheduling
    total_iterations = 0
    for epoch in range(int(cfg.probe.epochs)):
        epoch_batches = len(list(make_minibatches(train_index, int(cfg.probe.batch_size_tokens))))
        total_iterations += epoch_batches

    # LR scheduler setup
    warmup_frac = float(cfg.probe.get('warmup_frac', 0.0))
    min_lr = float(cfg.probe.get('min_lr', 0.0))
    learning_rate = float(cfg.probe.lr)  # Match pretrain variable name
    warmup_iters = int(total_iterations * warmup_frac)
    lr_decay_iters = total_iterations  # Decay over full training

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    print(f"Total iterations: {total_iterations}, Warmup iterations: {warmup_iters}")

    # Debug: Print a few training examples to verify alignment
    print(f"\n{'='*80}")
    print("DEBUG: Verifying training data alignment (first 3 clusters)")
    for debug_idx, cid in enumerate(list(train_ids)[:3]):
        print(f"\n{'='*80}")
        print(f"Training cluster {cid}:")
        examples_idx = cid_to_examples_idx[cid]
        _, s, gt, _ = examples[examples_idx]  # Use mapped index, not raw cid
        reads_with_padding, _ = s.split(':', 1)
        reads = reads_with_padding.lstrip('#')

        # Load and check data
        H, F = load_cluster_tuple(cid)
        print(f"  Hidden states shape: {H.shape}")
        print(f"  Frequencies shape: {F.shape}")

        # Show all reads aligned vertically
        reads_list = reads.split('|')
        print(f"\n  All {len(reads_list)} reads in cluster (aligned vertically):")

        max_display_len = min(80, len(gt))  # Display first 80 positions or full length
        for i, read in enumerate(reads_list):
            display_read = read[:max_display_len]
            if len(read) > max_display_len:
                display_read += "..."
            print(f"    Read {i:2d}: {display_read}")

        # Print ground truth aligned with reads
        print(f"    GT     : {gt[:max_display_len]}{'...' if len(gt) > max_display_len else ''}")
        print(f"    {'='*90}")

        # Print position ruler
        ruler_line1 = "    Pos    : " + "".join([f"{i//10}" if i % 10 == 0 else " " for i in range(max_display_len)])
        ruler_line2 = "    Pos    : " + "".join([f"{i%10}" for i in range(max_display_len)])
        print(ruler_line1)
        print(ruler_line2)

        # Show first 20 positions with frequencies
        print(f"\n  Frequency analysis (first 20 positions):")
        print(f"  {'Pos':<6} {'GT':<4} {'Freqs [A,C,G,T]':<25} {'Majority':<8} {'Correct?'}")
        print(f"  {'-'*60}")

        for pos in range(min(20, len(gt))):
            freq_str = f"[{F[pos,0]:.2f},{F[pos,1]:.2f},{F[pos,2]:.2f},{F[pos,3]:.2f}]"
            majority_idx = np.argmax(F[pos])
            majority_base = ['A','C','G','T'][majority_idx]
            correct = 'Y' if gt[pos] == majority_base else 'N'
            print(f"  {pos:<6} {gt[pos]:<4} {freq_str:<25} {majority_base:<8} {correct}")
    print('='*80 + '\n')

    # Train loop with validation 
    log_interval = int(cfg.probe.get('log_interval', 10))
    eval_interval = int(cfg.probe.get('eval_interval', 100))
    best_val_loss = float('inf')
    iter_num = 0
    running_loss = []

    for epoch in range(int(cfg.probe.epochs)):
        probe.train()
        random.shuffle(train_index)

        for chunk in tqdm(make_minibatches(train_index, int(cfg.probe.batch_size_tokens)), desc=f"Train epoch {epoch+1}"):
            # Load chunk
            H_list, F_list, logN_list = [], [], []
            for (cid, j) in chunk:
                H, F = load_cluster_tuple(cid)
                H_list.append(H[j])
                F_list.append(F[j])
                if append_logN:
                    N = examples[cid][3]
                    logN_list.append(np.log(max(N, 1.0)))

            Hb = torch.tensor(np.stack(H_list), device=device, dtype=torch.float32)  # [B, d]
            Fb = torch.tensor(np.stack(F_list), device=device, dtype=torch.float32)  # [B, 4]
            if append_logN:
                logNb = torch.tensor(np.array(logN_list), device=device, dtype=torch.float32)
            else:
                logNb = None

            logits = probe(Hb, logNb)                  # [B,4]

            # Compute loss based on chosen loss function
            if use_mse_loss:
                F_pred = torch.softmax(logits, dim=-1)     # distribution over A,C,G,T
                loss = loss_fn(F_pred, Fb)  # MSE between distributions
            else:
                # For cross-entropy, use raw logits and target labels (majority class)
                targets = torch.argmax(Fb, dim=1)  # [B] - majority base indices
                loss = loss_fn(logits, targets)    # Cross-entropy with class labels
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Update learning rate
            lr = get_lr(iter_num)
            for param_group in opt.param_groups:
                param_group['lr'] = lr

            running_loss.append(loss.item())
            iter_num += 1

            # Log every log_interval steps
            if iter_num % log_interval == 0:
                avg_loss = np.mean(running_loss)
                print(f"step {iter_num}: train loss {avg_loss:.6f}, lr {lr:.2e}")
                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": avg_loss,
                        "lr": lr,
                        "epoch": epoch + 1
                    })
                running_loss = []

            # Evaluate every eval_interval steps
            if iter_num % eval_interval == 0:
                probe.eval()
                val_losses = []
                with torch.no_grad():
                    # Sample a subset of validation data for quick evaluation
                    val_sample = random.sample(val_index, min(len(val_index), 5000))
                    for val_chunk in make_minibatches(val_sample, int(cfg.probe.batch_size_tokens)):
                        H_list, F_list, logN_list = [], [], []
                        for (cid, j) in val_chunk:
                            H, F = load_cluster_tuple(cid)
                            H_list.append(H[j])
                            F_list.append(F[j])
                            if append_logN:
                                N = examples[cid][3]
                                logN_list.append(np.log(max(N, 1.0)))

                        Hb = torch.tensor(np.stack(H_list), device=device, dtype=torch.float32)
                        Fb = torch.tensor(np.stack(F_list), device=device, dtype=torch.float32)
                        if append_logN:
                            logNb = torch.tensor(np.array(logN_list), device=device, dtype=torch.float32)
                        else:
                            logNb = None

                        logits = probe(Hb, logNb)

                        # Compute validation loss
                        if use_mse_loss:
                            F_pred = torch.softmax(logits, dim=-1)
                            val_loss = loss_fn(F_pred, Fb)
                        else:
                            targets = torch.argmax(Fb, dim=1)
                            val_loss = loss_fn(logits, targets)
                        val_losses.append(val_loss.item())

                avg_val_loss = float(np.mean(val_losses))
                print(f"step {iter_num}: val loss {avg_val_loss:.6f}")

                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "val/loss": avg_val_loss,
                    })

                # Track best validation loss and save checkpoint
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print(f"New best validation loss: {best_val_loss:.6f}")

                    # Save checkpoint with best validation loss
                    checkpoint = {
                        'probe_state': probe.state_dict(),
                        'optimizer_state': opt.state_dict(),
                        'iter_num': iter_num,
                        'epoch': epoch,
                        'best_val_loss': best_val_loss,
                        'probe_config': {
                            'd': d,
                            'out_dim': 4,
                            'append_logN': append_logN
                        },
                        'config': OmegaConf.to_container(cfg, resolve=True),
                        'wandb_run_id': wandb.run.id if wandb_log and wandb.run else None,
                    }
                    checkpoint_path = os.path.join(out_dir, 'checkpoint_best.pt')
                    print(f"Saving checkpoint to {checkpoint_path}")
                    torch.save(checkpoint, checkpoint_path)

                probe.train()  # Switch back to training mode

        # End of epoch validation on full validation set
        probe.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for chunk in tqdm(make_minibatches(val_index, int(cfg.probe.batch_size_tokens)), desc=f"Full validation epoch {epoch+1}"):
                H_list, F_list, logN_list = [], [], []
                for (cid, j) in chunk:
                    H, F = load_cluster_tuple(cid)
                    H_list.append(H[j])
                    F_list.append(F[j])
                    if append_logN:
                        N = examples[cid][3]
                        logN_list.append(np.log(max(N, 1.0)))

                Hb = torch.tensor(np.stack(H_list), device=device, dtype=torch.float32)
                Fb = torch.tensor(np.stack(F_list), device=device, dtype=torch.float32)
                if append_logN:
                    logNb = torch.tensor(np.array(logN_list), device=device, dtype=torch.float32)
                else:
                    logNb = None

                logits = probe(Hb, logNb)

                # Compute epoch validation loss
                if use_mse_loss:
                    F_pred = torch.softmax(logits, dim=-1)
                    loss = loss_fn(F_pred, Fb)
                else:
                    targets = torch.argmax(Fb, dim=1)
                    loss = loss_fn(logits, targets)
                epoch_val_losses.append(loss.item())

        epoch_val_loss = np.mean(epoch_val_losses)
        print(f"\n[End of epoch {epoch+1}] Full validation loss: {epoch_val_loss:.6f}\n")

        if wandb_log:
            wandb.log({
                "epoch": epoch + 1,
                "epoch_val/loss": epoch_val_loss,
            })

    # Save final checkpoint 
    final_checkpoint = {
        'probe_state': probe.state_dict(),
        'optimizer_state': opt.state_dict(),
        'iter_num': iter_num,
        'epochs_completed': int(cfg.probe.epochs),
        'best_val_loss': best_val_loss,
        'probe_config': {
            'd': d,
            'out_dim': 4,
            'append_logN': append_logN
        },
        'config': OmegaConf.to_container(cfg, resolve=True),
        'wandb_run_id': wandb.run.id if wandb_log and wandb.run else None,
    }
    final_checkpoint_path = os.path.join(out_dir, 'checkpoint_final.pt')
    print(f"\nSaving final checkpoint to {final_checkpoint_path}")
    torch.save(final_checkpoint, final_checkpoint_path)

    # Final Test Evaluation 
    # Optionally load best checkpoint for evaluation
    use_best_checkpoint = cfg.probe.get('use_best_checkpoint_for_test', True)
    if use_best_checkpoint and os.path.exists(os.path.join(out_dir, 'checkpoint_best.pt')):
        print("\nLoading best checkpoint for test evaluation...")
        best_ckpt = torch.load(os.path.join(out_dir, 'checkpoint_best.pt'), map_location=device)
        probe.load_state_dict(best_ckpt['probe_state'])
        print(f"Loaded checkpoint with best val loss: {best_ckpt['best_val_loss']:.6f}")

    print("\nRunning final test evaluation...")
    print(f"Total training steps: {iter_num}")

    probe.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        maj_true, maj_pred = [], []
        cluster_sizes = []  # Track cluster size for each position
        bases = ['A', 'C', 'G', 'T']

        for chunk in tqdm(make_minibatches(test_index, int(cfg.probe.batch_size_tokens)), desc="Test evaluation"):
            H_list, F_list, logN_list = [], [], []
            chunk_positions = []  # Track (cid, j) for this chunk
            chunk_cluster_sizes = []  # Track cluster sizes for this chunk

            for (cid, j) in chunk:
                H, F = load_cluster_tuple(cid)
                H_list.append(H[j]); F_list.append(F[j])
                chunk_positions.append((cid, j))
                # Get cluster size from examples
                N = examples[cid][3]  # This is the cluster size
                chunk_cluster_sizes.append(N)
                if append_logN:
                    logN_list.append(np.log(max(N, 1.0)))

            Hb = torch.tensor(np.stack(H_list), device=device, dtype=torch.float32)
            Fb = torch.tensor(np.stack(F_list), device=device, dtype=torch.float32)
            if append_logN:
                logNb = torch.tensor(np.array(logN_list), device=device, dtype=torch.float32)
            else:
                logNb = None

            Fhat = torch.softmax(probe(Hb, logNb), dim=-1).cpu().numpy()
            Fb_np = Fb.cpu().numpy()

            # Print all predictions vs ground truth
            for i, (cid, j) in enumerate(chunk_positions):
                true_freqs = Fb_np[i]
                pred_freqs = Fhat[i]
                true_base = bases[np.argmax(true_freqs)]
                pred_base = bases[np.argmax(pred_freqs)]
                N = chunk_cluster_sizes[i]

                print(f"Cluster {cid:4d}, Pos {j:3d}, N={N:2d}: "
                      f"GT[A:{true_freqs[0]:.2f} C:{true_freqs[1]:.2f} G:{true_freqs[2]:.2f} T:{true_freqs[3]:.2f}]={true_base} | "
                      f"Pred[A:{pred_freqs[0]:.2f} C:{pred_freqs[1]:.2f} G:{pred_freqs[2]:.2f} T:{pred_freqs[3]:.2f}]={pred_base}")

            y_pred.append(Fhat); y_true.append(Fb_np)
            maj_true.extend(np.argmax(Fb_np, axis=1).tolist())
            maj_pred.extend(np.argmax(Fhat, axis=1).tolist())
            cluster_sizes.extend(chunk_cluster_sizes)

        Yt = np.concatenate(y_true, axis=0)   # [T,4]
        Yp = np.concatenate(y_pred, axis=0)   # [T,4]
        cluster_sizes = np.array(cluster_sizes)  # [T]

        # Calculate overall metrics
        mse_overall = np.mean((Yt - Yp) ** 2)
        mae_overall = np.mean(np.abs(Yt - Yp))

        # Calculate per-base metrics
        mse_per_base = {}
        mae_per_base = {}
        for b, base in enumerate(bases):
            mse_per_base[base] = float(np.mean((Yt[:, b] - Yp[:, b]) ** 2))
            mae_per_base[base] = float(np.mean(np.abs(Yt[:, b] - Yp[:, b])))

        # Calculate KL Divergence: D_KL(P||Q) where P=true, Q=predicted
        def safe_row_kl(p, q, eps=1e-12):
            # p: true freqs (may be all zeros if no coverage)
            ps = p.sum()
            if ps <= 0:          # skip zero-coverage positions
                return np.nan
            p = (p / ps).astype(np.float64)

            # clip and renormalize q
            q = np.clip(q.astype(np.float64), eps, None)
            q /= q.sum()

            return float(np.sum(p * (np.log(p + eps) - np.log(q))))   # nats (ln)

        valid_mask = Yt.sum(axis=1) > 0
        kl_divs = np.array([safe_row_kl(p, q) for p, q in zip(Yt[valid_mask], Yp[valid_mask])])
        mean_kl_divergence = float(np.nanmean(kl_divs))

        # Majority accuracy
        maj_acc = float((np.array(maj_true) == np.array(maj_pred)).mean())

        # Calculate metrics by cluster size
        unique_cluster_sizes = sorted(set(cluster_sizes))
        metrics_by_cluster_size = {}

        for N in unique_cluster_sizes:
            mask = cluster_sizes == N
            if np.sum(mask) == 0:
                continue

            Yt_N = Yt[mask]
            Yp_N = Yp[mask]
            maj_true_N = np.array(maj_true)[mask]
            maj_pred_N = np.array(maj_pred)[mask]

            # MSE for this cluster size
            mse_N_overall = float(np.mean((Yt_N - Yp_N) ** 2))
            mse_N_per_base = {}
            for b, base in enumerate(bases):
                mse_N_per_base[base] = float(np.mean((Yt_N[:, b] - Yp_N[:, b]) ** 2))

            # MAE for this cluster size
            mae_N_overall = float(np.mean(np.abs(Yt_N - Yp_N)))
            mae_N_per_base = {}
            for b, base in enumerate(bases):
                mae_N_per_base[base] = float(np.mean(np.abs(Yt_N[:, b] - Yp_N[:, b])))

            # KL Divergence for this cluster size
            valid_mask_N = Yt_N.sum(axis=1) > 0
            if np.sum(valid_mask_N) > 0:
                kl_divs_N = np.array([safe_row_kl(p, q) for p, q in zip(Yt_N[valid_mask_N], Yp_N[valid_mask_N])])
                mean_kl_N = float(np.nanmean(kl_divs_N))
            else:
                mean_kl_N = float('nan')

            # Majority accuracy for this cluster size
            maj_acc_N = float((maj_true_N == maj_pred_N).mean())

            metrics_by_cluster_size[f"N{N}"] = {
                "mse_overall": float(mse_N_overall),
                "mse_per_base": mse_N_per_base,
                "mae_overall": float(mae_N_overall),
                "mae_per_base": mae_N_per_base,
                "mean_kl_divergence": float(mean_kl_N),
                "majority_accuracy": maj_acc_N,
                "num_positions": int(np.sum(mask))
            }

        summary = {
            "mse_overall": float(mse_overall),
            "mse_per_base": mse_per_base,
            "mae_overall": float(mae_overall),
            "mae_per_base": mae_per_base,
            "mean_kl_divergence": float(mean_kl_divergence),
            "majority_accuracy": maj_acc,
            "metrics_by_cluster_size": metrics_by_cluster_size,
            "num_test_positions": int(Yt.shape[0]),
            "num_train_clusters": len(train_ids),
            "num_val_clusters": len(val_ids),
            "num_test_clusters": len(test_ids),
            "best_val_loss": float(best_val_loss),
            "total_training_steps": iter_num,
            "epochs_completed": int(cfg.probe.epochs),
        }
        with open(os.path.join(cfg.probe.results_dir, "probe_counts_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("== Linear probe summary ==")
        print(json.dumps(summary, indent=2))

        # Log final test metrics to W&B
        if wandb_log:
            wandb_metrics = {
                "test/mse_overall": mse_overall,
                "test/mse_A": mse_per_base['A'],
                "test/mse_C": mse_per_base['C'],
                "test/mse_G": mse_per_base['G'],
                "test/mse_T": mse_per_base['T'],
                "test/mae_overall": mae_overall,
                "test/mae_A": mae_per_base['A'],
                "test/mae_C": mae_per_base['C'],
                "test/mae_G": mae_per_base['G'],
                "test/mae_T": mae_per_base['T'],
                "test/mean_kl_divergence": mean_kl_divergence,
                "test/majority_accuracy": maj_acc,
                "test/num_positions": int(Yt.shape[0]),
            }

            # Add cluster size specific metrics
            for cluster_key, metrics in metrics_by_cluster_size.items():
                wandb_metrics[f"test/mse_overall_{cluster_key}"] = metrics["mse_overall"]
                wandb_metrics[f"test/mae_overall_{cluster_key}"] = metrics["mae_overall"]
                wandb_metrics[f"test/mean_kl_divergence_{cluster_key}"] = metrics["mean_kl_divergence"]
                wandb_metrics[f"test/majority_accuracy_{cluster_key}"] = metrics["majority_accuracy"]
                wandb_metrics[f"test/num_positions_{cluster_key}"] = metrics["num_positions"]

                # Add per-base metrics for each cluster size
                for base in bases:
                    wandb_metrics[f"test/mse_{base}_{cluster_key}"] = metrics["mse_per_base"][base]
                    wandb_metrics[f"test/mae_{base}_{cluster_key}"] = metrics["mae_per_base"][base]

            wandb.log(wandb_metrics)
            wandb.finish()

if __name__ == "__main__":
    main()