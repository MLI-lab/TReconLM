"""
Exact Bayesian posterior P(x_t = a | y_1, ..., y_M) for a memoryless
IDS channel via forward-backward (BCJR) algorithm.

No inner code, assume uniform prior 1/q over the alphabet at each position.

The HMM state is the drift vector (d_1, ..., d_M), one drift per
received sequence. d_j = #insertions - #deletions that occurred in
received sequence y_j up to the current ground truth position.
Boundary conditions: all drifts start at 0 and end at m_j - n
(length of received sequence j minus transmitted sequence length).

Two decoding modes (cf. Maarouf et al., 2022, arXiv:2111.14452):
  joint:    exact a posteriori probabilities from the multi-dimensional drift trellis.
  separate: approximate a posteriori probabilities by combining per-read posteriors,
            P(x_t=a | y_1,...,y_M) ~ prod_j P(x_t=a | y_j) / P(a)^(M-1).

Usage:
  compare joint vs separate decoding:
    python analysis/ids_posterior.py --mode decoding --n_trials 20 --n 12 --N 2

  generate predictions tsv from exact posterior on random sequences:
    python analysis/ids_posterior.py --mode generate --n_trials 500 --n 15 --output posterior_preds.tsv --workers 50
    python analysis/ids_posterior.py --mode generate --n_trials 500 --n 15 --N 3 --output posterior_preds.tsv --workers 50

  run joint posterior on test set traces (filters for cluster_size=2):
    python analysis/ids_posterior.py --mode testset --input /path/to/predictions_output.tsv --output posterior_preds.tsv --N 2 --workers 50
    python analysis/ids_posterior.py --mode testset --input /path/to/predictions_output.tsv --output posterior_preds.tsv --N 2 --workers 50 --max_seqs 1000
  saves both a .tsv (for pattern_analysis.py) and a .npy with full posteriors (n_seqs, n, q).

  then run pattern_analysis.py on the output file.
  if --N is not set in generate mode, cluster size is randomly sampled from 2 to 5.

  python analysis/ids_posterior.py --help
"""

import numpy as np
import Levenshtein
from itertools import product as iterproduct


# ------------------------------------------------------------------
# Channel probability precomputation
# ------------------------------------------------------------------

def _precompute_channel_probs(received, p_i, p_d, p_s, q, max_ins):
    """
    Precompute single-symbol channel log-probabilities for one received sequence.

    For each possible starting position j in the received sequence, each
    possible number of output symbols r, and each possible input symbol a,
    compute log P(y[j:j+r] | single ground truth symbol = a).

    Returns log_ch, a 3d array of shape (m+1, K+1, q), initialized to -inf.
        axis 0 (j): cursor position in received seq (0 = start, m = end).
        axis 1 (r): how many received symbols this input symbol produced.
                    r=0 means deletion, r=1 means transmission (no insertions),
                    r>=2 means transmission plus insertions.
        axis 2 (a): the ground truth symbol (0=A, 1=C, 2=G, 3=T).

    The dp uses the lattice from Davey and MacKay (2001) and Maarouf et al. (2022),
    Section III, Fig 3, but simplified to input length n=1 (no inner code).

    Ft[p, v] is the lattice value, the probability of having consumed p input
    symbols (0 or 1 for us) and produced v output symbols so far.
    The three transitions in the lattice are:
      vertical   (deletion):  consume input, produce nothing.
      diagonal   (transmit):  consume input, produce one output (match or mismatch).
      horizontal (insertion): produce one random output, do not consume input.
    We save the value before adding insertions (F_out) because insertions
    after the last output symbol belong to the next ground truth symbol.
    """
    m = len(received)
    p_t = 1 - p_i - p_d  # transmission probability
    K = 1 + max_ins       # max received symbols per input symbol

    # initialize to -inf = log(0), meaning impossible by default
    log_ch = np.full((m + 1, K + 1, q), -np.inf)

    for j in range(m + 1):          # for each starting position in received seq
        max_r = min(K, m - j)       # maximum number of received symbols that one ground truth symbol can produce, given the current cursor position j in recived sequence
        for a in range(q):          # for each possible input symbol
            # lattice dp matrix, shape (2, max_r + 1).
            # rows (p = 0 or 1): how many input symbols have been consumed.
            #   only 2 rows because we process a single ground truth symbol.
            # columns (v = 0 to max_r): how many received symbols produced so far.
            # e.g. Ft[0, 3] = prob of 0 input consumed, 3 outputs (3 insertions).
            #      Ft[1, 2] = prob of 1 input consumed, 2 outputs (transmit + 1 ins, etc).
            Ft = np.zeros((2, max_r + 1))
            Ft[0, 0] = 1.0  # base case, 0 consumed, 0 produced, probability 1

            for v in range(max_r + 1):       # v = number of output symbols produced
                for p in range(2):           # p = input symbols consumed (0 or 1)
                    # vertical transition, deletion. consume input, no output
                    if p > 0:
                        Ft[p, v] += p_d * Ft[p - 1, v]

                    # diagonal transition, transmission. consume input, produce output
                    if p > 0 and v > 0:
                        y_v = received[j + v - 1]  # the actual received symbol
                        if a == y_v:
                            # correct transmission (no substitution)
                            Ft[p, v] += Ft[p - 1, v - 1] * p_t * (1 - p_s)
                        else:
                            # substitution, transmitted but wrong symbol
                            Ft[p, v] += Ft[p - 1, v - 1] * p_t * p_s / (q - 1)

                    # save before insertions. this is the output probability,
                    # the probability that exactly v symbols were produced and we are done
                    if p == 1 and Ft[1, v] > 0:
                        log_ch[j, v, a] = np.log(Ft[1, v])

                    # horizontal transition, insertion. random output, do not consume input
                    if v > 0:
                        Ft[p, v] += p_i * Ft[p, v - 1] / q

    return log_ch


# ------------------------------------------------------------------
# Joint decoding (exact, exponential in M)
# ------------------------------------------------------------------

def _joint_forward_backward(log_chs, n, ms, q, max_ins):
    """
    Exact joint posterior via BCJR on the M-dimensional drift trellis.
    (Maarouf et al. 2022, Section IV-A)

    The HMM state is a tuple (j_1, ..., j_M) of cursor positions, one per
    received sequence. This is equivalent to the joint drift vector
    (d_1, ..., d_M) since d_k = j_k - t at ground truth position t.

    States are stored sparsely as python dicts because most of the
    full (m_1+1) x ... x (m_M+1) state space is unreachable.

    Args:
        log_chs: list of M precomputed channel prob arrays from _precompute_channel_probs.
        n:       ground truth sequence length.
        ms:      list of M received sequence lengths [m_1, ..., m_M].
        q:       alphabet size.
        max_ins: max insertions per input symbol.

    Returns:
        posterior: (n, q) array of exact joint posteriors.
    """
    M = len(log_chs)
    K = 1 + max_ins
    log_prior = np.log(1.0 / q)

    def read_transitions(k, jk, a):
        """For received seq k at cursor jk, yield all valid (r, log_prob) pairs
        for input symbol a. r is how many received symbols this symbol produces."""
        max_r = min(K, ms[k] - jk)
        for r in range(max_r + 1):
            lc = log_chs[k][jk, r, a]
            if lc > -np.inf:
                yield r, lc

    # forward pass.
    # alpha[t][state] = log P(y_1[0:j_1], ..., y_M[0:j_M], processed x_0..x_{t-1})
    # where state = (j_1, ..., j_M), marginalized over all possible x_0..x_{t-1}.
    alpha = [dict() for _ in range(n + 1)]
    alpha[0][tuple(0 for _ in range(M))] = 0.0  # all cursors start at 0

    for t in range(n):
        for state, la in alpha[t].items():
            # try each possible ground truth symbol a at position t
            for a in range(q):
                # for each received seq, get the valid transitions (r, log_prob)
                per_read = []
                skip = False
                for k in range(M):
                    tr = list(read_transitions(k, state[k], a))
                    if not tr:
                        skip = True  # no valid transition for this read, skip symbol
                        break
                    per_read.append(tr)
                if skip:
                    continue

                # enumerate all combinations of per-read transitions.
                # each combo is ((r_1, lc_1), (r_2, lc_2), ..., (r_M, lc_M)).
                # the branch metric factorizes as prod_k P(segment_k | a).
                for combo in iterproduct(*per_read):
                    # new cursor positions after this transition
                    new_state = tuple(state[k] + combo[k][0] for k in range(M))
                    # log branch metric = sum of per-read log probs + log prior
                    val = la + sum(c[1] for c in combo) + log_prior

                    if new_state in alpha[t + 1]:
                        alpha[t + 1][new_state] = np.logaddexp(
                            alpha[t + 1][new_state], val)
                    else:
                        alpha[t + 1][new_state] = val

    # backward pass.
    # beta[t][state] = log P(remaining received symbols | state at time t).
    # final state: all cursors must be at the end of their received sequences.
    final_state = tuple(ms)
    beta = [dict() for _ in range(n + 1)]
    if final_state not in alpha[n]:
        raise ValueError(
            f"Final state {final_state} not reachable (try increasing max_ins).")
    beta[n][final_state] = 0.0  # boundary, drift = (m_1-n, ..., m_M-n)

    for t in range(n - 1, -1, -1):
        # only iterate over states that are reachable (exist in alpha)
        for state in alpha[t]:
            for a in range(q):
                per_read = []
                skip = False
                for k in range(M):
                    tr = list(read_transitions(k, state[k], a))
                    if not tr:
                        skip = True
                        break
                    per_read.append(tr)
                if skip:
                    continue

                for combo in iterproduct(*per_read):
                    new_state = tuple(state[k] + combo[k][0] for k in range(M))
                    # only contribute if the next state has a valid backward prob
                    if new_state not in beta[t + 1]:
                        continue
                    val = beta[t + 1][new_state] + sum(c[1] for c in combo) + log_prior
                    if state in beta[t]:
                        beta[t][state] = np.logaddexp(beta[t][state], val)
                    else:
                        beta[t][state] = val

    # posterior.
    # P(x_t = a | y_1,...,y_M) is proportional to the sum over all states and transitions
    # of alpha[t][state] * P(segments | a) * P(a) * beta[t+1][new_state].
    log_post = np.full((n, q), -np.inf)
    for t in range(n):
        for state, la in alpha[t].items():
            for a in range(q):
                per_read = []
                skip = False
                for k in range(M):
                    tr = list(read_transitions(k, state[k], a))
                    if not tr:
                        skip = True
                        break
                    per_read.append(tr)
                if skip:
                    continue

                for combo in iterproduct(*per_read):
                    new_state = tuple(state[k] + combo[k][0] for k in range(M))
                    if new_state not in beta[t + 1]:
                        continue
                    val = la + sum(c[1] for c in combo) + log_prior + beta[t + 1][new_state]
                    log_post[t, a] = np.logaddexp(log_post[t, a], val)

    # normalize to get proper distribution at each position
    log_Z = np.logaddexp.reduce(log_post, axis=1, keepdims=True)
    return np.exp(log_post - log_Z)


# ------------------------------------------------------------------
# Separate decoding (approximate, linear in M)
# ------------------------------------------------------------------

def _single_forward_backward(log_ch, n, m, q, max_ins):
    """
    BCJR forward-backward for a single received sequence.
    This is the M=1 special case where the state is just the scalar
    cursor position j (equivalently, drift d = j - t).

    Args:
        log_ch: precomputed channel probs, shape (m+1, K+1, q).
        n:      ground truth length.
        m:      received sequence length.
        q:      alphabet size.
        max_ins: max insertions per input symbol.

    Returns:
        posterior: (n, q) array, posterior[t, a] = P(x_t = a | y).
    """
    K = 1 + max_ins
    log_prior = np.log(1.0 / q)

    # alpha[t, j] = log P(y[0:j], processed ground truth symbols 0..t-1),
    # marginalized over all possible symbol values x_0..x_{t-1}.
    alpha = np.full((n + 1, m + 1), -np.inf)
    alpha[0, 0] = 0.0  # start, cursor at 0, drift = 0

    for t in range(n):                          # for each ground truth position
        for j in range(m + 1):                  # for each cursor position
            if alpha[t, j] == -np.inf:
                continue                        # unreachable state, skip
            for r in range(min(K, m - j) + 1):  # r = output symbols produced
                for a in range(q):              # a = possible ground truth symbol
                    lc = log_ch[j, r, a]
                    if lc > -np.inf:
                        # transition, cursor moves from j to j+r
                        alpha[t + 1, j + r] = np.logaddexp(
                            alpha[t + 1, j + r],
                            alpha[t, j] + lc + log_prior)

    # beta[t, j] = log P(y[j:m] | ground truth symbols t..n-1 still to process)
    beta = np.full((n + 1, m + 1), -np.inf)
    beta[n, m] = 0.0  # end, all input processed, cursor at m, drift = m-n

    for t in range(n - 1, -1, -1):
        for j in range(m + 1):
            for r in range(min(K, m - j) + 1):
                if beta[t + 1, j + r] == -np.inf:
                    continue
                for a in range(q):
                    lc = log_ch[j, r, a]
                    if lc > -np.inf:
                        beta[t, j] = np.logaddexp(
                            beta[t, j],
                            beta[t + 1, j + r] + lc + log_prior)

    # posterior, combine alpha and beta.
    # P(x_t = a | y) is proportional to
    # sum_j sum_r alpha[t,j] * P(y[j:j+r]|a) * P(a) * beta[t+1,j+r].
    # this marginalizes over the cursor position j (where in the received
    # sequence this ground truth symbol maps to) and r (how many received
    # symbols it produced, accounting for insertions and deletions).
    log_post = np.full((n, q), -np.inf)
    for t in range(n):
        for j in range(m + 1):
            if alpha[t, j] == -np.inf:
                continue
            for r in range(min(K, m - j) + 1):
                if beta[t + 1, j + r] == -np.inf:
                    continue
                for a in range(q):
                    lc = log_ch[j, r, a]
                    if lc > -np.inf:
                        log_post[t, a] = np.logaddexp(
                            log_post[t, a],
                            alpha[t, j] + lc + log_prior + beta[t + 1, j + r])

    # normalize so each row sums to 1
    log_Z = np.logaddexp.reduce(log_post, axis=1, keepdims=True)
    return np.exp(log_post - log_Z)


def _separate_decoding(log_chs, n, ms, q, max_ins):
    """
    Approximate a posteriori probabilities by decoding each received sequence
    independently and combining the results. (Maarouf et al. 2022, Section IV-B)

    P(x_t=a | y_1,...,y_M) is proportional to
    prod_j P(x_t=a | y_j) / P(a)^(M-1).
    With uniform prior P(a) = 1/q, this simplifies to prod_j P(x_t=a | y_j),
    then renormalize.

    This ignores coupling between read alignments through the shared ground
    truth, but is much faster (linear in M vs exponential for joint).
    """
    log_post_combined = np.zeros((n, q))
    for k, log_ch in enumerate(log_chs):
        post_k = _single_forward_backward(log_ch, n, ms[k], q, max_ins)
        log_post_combined += np.log(np.clip(post_k, 1e-300, None))
    log_Z = np.logaddexp.reduce(log_post_combined, axis=1, keepdims=True)
    return np.exp(log_post_combined - log_Z)


# ------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------

def ids_posterior(reads, n, p_i=0.05, p_d=0.05, p_s=0.05, q=4,
                 max_ins=3, joint=True):
    """
    Compute posterior P(x_t = a | y_1, ..., y_M) for the IDS channel.

    Args:
        reads:   single read (1d int array) or list of reads (each 1d int array).
        n:       transmitted sequence length.
        p_i:     insertion probability.
        p_d:     deletion probability.
        p_s:     substitution probability (conditional on transmission).
        q:       alphabet size (4 for DNA).
        max_ins: max insertions per input symbol.
        joint:   true = exact joint decoding (exponential in M),
                 false = separate decoding then combine (linear in M).

    Returns:
        posterior: (n, q) array, posterior[t, a] = P(x_t = a | y_1,...,y_M).
    """
    # accept single read or list of reads
    if isinstance(reads, np.ndarray) and reads.ndim == 1:
        reads = [reads]

    M = len(reads)
    ms = [len(r) for r in reads]

    # precompute channel probs for each read
    log_chs = [_precompute_channel_probs(r, p_i, p_d, p_s, q, max_ins)
               for r in reads]

    if M == 1 or not joint:
        return _separate_decoding(log_chs, n, ms, q, max_ins)
    else:
        return _joint_forward_backward(log_chs, n, ms, q, max_ins)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
INT_TO_BASE = ['A', 'C', 'G', 'T']


def str_to_ints(seq):
    return np.array([BASE_TO_INT[b] for b in seq])


def ints_to_str(arr):
    return ''.join(INT_TO_BASE[i] for i in arr)


# ------------------------------------------------------------------
# Generate posterior predictions from existing model test set
# ------------------------------------------------------------------

def _process_from_testset(args_tuple):
    """Worker for processing one row from the model's test set."""
    (gt_str, traces_str, p_i, p_d, p_s, q, joint) = args_tuple

    reads_str = traces_str.split('|||')
    reads_int = [str_to_ints(r.replace('|', '')) for r in reads_str]
    n = len(gt_str)
    x_int = str_to_ints(gt_str)

    post = ids_posterior(reads_int, n, p_i, p_d, p_s, q, joint=joint)
    pred_int = post.argmax(axis=1)
    pred = ints_to_str(pred_int)

    n_correct = int((pred_int == x_int).sum())
    ham_dist = n - n_correct
    lev_dist = Levenshtein.distance(gt_str, pred)
    is_exact = int(gt_str == pred)

    return gt_str, pred, n_correct, ham_dist, lev_dist, is_exact, post


def generate_from_testset(input_path, output_path,
                          p_i=0.05, p_d=0.05, p_s=0.05, q=4,
                          joint=True, cluster_size=None,
                          n_workers=1, max_seqs=None):
    """
    Read a test set file (with ground_truth and noisy_traces columns),
    run the exact Bayesian posterior on the reads, and write both
    a predictions tsv and a numpy file with the full posteriors.

    Args:
        input_path:  path to TSV with ground_truth and noisy_traces columns.
        output_path: where to write the posterior predictions tsv.
                     posteriors are saved to the same path with .npy extension.
        p_i, p_d, p_s: IDS channel parameters (must match training config).
        q:           alphabet size.
        joint:       use joint (true) or separate (false) decoding.
        cluster_size: if set, only process clusters of this size
                      (e.g. 2 for joint decoding feasibility).
        n_workers:   number of parallel workers.
        max_seqs:    if set, only process this many sequences.
    """
    import csv
    from multiprocessing import Pool

    # read input, optionally filter by cluster size
    rows = []
    n_skipped = 0
    with open(input_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            cs = int(row['cluster_size'])
            if cluster_size is not None and cs != cluster_size:
                n_skipped += 1
                continue
            rows.append((row['ground_truth'], row['noisy_traces']))
            if max_seqs and len(rows) >= max_seqs:
                break

    n_seqs = len(rows)
    n = len(rows[0][0])
    print(f"  loaded {n_seqs} sequences from {input_path}, n={n}")
    if n_skipped > 0:
        print(f"  skipped {n_skipped} sequences (cluster_size != {cluster_size})")
    print(f"  joint={joint}, workers={n_workers}")

    args_list = [
        (gt, traces, p_i, p_d, p_s, q, joint)
        for gt, traces in rows
    ]

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    results = []
    if n_workers <= 1:
        iterable = args_list
        if has_tqdm:
            iterable = tqdm(iterable, desc="sequences", total=n_seqs)
        for i, args_tuple in enumerate(iterable):
            if not has_tqdm and ((i + 1) % 100 == 0 or i == 0):
                print(f"  sequence {i + 1}/{n_seqs}...", flush=True)
            results.append(_process_from_testset(args_tuple))
    else:
        with Pool(n_workers) as pool:
            imap_iter = pool.imap(_process_from_testset, args_list, chunksize=5)
            if has_tqdm:
                imap_iter = tqdm(imap_iter, desc="sequences", total=n_seqs)
            for i, result in enumerate(imap_iter):
                if not has_tqdm and ((i + 1) % 100 == 0 or i == 0):
                    print(f"  sequence {i + 1}/{n_seqs}...", flush=True)
                results.append(result)

    n_correct_total = 0
    n_total = 0
    all_posteriors = []
    all_ham = []
    all_lev = []
    n_exact = 0

    with open(output_path, 'w') as f:
        f.write("ground_truth\tprediction\t"
                "ham_distance\tlev_distance\n")
        for gt, posterior_pred, n_correct, ham_dist, lev_dist, is_exact, post in results:
            f.write(f"{gt}\t{posterior_pred}\t"
                    f"{ham_dist}\t{lev_dist}\n")

            n_correct_total += n_correct
            n_total += n
            all_posteriors.append(post)
            all_ham.append(ham_dist)
            all_lev.append(lev_dist)
            n_exact += is_exact

    # save posteriors as numpy array, shape (n_seqs, n, q)
    posteriors_array = np.stack(all_posteriors)
    npy_path = output_path.replace('.tsv', '.npy')
    np.save(npy_path, posteriors_array)

    acc = n_correct_total / n_total
    failure_rate = 1 - n_exact / n_seqs

    print(f"\n  wrote {n_seqs} predictions to {output_path}")
    print(f"  wrote posteriors {posteriors_array.shape} to {npy_path}")
    print(f"\n  {'metric':<25} {'posterior':>10}")
    print(f"  {'per-base accuracy':<25} {acc:>10.1%}")
    print(f"  {'failure rate':<25} {failure_rate:>10.1%}")
    print(f"  {'mean hamming dist':<25} {np.mean(all_ham):>10.2f}")
    print(f"  {'mean levenshtein dist':<25} {np.mean(all_lev):>10.2f}")
    print(f"  {'median hamming dist':<25} {np.median(all_ham):>10.1f}")
    print(f"  {'median levenshtein dist':<25} {np.median(all_lev):>10.1f}")


# ------------------------------------------------------------------
# Generate posterior predictions from random sequences
# ------------------------------------------------------------------

def _process_one_sequence(args_tuple):
    """Worker function for parallel generation. Takes a tuple of all needed args."""
    (seq_i, n, N, p_i, p_d, p_s, q, joint, seed) = args_tuple

    import sys
    sys.path.insert(0, '/workspaces/TReconLM')
    from src.data_pkg.IDS_channel import IDS_channel
    from src.data_pkg.data_generation import generate_ground_truth_sequence

    rng = np.random.RandomState(seed + seq_i)

    channel_stats = {
        'insertion_probability': p_i,
        'deletion_probability': p_d,
        'substitution_probability': p_s,
    }

    gt = generate_ground_truth_sequence(n, rng)
    x_int = str_to_ints(gt)

    # sample cluster size uniformly from 2 to N if N > 0, otherwise use fixed N
    if N is None:
        A = rng.randint(2, 6)  # 2 to 5
    else:
        A = N

    reads_str = [IDS_channel(gt, channel_stats, rng) for _ in range(A)]
    reads_int = [str_to_ints(r) for r in reads_str]

    post = ids_posterior(reads_int, n, p_i, p_d, p_s, q, joint=joint)
    pred_int = post.argmax(axis=1)
    pred = ints_to_str(pred_int)

    n_correct = int((pred_int == x_int).sum())
    return gt, pred, n_correct, A


def generate_posterior_predictions(output_path, n_seqs=500, n=15, N=None,
                                   p_i=0.05, p_d=0.05, p_s=0.05, q=4,
                                   joint=False, seed=42, n_workers=1):
    """
    Generate a predictions file from the exact Bayesian posterior.
    Output format matches what pattern_analysis.py expects:
    tab-separated with columns "ground_truth" and "prediction".

    For each random ground truth sequence, generate N noisy reads via the
    IDS channel, compute the posterior, and take the MAP (argmax) prediction
    at each position.

    Args:
        output_path: where to write the tsv file.
        n_seqs:      number of random sequences to generate.
        n:           ground truth sequence length.
        N:           cluster size (number of reads). if not set, randomly
                     sample from 2 to 5 for each sequence.
        p_i, p_d, p_s: IDS channel parameters.
        q:           alphabet size.
        joint:       use joint (true) or separate (false) decoding.
        seed:        random seed.
        n_workers:   number of parallel workers.
    """
    from multiprocessing import Pool

    args_list = [
        (seq_i, n, N, p_i, p_d, p_s, q, joint, seed)
        for seq_i in range(n_seqs)
    ]

    print(f"  generating {n_seqs} sequences, n={n}, "
          f"N={'random(2-5)' if N is None else N}, "
          f"workers={n_workers}, joint={joint}")

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    results = []
    if n_workers <= 1:
        iterable = enumerate(args_list)
        if has_tqdm:
            iterable = tqdm(list(iterable), desc="sequences", total=n_seqs)
            for i, args_tuple in iterable:
                results.append(_process_one_sequence(args_tuple))
        else:
            for i, args_tuple in iterable:
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"  sequence {i + 1}/{n_seqs}...")
                results.append(_process_one_sequence(args_tuple))
    else:
        with Pool(n_workers) as pool:
            imap_iter = pool.imap(_process_one_sequence, args_list, chunksize=10)
            if has_tqdm:
                imap_iter = tqdm(imap_iter, desc="sequences", total=n_seqs)
            for i, result in enumerate(imap_iter):
                if not has_tqdm and ((i + 1) % 50 == 0 or i == 0):
                    print(f"  sequence {i + 1}/{n_seqs}...")
                results.append(result)

    n_correct_total = 0
    n_total = 0

    with open(output_path, 'w') as f:
        f.write("ground_truth\tprediction\n")
        for gt, pred, n_correct, A in results:
            f.write(f"{gt}\t{pred}\n")
            n_correct_total += n_correct
            n_total += n

    acc = n_correct_total / n_total
    print(f"\n  wrote {n_seqs} sequences to {output_path}")
    print(f"  overall accuracy: {acc:.1%}")
    print(f"\n  now run: python analysis/pattern_analysis.py on this file")


# ------------------------------------------------------------------
# Test: joint vs separate decoding (Maarouf et al. 2022)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import argparse
    sys.path.insert(0, '/workspaces/TReconLM')
    from src.data_pkg.IDS_channel import IDS_channel
    from src.data_pkg.data_generation import generate_ground_truth_sequence

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["decoding", "generate", "testset"], default="decoding",
                        help="decoding: compare joint vs separate. generate: write predictions tsv. testset: run posterior on model test set.")
    parser.add_argument("--n_trials", type=int, default=10, help="number of test sequences")
    parser.add_argument("--n", type=int, default=12, help="ground truth sequence length")
    parser.add_argument("--N", type=int, default=None,
                        help="cluster size (number of reads). if not set, randomly sample from 2 to 5.")
    parser.add_argument("--input", type=str, default=None,
                        help="input predictions file for testset mode")
    parser.add_argument("--output", type=str, default="posterior_predictions.tsv",
                        help="output file path for generate/testset mode")
    parser.add_argument("--max_seqs", type=int, default=None,
                        help="max sequences to process (testset mode)")
    parser.add_argument("--workers", type=int, default=1, help="number of parallel workers")
    parser.add_argument("--p_i", type=float, default=0.05, help="insertion probability")
    parser.add_argument("--p_d", type=float, default=0.05, help="deletion probability")
    parser.add_argument("--p_s", type=float, default=0.05, help="substitution probability")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--separate", action="store_true",
                        help="use separate (approximate) decoding instead of joint")
    args = parser.parse_args()

    if args.mode == "testset":
        if not args.input:
            print("error: --input required for testset mode")
            sys.exit(1)
        generate_from_testset(
            input_path=args.input,
            output_path=args.output,
            p_i=args.p_i, p_d=args.p_d, p_s=args.p_s,
            joint=not args.separate,
            cluster_size=args.N,
            n_workers=args.workers,
            max_seqs=args.max_seqs)

    elif args.mode == "generate":
        generate_posterior_predictions(
            output_path=args.output,
            n_seqs=args.n_trials, n=args.n, N=args.N,
            p_i=args.p_i, p_d=args.p_d, p_s=args.p_s,
            joint=not args.separate, seed=args.seed,
            n_workers=args.workers)

    else:
        rng = np.random.RandomState(args.seed)
        q = 4
        A = args.N if args.N is not None else 2

        channel_stats = {
            'insertion_probability': args.p_i,
            'deletion_probability': args.p_d,
            'substitution_probability': args.p_s,
        }

        acc_single_all, acc_sep_all, acc_joint_all = [], [], []

        for trial in range(args.n_trials):
            gt = generate_ground_truth_sequence(args.n, rng)
            x_int = str_to_ints(gt)

            reads_str = [IDS_channel(gt, channel_stats, rng) for _ in range(A)]
            reads_int = [str_to_ints(r) for r in reads_str]

            post_single = ids_posterior(reads_int[0], args.n, args.p_i, args.p_d, args.p_s, q)
            post_sep = ids_posterior(reads_int, args.n, args.p_i, args.p_d, args.p_s, q, joint=False)
            post_joint = ids_posterior(reads_int, args.n, args.p_i, args.p_d, args.p_s, q, joint=True)

            a1 = (post_single.argmax(1) == x_int).mean()
            a_s = (post_sep.argmax(1) == x_int).mean()
            a_j = (post_joint.argmax(1) == x_int).mean()

            acc_single_all.append(a1)
            acc_sep_all.append(a_s)
            acc_joint_all.append(a_j)

            print(f"  trial {trial + 1:3d}/{args.n_trials}: "
                  f"single={a1:.0%}  separate={a_s:.0%}  joint={a_j:.0%}  "
                  f"gt={gt}  reads={[len(r) for r in reads_str]}")

        print(f"\n{'='*60}")
        print(f"n={args.n}, N={A}, {args.n_trials} trials, "
              f"p_i={args.p_i}, p_d={args.p_d}, p_s={args.p_s}")
        print(f"{'='*60}")
        print(f"Single read:        {np.mean(acc_single_all):.1%} +/- {np.std(acc_single_all):.1%}")
        print(f"Separate ({A} reads): {np.mean(acc_sep_all):.1%} +/- {np.std(acc_sep_all):.1%}")
        print(f"Joint    ({A} reads): {np.mean(acc_joint_all):.1%} +/- {np.std(acc_joint_all):.1%}")
        print()
        n_better = sum(j > s for j, s in zip(acc_joint_all, acc_sep_all))
        n_equal = sum(j == s for j, s in zip(acc_joint_all, acc_sep_all))
        n_worse = sum(j < s for j, s in zip(acc_joint_all, acc_sep_all))
        print(f"Joint better than separate: {n_better}/{args.n_trials}")
        print(f"Joint equal to separate:    {n_equal}/{args.n_trials}")
        print(f"Joint worse than separate:  {n_worse}/{args.n_trials}")
