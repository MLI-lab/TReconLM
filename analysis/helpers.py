import numpy as np
import re

def find_homopolymers(seq, min_len=3):
    """Return list of (start, end, base) tuples for each homopolymer.
    Note: end is EXCLUSIVE (one past the last position of the HP)
    Example: AAA at positions 2,3,4 returns (2, 5, 'A')
    """
    homopolymers = []
    pattern = rf'(A{{{min_len},}}|C{{{min_len},}}|G{{{min_len},}}|T{{{min_len},}})'
    for m in re.finditer(pattern, seq):
        homopolymers.append((m.start(), m.end(), seq[m.start()]))
    return homopolymers

def get_edit_operations(gt, pred):
    """
    Compute minimal edit operations using dynamic programming.
    Returns list of operations: ('sub', pos, old, new), ('ins', pos, base), ('del', pos, base)
    """
    m, n = len(gt), len(pred)
    
    # DP table for edit distance
    dp = np.zeros((m+1, n+1), dtype=int)
    
    # Initialize
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            if gt[i-1] == pred[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # deletion
                                   dp[i][j-1],    # insertion
                                   dp[i-1][j-1])  # substitution
    
    # Backtrack to find actual operations
    operations = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and gt[i-1] == pred[j-1]:
            # Match - no operation
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            # Substitution
            operations.append(('sub', i-1, gt[i-1], pred[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            # Deletion
            operations.append(('del', i-1, gt[i-1]))
            i -= 1
        else:
            # Insertion
            operations.append(('ins', i, pred[j-1]))
            j -= 1
    
    return operations

def extract_context(sequence, pos, window_size):
    """
    Extract context window around position.
    Returns the k-mer centered at pos, or None if too close to edges.
    """
    half_window = window_size // 2
    start = pos - half_window
    end = pos + half_window + 1
    
    # Skip if too close to sequence edges
    if start < 0 or end > len(sequence):
        return None
    
    return sequence[start:end]