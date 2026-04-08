import numpy as np
import random
from ..utils.data_functions import filter_string
from ..utils.helper_functions import compute_homopolymer_map, weighted_choice
import sys
import json

def generate_insertion_pattern(n):
    """
    Generates all possible insertion patterns for a sequence of length `n`.

    Each pattern is a string of length `n` with exactly one 'I' (representing an insertion)
    and all other positions as '-' (no insertion). The 'I' is placed in a all possible different position
    in to generate all possible pattern.

    Parameters:
        n (int): Length of the sequence.

    Returns:
        List[str]: A list of insertion patterns, one for each possible insertion position.
    """
    patterns = []
    
    # Loop through the range of n to generate each pattern
    for i in range(n):
        # Generate a pattern for the current position
        pattern = '-' * i + 'I' + '-' * (n - i - 1)
        # Append the generated pattern to the list
        patterns.append(pattern)
    
    return patterns

def generate_alignment_pattern(sequences):
    """
    Aligns a list of sequences that may contain insertions marked by 'I', producing equal-length alignment patterns.

    Insertions ('I') represent bases that were not present in the ground truth and are therefore not aligned to any
    specific position in the original sequence. When multiple sequences have insertions at the same reference position,
    there is no guarantee these insertions are related or semantically equivalent.

    To avoid falsely aligning such unrelated insertions, each is assigned a unique insertion pattern (e.g., 'I-', '-I'),
    spreading insertions across adjacent alignment columns. Sequences without insertions in that column are padded with
    an equal number of '-' characters to maintain alignment length consistency.

    If no insertions are present in a column, characters are aligned directly (with '-' for deletion). The final output ensures all sequences
    are of equal length, with insertions and other characters consistently placed.

    Parameters:
        sequences (List[str]): List of input sequences to align. Each sequence may contain 'I' to indicate an insertion.

    Returns:
        List[str]: Aligned versions of the input sequences, padded and structured to be of equal length.
    """

    n = len(sequences)

    alignment_pattern = ['' for _ in range(n)] #One output string per input sequence

    t_vec = [[0] for _ in range(n)]
    i = 0
    
    while (any(len(s) > 0 for s in sequences)): # Keep aligning until all sequences are fully consumed

        column = [s[0] if s != '' else '-' for s in sequences] # Take the first character of each sequence (or '-' if the sequence is empty), that’s the current column we’re aligning: E.g. sequences = ['IATC', 'ICGA', 'ACGT'], column = ['I','I','A']
        check_value = 'I' in column
        
        if check_value == False: # if no I in current collumn just align chatacters normally 
            for column_index, char in enumerate(column):
                alignment_pattern[column_index] = alignment_pattern[column_index] + char

            sequences = [s[1:] for s in sequences]
            
        elif check_value == True:
            check_column = [elem == 'I' for elem in column] 
            sum_column = sum([1 for char in column if char == 'I']) #counts how many sequences have an 'I' at the current position. 

            insertion_pattern = generate_insertion_pattern(sum_column) #generate insertion patter of length number of insertions as, in example above, we  need to align 2 insertions across 2 positions, so we create 2 unique patterns that spread insertions accross colluns --> WHY CAN WE NOT HAVE INSERTIONS AT SAME POSITION?
            temp = 0
            for column_index, column_check_value in enumerate(check_column):

                if column_check_value == True: # Check if this collumn e.g. sequence has insertion, if yes give it one of the insertion patterns, if not, pad with '-' * sum_column
                    
                    alignment_pattern[column_index] =  alignment_pattern[column_index] + insertion_pattern[temp] 
                    sequences[column_index] = sequences[column_index][1:]
                    temp += 1
                
                elif column_check_value == False:
                    alignment_pattern[column_index] =  alignment_pattern[column_index] + '-' * sum_column

        i += 1
    return alignment_pattern

def generate_alignment(alignment_pattern):
    """
    Takes an alignment_pattern and translates special alignment symbols like 'I' and 'D' into real characters or gaps.
    """

    alphabet = ['A', 'C', 'G', 'T']

    for index, seq in enumerate(alignment_pattern):
        seq = list(seq)
        for i in range(len(seq)):
            if seq[i] == 'I':
                seq[i] = random.choice(alphabet)
            if seq[i] == 'D':
                seq[i] = '-'
        alignment_pattern[index] = ''.join(seq)

    return alignment_pattern

def replace_I(obs, alg):

    """
    Replaces all 'I' characters in the obs string (the observation) with the corresponding character at the same position from the alg string (the alignment).
    """
    
    if len(alg) != len(obs):
        print('error: strings are not of equal length')
        raise SyntaxError
    length = len(obs)

    obs = list(obs)
    alg = list(alg)

    for i in range(length):
        if obs[i] == 'I':
            obs[i] = alg[i]

    return ''.join(obs)

def IDS_alignment_channel(ground_truth_sequence, channel_statistics, observation_size, target_type, print_flag=False, rng=None):
    """
    Generates `observation_size` independent noisy copies of a ground truth DNA sequence
    using an Insertion–Deletion–Substitution (IDS) channel, along with their aligned representations.

    Insertions are handled based on the `target_type`:
        - 'std_*': insertions are initially marked with 'I' and resolved to random nucleotides after alignment.
        - 'ext_*': insertions are realized immediately as random nucleotides.

    All alignment sequences are padded to the same length using `generate_alignment_pattern`. For 'std_*' types,
    alignment and observation strings are further updated to replace all 'I' placeholders with concrete bases.

    Parameters:
        ground_truth_sequence (str): Clean DNA input sequence.
        channel_statistics (dict): Probabilities for 'insertion', 'deletion', and 'substitution'.
        observation_size (int): Number of corrupted observations to generate.
        target_type (str): One of 'std_MSA', 'ext_MSA', 'std_NESTED', or 'ext_NESTED'.
        print_flag (bool): Optional debug flag (unused).

    Returns:
        Tuple[List[str], List[str]]:
            - observation_list: Corrupted sequences with all insertions resolved to real bases.
            - alignment_list: Aligned sequences of equal length, indicating insertions and deletions.
    """

    rng = rng or random

    def ids_alignment(x, channel_statistics, target_type):

        y = []  # Output sequence
        alignment_seq = []  # Alignment sequence

        t = 0
        alphabet = ['A', 'C', 'G', 'T']
        length = len(x)

        pi = channel_statistics['insertion_probability']
        pd = channel_statistics['deletion_probability']
        ps = channel_statistics['substitution_probability']

        while (t < length): # t is position in the sequence, if we insert do not increment 
            rd = rng.uniform(0.0, 1.0)

            if (rd<pi):

                if 'std' in target_type:
                    y.append('I')
                    alignment_seq.append('I')

                elif 'ext' in target_type:
                    char = rng.choice(alphabet)
                    y.append(char)
                    alignment_seq.append('I')  
                else:
                    print('error: target_type not defined')
                    raise SyntaxError
                
            elif (rd<(pi+pd)):
                alignment_seq.append('D')
                t += 1
                
            elif (rd<(pi+pd+ps)):
                sub_list = [letter for letter in alphabet if letter != x[t]]
                y_sub = rng.choice(sub_list)
                y.append(y_sub)
                alignment_seq.append(x[t])
                t += 1

            else:
                y.append(x[t])
                alignment_seq.append(x[t])
                t += 1

        y = ''.join(y)
        alignment_seq = ''.join(alignment_seq)

        return y, alignment_seq

    observation_list = []
    alignment_list = []

    # create channel matrices
    for j in range(observation_size):     
        #y, alignment_seq = IDS(ground_truth_sequence, channel_statistics)
        y, alignment_seq = ids_alignment(ground_truth_sequence, channel_statistics, target_type)
        observation_list.append(y)
        alignment_list.append(alignment_seq)

    alignment_list = generate_alignment_pattern(alignment_list)
    
    if 'std' in target_type:
        alignment_list = generate_alignment(alignment_list) # inserts a random base for I
    
        for index, (obs, alg) in enumerate(zip(observation_list, alignment_list)):

            alg = filter_string(alg)
            observation_list[index] = replace_I(obs, alg)

    return observation_list, alignment_list

def IDS_channel(x, channel_statistics, rng):
    """
    Simulates one corrupted version of a DNA sequence `x` by passing it through an 
    Insertion–Deletion–Substitution (IDS) channel.

    Each base in the input sequence has a chance of being:
        - Inserted: a random base is added (without advancing the input pointer)
        - Deleted: the current base is skipped
        - Substituted: the current base is replaced with a different one
        - Kept unchanged

    The probabilities for each type of corruption are specified in the `channel_statistics` dictionary.

    Parameters:
        x (str): The ground truth input DNA sequence (e.g., "ACGT...").
        channel_statistics (dict): A dictionary with keys:
            - 'insertion_probability': probability of inserting a random base
            - 'deletion_probability' : probability of deleting the current base
            - 'substitution_probability': probability of substituting the current base

    Returns:
        str: The corrupted output sequence `y`, potentially shorter or longer than `x`.

    Notes:
        - Insertions are realized by sampling a random base from ['A', 'C', 'G', 'T'] and appending it to the output.
        - Deletions are performed by skipping the base in the input.
        - Substitutions replace the base with a random, different base.
        - The function also internally records the type of each edit, but does not return these lists.
    """

    rng = rng or random 

    y = []  # Output sequence
        
    t = 0
    alphabet = ['A', 'C', 'G', 'T']
    length = len(x)

    insertion_list = []
    deletion_list = []
    substitution_list = []

    ids_print_flag = False

    pi = channel_statistics['insertion_probability']
    pd = channel_statistics['deletion_probability']
    ps = channel_statistics['substitution_probability']

    while (t < length):
        rd = rng.uniform(0.0, 1.0)

        if (rd<pi): #insert 
            char = rng.choice(alphabet)
            y.append(char)

            insertion_list.append(char)
            deletion_list.append('-')
            substitution_list.append('-')

        elif (rd<(pi+pd)): # delete 
            t += 1

            insertion_list.append('-')
            deletion_list.append('D')
            substitution_list.append('-')
                
        elif (rd<(pi+pd+ps)): #substitute 
            sub_list = [letter for letter in alphabet if letter != x[t]]
            y_sub = rng.choice(sub_list)
            y.append(y_sub)
            t += 1

            insertion_list.append('-')
            deletion_list.append('-')
            substitution_list.append(y_sub)

        else: #transmit
            y.append(x[t])
            t += 1

            insertion_list.append('-')
            deletion_list.append('-')
            substitution_list.append('-')

    y = ''.join(y)

    if ids_print_flag:
        print(insertion_list)
        print(deletion_list)
        print(substitution_list)

    return y


def _sample_burst_length(burst_weight_entry, rng):
    """Sample a burst length from the learned distribution.

    Args:
        burst_weight_entry: dict with 'lengths' (list[int]) and 'weights' (list[float]).
        rng: random number generator.

    Returns:
        int: sampled burst length (>= 1).
    """
    lengths = burst_weight_entry['lengths']
    weights = burst_weight_entry['weights']
    r = rng.random()
    cumsum = 0.0
    for l, w in zip(lengths, weights):
        cumsum += w
        if cumsum >= r:
            return l
    return lengths[-1]


def _partition_into_bursts(n_errors, burst_weights, rng):
    """Partition a total error count into burst events using the learned distribution.

    Samples burst lengths until the total equals or exceeds n_errors, then trims
    the last burst so the sum is exactly n_errors.

    Args:
        n_errors (int): total number of individual errors to distribute.
        burst_weights: dict with 'lengths' and 'weights' from error_model.
        rng: random number generator.

    Returns:
        list[int]: burst lengths summing to n_errors.
    """
    if n_errors <= 0:
        return []
    bursts = []
    remaining = n_errors
    while remaining > 0:
        bl = _sample_burst_length(burst_weights, rng)
        bl = min(bl, remaining)
        bursts.append(bl)
        remaining -= bl
    return bursts


def error_model_IDS_channel(x, n_sub, n_del, n_ins, error_model, homopolymer_map, rng=None):
    """
    Corrupts a DNA sequence with approximately n_sub substitutions, n_del deletions,
    and n_ins insertions, distributed across positions proportional to
    context-dependent weights (nucleotide, homopolymer, position zone).

    Errors are placed as bursts (consecutive runs) sampled from the learned burst
    length distribution. This matches real-world nanopore error patterns where
    ~15-20% of errors occur in consecutive runs of 2+.

    Parameters:
        x (str): Ground truth sequence.
        n_sub (int): Number of substitutions to apply.
        n_del (int): Number of deletions to apply.
        n_ins (int): Number of insertions to apply.
        error_model (dict): Preprocessed error model.
        homopolymer_map (list): Homopolymer run length at each position.
        rng: Random number generator.

    Returns:
        str: The corrupted sequence.
    """
    rng = rng or random
    alphabet = ['A', 'C', 'G', 'T']
    length = len(x)
    sub_weights_dict = error_model.get('sub_weights', {})
    ins_weights_dict = error_model.get('ins_weights', {})
    per_nt = error_model['per_nucleotide']
    multipliers = error_model['multipliers']
    burst_length_weights = error_model.get('burst_length_weights', {})

    # Step 1: compute per-position context weights for each error type
    w_sub = []
    w_del = []
    w_ins = []

    for t in range(length):
        base = x[t]
        hp_len = homopolymer_map[t] if t < len(homopolymer_map) else 1
        zone = 'start' if t < 10 else ('end' if t >= length - 10 else 'middle')

        # Base rates
        base_sub = per_nt.get(base, {}).get('sub_rate', 0.01)
        base_del = per_nt.get(base, {}).get('del_rate', 0.01)
        base_ins = per_nt.get(base, {}).get('ins_rate', 0.01)

        # Context multipliers
        hp_key = str(hp_len)
        hp_m = multipliers['homopolymer'].get(hp_key, {'sub': 1.0, 'del': 1.0, 'ins': 1.0})
        z_m = multipliers['position_zone'].get(zone, {'sub': 1.0, 'del': 1.0, 'ins': 1.0})

        w_sub.append(base_sub * hp_m['sub'] * z_m['sub'])
        w_del.append(base_del * hp_m['del'] * z_m['del'])
        w_ins.append(base_ins * hp_m['ins'] * z_m['ins'])

    # Normalize weights to probabilities
    def normalize(weights):
        total = sum(weights)
        if total == 0:
            return [1.0 / len(weights)] * len(weights)
        return [w / total for w in weights]

    def weighted_pick_one(available, weights_all, rng):
        """Pick one position from available indices, weighted by weights_all."""
        w = [weights_all[i] for i in available]
        total = sum(w)
        if total == 0:
            return available[rng.randint(0, len(available) - 1)]
        r = rng.random() * total
        cumsum = 0.0
        for i in range(len(available)):
            cumsum += w[i]
            if cumsum >= r:
                return available[i]
        return available[-1]

    def place_bursts(bursts, weights, occupied, length, rng):
        """Place burst events at weighted positions, extending consecutively.

        For each burst of length B, pick a start position from the available
        (non-occupied) positions weighted by `weights`, then mark the next B
        consecutive non-occupied positions starting from there.

        Args:
            bursts: list of burst lengths to place.
            weights: per-position weights for choosing start positions.
            occupied: set of already-occupied positions (modified in place).
            length: sequence length.
            rng: random number generator.

        Returns:
            set of all positions assigned to these bursts.
        """
        positions = set()
        for burst_len in bursts:
            available = [i for i in range(length) if i not in occupied]
            if not available:
                break
            start = weighted_pick_one(available, weights, rng)
            # Extend burst consecutively from start position
            placed = 0
            pos = start
            while placed < burst_len and pos < length:
                if pos not in occupied:
                    positions.add(pos)
                    occupied.add(pos)
                    placed += 1
                pos += 1
            # If we hit the end, wrap backwards from start
            pos = start - 1
            while placed < burst_len and pos >= 0:
                if pos not in occupied:
                    positions.add(pos)
                    occupied.add(pos)
                    placed += 1
                pos -= 1
        return positions

    # Step 2: partition error counts into bursts
    n_sub = min(n_sub, length)
    n_del = min(n_del, length)
    n_ins = min(n_ins, length)

    sub_burst_weights = burst_length_weights.get('substitution', {'lengths': [1], 'weights': [1.0]})
    del_burst_weights = burst_length_weights.get('deletion', {'lengths': [1], 'weights': [1.0]})
    ins_burst_weights = burst_length_weights.get('insertion', {'lengths': [1], 'weights': [1.0]})

    sub_bursts = _partition_into_bursts(n_sub, sub_burst_weights, rng)
    del_bursts = _partition_into_bursts(n_del, del_burst_weights, rng)
    ins_bursts = _partition_into_bursts(n_ins, ins_burst_weights, rng)

    # Step 3: place bursts at weighted positions
    occupied = set()
    sub_positions = place_bursts(sub_bursts, w_sub, occupied, length, rng)
    del_positions = place_bursts(del_bursts, w_del, occupied, length, rng)

    # For insertions: place burst start positions, then insert multiple bases there
    ins_at_pos = {}
    ins_available = list(range(length))
    for burst_len in ins_bursts:
        if not ins_available:
            break
        start = weighted_pick_one(ins_available, w_ins, rng)
        # Consecutive insertion burst: insert bases at consecutive positions
        placed = 0
        pos = start
        while placed < burst_len and pos < length:
            ins_at_pos[pos] = ins_at_pos.get(pos, 0) + 1
            placed += 1
            pos += 1
        # Wrap backwards if needed
        pos = start - 1
        while placed < burst_len and pos >= 0:
            ins_at_pos[pos] = ins_at_pos.get(pos, 0) + 1
            placed += 1
            pos -= 1

    # Step 4: build the read
    read = []
    for t in range(length):
        if t in del_positions:
            # Deletion: skip this base
            pass
        elif t in sub_positions:
            # Substitution: replace with a different base
            base = x[t]
            if base in sub_weights_dict:
                sw = sub_weights_dict[base]
                read.append(weighted_choice(sw['targets'], sw['weights'], rng))
            else:
                sub_list = [b for b in alphabet if b != base]
                read.append(rng.choice(sub_list))
        else:
            # No error: keep original base
            read.append(x[t])

        # Apply insertions after this position (biased by learned insertion frequencies)
        if t in ins_at_pos:
            ref_base = x[t]
            for _ in range(ins_at_pos[t]):
                if ref_base in ins_weights_dict:
                    iw = ins_weights_dict[ref_base]
                    read.append(weighted_choice(iw['targets'], iw['weights'], rng))
                else:
                    read.append(rng.choice(alphabet))

    return ''.join(read)


if __name__ == '__main__':

    #random.seed(42)
    #np.random.seed(42)

    test_size = int(1e0)
    test_size = 1

    length_ground_truth = 10
    observation_size = 5
    print_flag = False
    channel_statistics = {'substitution_probability': 0.1, 'deletion_probability': 0.1, 'insertion_probability': 0.1}

    ham_arr = np.zeros(test_size)
    lev_arr = np.zeros(test_size)

    target_type = 'CPRED'

    if target_type == 'CPRED':

        observation_list = []
        ground_truth_sequence = ''.join(random.choices('ACTG', k=length_ground_truth))
        print(ground_truth_sequence)
        print('##################################################')

        for i in range(test_size):
            obs_seq = IDS_channel(ground_truth_sequence, channel_statistics)
            print(ground_truth_sequence)
            print(obs_seq)
            observation_list.append(obs_seq)    
        
        print('------------------------------------------------------------')
        print('------------------------------------------------------------')
        
    else:
        
        for i in range(test_size):
            ground_truth_sequence = ''.join(random.choices('ACTG', k=length_ground_truth))
            observation_list, alignment = IDS_alignment_channel(ground_truth_sequence = ground_truth_sequence, channel_statistics = channel_statistics,
                                               observation_size = observation_size,
                                                target_type =  target_type, print_flag = False)
