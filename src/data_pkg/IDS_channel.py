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


def realistic_IDS_channel(x, error_model, gc_bin, homopolymer_map, rng=None):
    """
    Simulates one corrupted version of a DNA sequence using a realistic error model
    estimated from real-world data.

    Unlike IDS_channel which uses fixed error probabilities for all positions,
    this function uses context-dependent rates based on:
    - The nucleotide at each position (A/C/G/T)
    - Homopolymer run length (1, 2, 3, 4, 5+)
    - GC content of the sequence (low/mid/high)
    - Position zone (start/middle/end)

    Error rates are looked up from joint statistics when sufficient data exists
    (>= 1000 observations), otherwise computed multiplicatively from marginal rates.

    Parameters:
        x (str): The ground truth input DNA sequence.
        error_model (dict): Preprocessed error model from load_error_model().
        gc_bin (str): GC content bin for this sequence ('low', 'mid', or 'high').
        homopolymer_map (list): Precomputed homopolymer run length at each position.
        rng: Random number generator (random.Random instance or None).

    Returns:
        str: The corrupted output sequence.
    """
    rng = rng or random
    alphabet = ['A', 'C', 'G', 'T']
    length = len(x)
    y = []
    t = 0

    joint_rates = error_model['joint_rates']
    per_nt = error_model['per_nucleotide']
    multipliers = error_model['multipliers']
    sub_weights = error_model['sub_weights']          # {base: {targets: [...], weights: [...]}}
    burst_weights = error_model['burst_length_weights']  # {type: {lengths: [...], weights: [...]}}

    while t < length:
        base = x[t]
        hp_len = homopolymer_map[t] if t < len(homopolymer_map) else 1
        zone = 'start' if t < 10 else ('end' if t >= length - 10 else 'middle')

        # Look up error rates
        jkey = f"{base}|hp={hp_len}|gc={gc_bin}|{zone}"

        if jkey in joint_rates and joint_rates[jkey]['total'] >= 1000:
            # Use joint rate: sample from CI
            jr = joint_rates[jkey]
            p_sub = rng.uniform(jr['sub_rate_ci'][0], jr['sub_rate_ci'][1])
            p_del = rng.uniform(jr['del_rate_ci'][0], jr['del_rate_ci'][1])
            p_ins = rng.uniform(jr['ins_rate_ci'][0], jr['ins_rate_ci'][1])
        else:
            # Fallback: multiplicative model
            base_sub = per_nt[base]['sub_rate']
            base_del = per_nt[base]['del_rate']
            base_ins = per_nt[base]['ins_rate']

            hp_key = str(hp_len)
            hp_m = multipliers['homopolymer'].get(hp_key, {'sub': 1.0, 'del': 1.0, 'ins': 1.0})
            z_m = multipliers['position_zone'].get(zone, {'sub': 1.0, 'del': 1.0, 'ins': 1.0})
            gc_m_key = '<0.30' if gc_bin == 'low' else ('>0.70' if gc_bin == 'high' else '0.30-0.70')
            gc_m = multipliers['gc_content'].get(gc_m_key, {'sub': 1.0, 'del': 1.0, 'ins': 1.0})

            p_sub = base_sub * hp_m['sub'] * z_m['sub'] * gc_m['sub']
            p_del = base_del * hp_m['del'] * z_m['del'] * gc_m['del']
            p_ins = base_ins * hp_m['ins'] * z_m['ins'] * gc_m['ins']

        # Clamp total probability to < 1
        p_total = p_sub + p_del + p_ins
        if p_total >= 1.0:
            scale = 0.95 / p_total
            p_sub *= scale
            p_del *= scale
            p_ins *= scale

        # Roll dice
        rd = rng.uniform(0.0, 1.0)

        if rd < p_del:
            # Deletion: sample burst length
            bw = burst_weights.get('deletion', {'lengths': [1], 'weights': [1.0]})
            burst = weighted_choice(bw['lengths'], bw['weights'], rng)
            t += min(burst, length - t)  # don't skip past end

        elif rd < p_del + p_ins:
            # Insertion: sample burst length, insert random bases
            bw = burst_weights.get('insertion', {'lengths': [1], 'weights': [1.0]})
            burst = weighted_choice(bw['lengths'], bw['weights'], rng)
            for _ in range(burst):
                y.append(rng.choice(alphabet))
            # Don't advance t — current base still needs processing

        elif rd < p_del + p_ins + p_sub:
            # Substitution: sample target base from real distribution
            if base in sub_weights:
                sw = sub_weights[base]
                y.append(weighted_choice(sw['targets'], sw['weights'], rng))
            else:
                sub_list = [b for b in alphabet if b != base]
                y.append(rng.choice(sub_list))
            t += 1

        else:
            # No error
            y.append(x[t])
            t += 1

    return ''.join(y)


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
