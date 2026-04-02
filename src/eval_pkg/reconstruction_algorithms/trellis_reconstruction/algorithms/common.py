import math
import logging
from decimal import *
import numpy as np
from Levenshtein import distance as levenshtein_distance
import networkx as nx
from collections import defaultdict
import random


USE_LOG_PROB = True


def hamming_distance(ground_truth, reconstructed):
    """
    Computes normalized Hamming distance between ground_truth and reconstructed sequence.
    
    If reconstructed is shorter than ground_truth, randomly fill with A/C/G/T.
    If reconstructed is longer, cut it to the ground_truth length.
    """

    L = len(ground_truth)
    reconstructed = list(reconstructed)

    if len(reconstructed) < L:
        # Pad randomly until reaching length L
        bases = ['A', 'C', 'G', 'T']
        while len(reconstructed) < L:
            reconstructed.append(random.choice(bases))
    elif len(reconstructed) > L:
        # Cut down to L
        reconstructed = reconstructed[:L]

    # Now compute Hamming distance
    mismatches = sum(c1 != c2 for c1, c2 in zip(ground_truth, reconstructed))
    normalized_distance = mismatches / L

    return normalized_distance

#For consistency use the same function for hamming distance 
#def hamming_distance(str1, str2): 
#    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


def compute_Fs_for_all_nodes(trellis: nx.DiGraph, topological_ordering):
    origin_vertex = topological_ordering[0]
    F_values = dict()
    if USE_LOG_PROB:
        F_values[origin_vertex] = math.log(1)
    else:
        F_values[origin_vertex] = Decimal(1)
    for v in topological_ordering[1:]:
        F_values[v] = compute_F_value_for_single_node(trellis, v, F_values)
    logging.debug("F_values:")
    logging.debug(F_values)
    #print("F_values:")
    #print(F_values)
    return F_values


def compute_F_value_for_single_node(trellis, v, F_values):
    if USE_LOG_PROB:
        f_value = -math.inf
    else:
        f_value = Decimal(0)
    for (from_v, to_v, weight) in trellis.in_edges(v, data="weight"):
        if to_v != v:
            logging.error("BUG, to_v should equal v")
        if USE_LOG_PROB:
            f_value = np.logaddexp(f_value, math.log(weight) + F_values[from_v])
        else:
            f_value += Decimal(weight) * F_values[from_v]

    logging.debug(f"F[{v}]={f_value}")
    return f_value


def compute_Bs_for_all_nodes(trellis: nx.DiGraph, topological_ordering, traces):
    B_values = dict()
    absorbing_state_indices = tuple([len(t) for t in traces])
    for v in reversed(topological_ordering):
        B_values[v] = compute_B_value_for_single_node(trellis, v, traces, B_values)
    logging.debug("B_values:")
    logging.debug(B_values)
    #print("B_values:")
    #print(B_values)
    return B_values


def compute_B_value_for_single_node(trellis: nx.DiGraph, v, traces, B_values):
    # TODO: optimize, compute once
    absorbing_state_indices = tuple([len(t) for t in traces])
    if len(trellis.edges(v)) == 0:
        # absorbing state
        if v.indices == absorbing_state_indices:
            logging.debug(f"REAL absorbing state {v}")
            if USE_LOG_PROB:
                b_value = math.log(1)
            else:
                b_value = Decimal(1)
        else:
            logging.debug(f"FAKE absorbing state {v}")
            if USE_LOG_PROB:
                b_value = -math.inf
                #b_value = math.log(1)
            else:
                b_value = Decimal(0)
                #b_value = Decimal(1)
    else:
        if USE_LOG_PROB:
            b_value = -math.inf
        else:
            b_value = 0
        for (from_v, to_v, weight) in trellis.edges(v, data="weight"):
            if USE_LOG_PROB:
                b_value = np.logaddexp(b_value, math.log(weight) + B_values[to_v])
            else:
                b_value += Decimal(weight) * B_values[to_v]

    return b_value


def compute_Sm_foreach_stage_and_symbol(trellis: nx.DiGraph, traces):
    Sm_per_stage = defaultdict(lambda: defaultdict(list))
    # for stage in range(max_stage+1):
    #    for c in Alphabet:
    #        # Sm_per_stage[stage][c].append()
    for v in trellis.nodes:
        #if v.sub_stage == len(traces) - 1:
        if v.sub_stage == len(traces):
            # last sub stage in stage before moving to next symbol
            Sm_per_stage[v.stage][v.symbol].append(v)
    return Sm_per_stage


def get_vertices_by_stage_by_substage_sorted_topologically(topological_ordering):
    vertices_dict = defaultdict(lambda: defaultdict(list))
    for v in topological_ordering:
        vertices_dict[v.stage][v.sub_stage].append(v)

    return vertices_dict
