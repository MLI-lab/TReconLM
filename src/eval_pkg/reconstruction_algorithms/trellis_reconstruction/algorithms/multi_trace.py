import random
import math

import networkx as nx

from consts import *
import algorithms.common
import trellis_graph

from typing import NamedTuple

import numpy as np
from scipy.stats import rv_discrete
from enum import IntEnum


def compute_multi_trace_estimation(traces, original):
    trellis = trellis_graph.build_new(original, traces)
    return compute_marginal_prob_for_each_vertex(trellis, traces, original)


def compute_marginal_prob_for_each_vertex(trellis, traces, original):
    topological_ordering = list(nx.topological_sort(trellis))
    logging.info(topological_ordering)
    F_values = algorithms.common.compute_Fs_for_all_nodes(trellis, topological_ordering)
    B_values = algorithms.common.compute_Bs_for_all_nodes(trellis, topological_ordering, traces)
    joint_probabilities = dict()
    for v in trellis.nodes:
        if USE_LOG_PROB:
            joint_probabilities[v] = F_values[v] + B_values[v]
        else:
            joint_probabilities[v] = F_values[v] * B_values[v]
        logging.debug(f"{v}: {joint_probabilities[v]} (F:{F_values[v]}, B:{B_values[v]})")

    # max_stage = max(trellis.nodes, key=lambda v: v.stage).stage
    max_stage = len(original)
    # TODO: is this necessary? we have out of bound edges by 1
    # max_stage -= 1
    logging.info(f"max stage: {max_stage}")
    logging.debug(trellis.nodes)
    # TODO: move to function
    traces_lengths = tuple([len(t) for t in traces])
    most_likely_original = []
    Sm = algorithms.common.compute_Sm_foreach_stage_and_symbol(trellis, traces)
    # for stage in range(0, max_stage+1):
    for stage in range(0, max_stage):
        # compute V for each stage
        prob_vector = dict()
        for c in Alphabet:
            if USE_LOG_PROB:
                V_of_symbol = np.logaddexp.reduce([joint_probabilities[v] for v in Sm[stage][c]])
            else:
                V_of_symbol = sum(joint_probabilities[v] for v in Sm[stage][c])
            # symbol_in_stage_vertex = TrellisVertex(stage, len(traces)-1, traces_lengths, c)
            # print(f"{symbol_in_stage_vertex}: {joint_probabilities[symbol_in_stage_vertex]} (F:{F_values[symbol_in_stage_vertex]}, B:{B_values[symbol_in_stage_vertex]})")
            prob_vector[c] = V_of_symbol

        if USE_LOG_PROB:
            sum_before_normalizing = np.logaddexp.reduce(list(prob_vector.values()))
            print(f"prob vector: {prob_vector}")
            print(f"exp of prob vector sum: {math.exp(sum_before_normalizing)}")
        else:
            sum_before_normalizing = sum(prob_vector.values())
            print(f"sum for stage {stage} is {sum_before_normalizing}. prob_vector: {prob_vector}")
        # normalize vector
        # factor = 1 / sum_before_normalizing
        # print(f"normalizing factor: {factor}")
        #
        # prob_vector_normalized = {v: p*factor for v, p in prob_vector.items()}
        # sum_after_normalizing = sum(prob_vector_normalized.values())
        # print(f"NORMALIZED: sum for stage {stage} is {sum_after_normalizing}. prob_vector: {prob_vector_normalized}")

        # most_likely_symbol = max(prob_vector_normalized.items(), key=lambda p: p[1])[0]
        most_likely_symbol = max(prob_vector.items(), key=lambda p: p[1])[0]
        print(f"most likely symbol for stage {stage} = {most_likely_symbol}")
        most_likely_original.append(most_likely_symbol)

    estimated_original_str = "".join(most_likely_original)
    print(f"most likely original: {estimated_original_str}")
    hamm = algorithms.common.hamming_distance(original, estimated_original_str)
    print(f"hamming distance: {hamm}")
    levenstein = algorithms.common.levenshtein_distance(original, estimated_original_str)
    print(f"levenstein distance: {levenstein}")
    print(f"traces: {traces}")

    return original, estimated_original_str, hamm, levenstein
