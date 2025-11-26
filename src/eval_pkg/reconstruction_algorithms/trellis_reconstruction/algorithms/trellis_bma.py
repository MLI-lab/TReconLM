from .. import trellis_graph
from . import common
from .. import consts

from typing import NamedTuple, List, Dict, DefaultDict
import networkx as nx
import numpy as np
from collections import defaultdict
import math
from . import common
import src.eval_pkg.reconstruction_algorithms.trellis_reconstruction.algorithms.common as algorithms
from ..consts import Alphabet



ESTIMATE_SECOND_HALF_REVERSED = False
USE_LOG_PROB = True  # or False if you prefer


#Each trace builds a trellis graph where:
#- Each vertex represents a symbol (A, C, G, T) along with an error type (substitution, insertion, deletion, correct).
#- Each edge corresponds to a possible mutation (ins/del/sub/correct) with an associated probability.

#Forward probabilities (F-values) at each vertex represent:
#- The likelihood of reaching this vertex starting from the beginning of the sequence, 
#- Accumulating the probabilities along the incoming edges (in log-space or normal space).

#Backward probabilities (B-values) at each vertex represent:
#- The likelihood of successfully completing the sequence from this vertex to the end,
#- Accumulating probabilities along outgoing paths.

#At each position (stage) in the reconstructed sequence:
#- For each possible base (A/C/G/T), we look at vertices emitting that symbol across all traces.
#- For each trace:
#    - Combine forward probability and scaled backward probability (scaling via beta_b).
#    - Aggregate over all relevant vertices for the symbol to get an internal score (V^k).
#- Across traces:
#    - Adjust the internal score by combining it with the corresponding scores from other traces 
#      (weighted by beta_i and beta_e parameters) to compute a "gamma" for each symbol.

#Normalization:
#- Normalize gamma scores across symbols to avoid numerical drift, 
#  resulting in a probability distribution over {A, C, G, T} at each position.

#Decision:
#- Choose the base with the highest normalized gamma score as the reconstructed base at that stage.

#Update:
#- After deciding a symbol at a stage, forward (or backward if reversed) probabilities 
#  are updated to propagate the impact of the decision onto future stages.

#Optional reversal:
#- If `ESTIMATE_SECOND_HALF_REVERSED` is set, the second half of the sequence is reconstructed 
#  from right to left (on reversed traces) to better handle insertion and deletion errors.

class TrellisBMAParams:
    def __init__(self, beta_b: float, beta_i: float, beta_e: float, P_DEL: float, P_SUB: float, P_INS: float):
        self.beta_b = beta_b
        self.beta_i = beta_i
        self.beta_e = beta_e
        self.P_DEL = P_DEL
        self.P_SUB = P_SUB
        self.P_INS = P_INS
        self.P_COR = 1 - P_DEL - P_SUB - P_INS


class TrellisMetadata(NamedTuple): 
    """
    Each trace gets one TrellisMetadata with precomputed forward (F_values) and backward (B_values) probabilities.
    Sm_per_stage_per_symbol tells you at each stage what vertices correspond to A/C/G/T.
    Vk_estimations_per_stage_per_symbol holds local estimates at each stage and symbol for one trellis.

    Topological_ordering is simply a list of all vertices (nodes) in the trellis graph.
    Ordered so that if node A points to node B (A → B), then A comes before B in the: Parents (previous steps) come before children (next steps).
    This ensures that when processing forward or backward, we never process a node before its dependencies are processed.

    vertices_by_stage_by_substage_sorted_topologically organizes all vertices from all traces into a nested dictionary:

    The first key is the stage (i.e., position in the original sequence we want to reconstruct).
    The second key is the substage, grouping vertices belonging to the same trace or event type (insertions, deletions, etc.).
    Each value is a list of TrellisVertex objects, sorted topologically.
    Each TrellisVertex represents a possible event (match, substitution, insertion, deletion) and will later contribute to estimating the most likely base (A, C, G, T) at that stage.
    {
        0: {  # stage 0
            0: [Vertex_A_0],  # Trace A nodes
            1: [Vertex_B_0, Vertex_B_0_insertion]  # Trace B nodes
        },
        1: {  # stage 1
            0: [Vertex_A_1, Vertex_B_1]
        },
        2: {
            0: [],  # (might be empty if deletion)
            1: [Vertex_B_2]
        }
    }

    """
    trellis: nx.DiGraph
    topological_ordering: List[trellis_graph.TrellisVertex]
    F_values: Dict[trellis_graph.TrellisVertex, float]
    B_values: Dict[trellis_graph.TrellisVertex, float]
    Sm_per_stage_per_symbol: Dict[int, Dict[str, List[trellis_graph.TrellisVertex]]]
    vertices_by_stage_by_substage_sorted_topologically: Dict[int, Dict[int, List[trellis_graph.TrellisVertex]]]
    Vk_estimations_per_stage_per_symbol: DefaultDict[int, Dict[str, float]] = {}


def compute_trellis_bma_estimation(traces, original, params: TrellisBMAParams):
    """
    Reconstructs a sequence from noisy traces using trellis-based Bayesian multiple alignment (BMA).
    
    If enabled, the sequence is reconstructed in two halves: 
    the first half is reconstructed normally, and the second half is reconstructed 
    by building trellises on reversed traces and reversed ground truth.
    
    Returns the original sequence, reconstructed sequence, Hamming distance, and Levenshtein distance.
    """

    if ESTIMATE_SECOND_HALF_REVERSED:
        first_half_estimates = build_trellis_and_estimate(traces, original, len(original) // 2, params)
        reversed_traces = [t[::-1] for t in traces]
        reversed_original = original[::-1]
        second_half_length = len(original) - (len(original)//2)
        second_half_estimates = list(reversed(build_trellis_and_estimate(reversed_traces, reversed_original, second_half_length, params)))
        final_estimate = first_half_estimates + second_half_estimates
    else:
        final_estimate = build_trellis_and_estimate(traces, original, len(original), params)

    final_estimate_str = "".join(final_estimate)

    return original, final_estimate_str


def build_trellis_and_estimate(traces, original, estimation_len, params: TrellisBMAParams):
    trellises_metadata = []

    for trace in traces:
        trellis = trellis_graph.build_new(original, [trace], params)
        topological_ordering = list(nx.topological_sort(trellis))
        F_values = algorithms.compute_Fs_for_all_nodes(trellis, topological_ordering)
        B_values = algorithms.compute_Bs_for_all_nodes(trellis, topological_ordering, [trace])
        Sm_per_stage = algorithms.compute_Sm_foreach_stage_and_symbol(trellis, [trace])
        vertices_by_stage_by_substage = algorithms.get_vertices_by_stage_by_substage_sorted_topologically(
            topological_ordering)
        vk_estimations = defaultdict(dict)
        trellises_metadata.append(TrellisMetadata(trellis, topological_ordering,
                                                  F_values, B_values, Sm_per_stage,
                                                  vertices_by_stage_by_substage, vk_estimations))

    V_per_index_per_symbol = defaultdict(dict) #V_per_index_per_symbol[stage][symbol] = overall score for symbol at a given stage, across traces
    estimates = []

    for stage in range(estimation_len): # Loop through each stage (position in target sequence):
        # Updates probabilities (forward/backward) inside each trellis based on agreement across traces and computes a combined score for each symbol at this stage.
        sync_probabilities_and_estimate_stage(stage, trellises_metadata, traces,
                                              V_per_index_per_symbol[stage], params, is_first_half=True)

        #For the current stage: Among A, C, T, G, pick the symbol with the highest combined score across traces.
        estimates.append(max(Alphabet, key=lambda c: V_per_index_per_symbol[stage][c]))

    return estimates


def sync_probabilities_and_estimate_stage(stage, trellises_metadata, traces, V_per_symbol, params: TrellisBMAParams, is_first_half=True):
    # compute V^k for all trellises
    for trace_idx, trellis_metadata in enumerate(trellises_metadata):
        #For each trace's trellis: compute_vk_for_trellis computes V^k(stage, symbol):
        # How likely each symbol is at this stage inside this trace. Combines forward value F(v) and backward value B(v) per vertex.
        # After this, each trellis knows internally how much it "votes" for A, C, G, T at this stage.
        compute_vk_for_trellis(stage, trace_idx, trellis_metadata, params)

    # Use V^k to update forward values for each trellis
    for trace_idx, trellis_metadata in enumerate(trellises_metadata):
        # Compute gamma^k(m) for all trellises
        # how strongly traces agree on each symbol.
        # output is final score, i.e. how likely symbol is, for each symbol at a stage 
        symbol_gammas = {c: compute_gamma_coeff_for_trellis_and_symbol(stage, c, trellis_metadata, trellises_metadata, params)
                         for c in Alphabet}
        # normalize gammas to avoid drift (numbers getting too large or small):    
        # In log-space: subtract the maximum gamma.
        #Otherwise: divide so probabilities sum to 1.
        if USE_LOG_PROB:
            factor = max(symbol_gammas.values())
            # subtract the maximum gamma for relative preferences between symbols stay the same, just all values get shifted down to avoid big numbers.
            normalized_gammas = {c: symbol_gammas[c] - factor for c in Alphabet}
        else:
            factor = 1 / sum(symbol_gammas.values())
            # divide each gamma by the total sum.
            normalized_gammas = {c: symbol_gammas[c] * factor for c in Alphabet}

        # inject the new estimated symbol probabilities into the trellis nodes.   
        # So if we process string normally we update all forward probabilities for next symbols and if we process string in reverse we update the backward probabilities    
        # So we decide sequentially, stage-by-stage, picking one symbol at a time and decidions influence later decisions  
        for symbol in Alphabet:
            for v in trellis_metadata.Sm_per_stage_per_symbol[stage][symbol]:
                if is_first_half:
                    if USE_LOG_PROB:
                        trellis_metadata.F_values[v] += normalized_gammas[symbol]
                    else:
                        trellis_metadata.F_values[v] *= normalized_gammas[symbol]
                else:
                    if USE_LOG_PROB:
                        trellis_metadata.B_values[v] += normalized_gammas[symbol]
                    else:
                        trellis_metadata.B_values[v] *= normalized_gammas[symbol]
        # then recompute the forward or backward probabilities of the next symbols with the updated scores                             
        if is_first_half:
            for substage in [-1] + list(range(len(traces))):
                for v in trellis_metadata.vertices_by_stage_by_substage_sorted_topologically[stage + 1][substage]:
                    new_f = algorithms.compute_F_value_for_single_node(trellis_metadata.trellis, v,
                                                                              trellis_metadata.F_values)
                    trellis_metadata.F_values[v] = new_f
        else:

            for substage in reversed([-1] + list(range(len(traces)))):
                for v in trellis_metadata.vertices_by_stage_by_substage_sorted_topologically[stage][substage]:
                    new_b = algorithms.compute_B_value_for_single_node(trellis_metadata.trellis, v, traces,
                                                                              trellis_metadata.B_values)
                    trellis_metadata.B_values[v] = new_b

    # combine V^k to compute V(M_l = m)
    # Aggregates the Vk estimations for each base (A, C, G, T) across all traces. ( earlier we computed a score for each trace now we need to combine )
    for symbol in Alphabet:
        if USE_LOG_PROB:
            V_per_symbol[symbol] = sum(metadata.Vk_estimations_per_stage_per_symbol[stage][symbol]
                                       for metadata in trellises_metadata)
        else:
            V_per_symbol[symbol] = math.prod(metadata.Vk_estimations_per_stage_per_symbol[stage][symbol]
                                             for metadata in trellises_metadata)


def compute_vk_for_trellis(stage: int, trace_idx: int, trellis_metadata: TrellisMetadata, params: TrellisBMAParams):#, dominant_prob_dict, secondary_prob_dict):
    # For each symbol, get all vertices in the trellis at this stage corresponding to that symbol ( i.e. a vertex represents symbol and what happened i.e. sub/ins/del etc. )
    # # Forward probabilities F(v) are computed by propagating along edges and accumulating insertion, deletion, substitution, and correct-match probabilities.
    for symbol in Alphabet:
        stage_final_vertices = trellis_metadata.Sm_per_stage_per_symbol[stage][symbol]

        if USE_LOG_PROB:
            # We don't scale the forward probability here because forward propagation already includes the edge probabilities naturally (insertions, deletions, substitutions), and because the balance between forward and backward information is controlled only through scaling the backward scores with beta_b.
            f_values = [trellis_metadata.F_values[v] for v in stage_final_vertices]
            # Scale backward probabilities B(v) by beta_b to control their influence; skip if B(v) is -inf.
            b_values = [params.beta_b * trellis_metadata.B_values[v]
                        if params.beta_b != 0 or trellis_metadata.B_values[v] != -math.inf
                        else -math.inf
                        for v in stage_final_vertices]
            if stage > 1145:
                print(f_values)
                print(b_values)
            # Aggregate forward and scaled-backward scores across all vertices emitting the symbol using log-sum-exp.
            estimated_v = np.logaddexp.reduce([trellis_metadata.F_values[v] + params.beta_b * trellis_metadata.B_values[v]
                                               if params.beta_b != 0 or trellis_metadata.B_values[v] != -math.inf
                                               else -math.inf
                                               for v in stage_final_vertices])
        else:
            estimated_v = sum(trellis_metadata.F_values[v] * (trellis_metadata.B_values[v] ** params.beta_b)
                              if not math.isnan(trellis_metadata.B_values[v] ** params.beta_b)
                              else 0
                              for v in stage_final_vertices)
        trellis_metadata.Vk_estimations_per_stage_per_symbol[stage][symbol] = estimated_v



def compute_gamma_coeff_for_trellis_and_symbol(stage, symbol, trellis_metadata, trellises_metadata, params: TrellisBMAParams):
    """
    Gamma represents how much weight to adjust the current F-values (or B-values) based on this symbol.
    Adjusts probability of input symbol.

    Note: we do not average prob accross traces but take whole sum . Here does not matter because we only compare gammas across symbols at each stage.
    Scaling all gammas (for all A/C/G/T) at a stage by a constant factor (e.g., number of traces) does not change the relative ranking.
    """ 
    # First, get this trace's Vk value for this symbol at this stage. Vk is like the confidence score of this symbol from this single trace.
    vk_estimation = trellis_metadata.Vk_estimations_per_stage_per_symbol[stage][symbol]
    #if working with log probabilities branch off ( uses sums not products )
    if USE_LOG_PROB:
        internal_factor = vk_estimation * params.beta_i # Strength of the current trace's opinion, weighted by internal influence
        #Strength of the other traces' opinions: Sum of all Vk's (including this trace). Subtract this trace's Vk (to get "other traces only"). Weight it by external influence.
        external_factor = sum(trellis_j.Vk_estimations_per_stage_per_symbol[stage][symbol]
                              for trellis_j in trellises_metadata)
        external_factor -= vk_estimation
        external_factor = external_factor * params.beta_e
        gamma = internal_factor + external_factor

    else:
        internal_factor = vk_estimation ** params.beta_i
        external_factor = math.prod(trellis_j.Vk_estimations_per_stage_per_symbol[stage][symbol]
                                    for trellis_j in trellises_metadata)
        external_factor /= vk_estimation
        external_factor = external_factor ** params.beta_e
        gamma = internal_factor * external_factor

    return gamma
