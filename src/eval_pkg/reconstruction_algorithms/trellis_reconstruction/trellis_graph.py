import itertools
import logging
import networkx as nx
from typing import NamedTuple, Tuple, List

from . import IO
from .consts import *


class TrellisVertex(NamedTuple):
    # represents current symbol index in string
    stage: int
    # represents current trace index
    sub_stage: int
    indices: Tuple[int, ...]
    symbol: str

    # type: str

    def __str__(self):
        return f"({self.stage}, {self.sub_stage}, {self.indices}, {self.symbol})"

    def __repr__(self):
        return f"({self.stage}, {self.sub_stage}, {self.indices}, {self.symbol})"


def build_new(original, traces, params):
    traces_indices = [list(range(len(t) + 1)) for t in traces]
    indices_combinations = list(itertools.product(*traces_indices))

    stage_origin_out_edges = []
    stage_origin_in_edges = []
    cor_edges = []
    sub_edges = []
    ins_edges = []
    del_edges = []

    for stage in range(len(original)):
    #for stage in range(max(traces_lengths)):
        origin_out_weight = 1 / len(Alphabet)
        if stage == 0:
            origin_indices = tuple([0] * len(traces))
            #stage_initial_indices = tuple([stage] * len(traces))
            origin = TrellisVertex(0, -1, origin_indices, "*")
            #stage_origin = TrellisVertex(stage, -1, stage_initial_indices, "*")
            origin_edges = [(origin, TrellisVertex(0, 0, origin_indices, c), origin_out_weight) for c in Alphabet]
            #stage_origin_edges = [(stage_origin,
            #                       TrellisVertex(stage, 0, stage_initial_indices, c),
            #                       1 / len(Alphabet))
            #                      for c in Alphabet]
            stage_origin_out_edges.extend(origin_edges)
            #stage_origin_out_edges.extend(stage_origin_edges)
        else:
            stage_origin_vertices = [TrellisVertex(stage, -1, indices, "*") for indices in indices_combinations]
            stage_origin_edges = [(v, TrellisVertex(stage, 0, v.indices, c), origin_out_weight)
                                  for v in stage_origin_vertices
                                  for c in Alphabet]
            stage_origin_out_edges.extend(stage_origin_edges)

        stage_final_vertices = [TrellisVertex(stage, len(traces), indices, c)
                                for indices in indices_combinations
                                for c in Alphabet]
        stage_final_edges = [(v, TrellisVertex(stage + 1, -1, v.indices, "*"), 1) for v in stage_final_vertices]
        stage_origin_in_edges.extend(stage_final_edges)

        #stage_final_indices = tuple([stage + 1] * len(traces))
        #next_stage_origin = TrellisVertex(stage + 1, -1, stage_final_indices, "*")
        #stage_final_edges = [(TrellisVertex(stage, len(traces), stage_final_indices, c),
        #                      next_stage_origin,
        #                      1)
        #                     for c in Alphabet]
        #stage_origin_in_edges.extend(stage_final_edges)

        for substage in range(len(traces)):
            # TODO: this will change when adding insertion and deletion
            for indices in indices_combinations:
                # deletion
                for c in Alphabet:
                    current_v = TrellisVertex(stage, substage, indices, c)
                    next_del_v = TrellisVertex(stage, substage + 1, indices, c)
                    del_edges.append((current_v, next_del_v, params.P_DEL))
                if indices[substage] >= len(traces[substage]):
                    # don't go over bounds
                    continue
                for c in Alphabet:
                    current_v = TrellisVertex(stage, substage, indices, c)
                    next_indices = (*indices[:substage], indices[substage] + 1, *indices[substage + 1:])
                    # insertion
                    next_ins_v = TrellisVertex(stage, substage, next_indices, c)
                    ins_edges.append((current_v, next_ins_v, params.P_INS / len(Alphabet)))

                    next_cor_or_sub_v = TrellisVertex(stage, substage + 1, next_indices, c)
                    # correct
                    if traces[substage][indices[substage]] == c:
                        cor_edges.append((current_v, next_cor_or_sub_v, params.P_COR))
                    # substitution
                    else:
                        sub_edges.append((current_v, next_cor_or_sub_v, params.P_SUB / (len(Alphabet)-1)))

    if logging.root.isEnabledFor(logging.DEBUG):
        IO.print_edges(stage_origin_out_edges, "stage_origin_out_edges")
        IO.print_edges(stage_origin_in_edges, "stage_origin_in_edges")
        IO.print_edges(cor_edges, "cor_edges")
        IO.print_edges(sub_edges, "sub_edges")
        IO.print_edges(ins_edges, "ins_edges")
        IO.print_edges(del_edges, "del_edges")

    trellis = nx.DiGraph()
    edges = stage_origin_out_edges + stage_origin_in_edges + del_edges + ins_edges + cor_edges + sub_edges
    trellis.add_weighted_edges_from(edges)
    if REMOVE_UNNEEDED_TRELLIS_SOURCES:
        remove_sources_except_origin(trellis, traces)
    verify_trellis(trellis)
    return trellis
    compute_marginal_prob_for_each_vertex(trellis, traces, original)
    #print("drawing trellis now")
    #nx.draw(trellis)
    #plt.show()


def remove_sources_except_origin(trellis: nx.DiGraph, traces: List[str]):
    has_sources = True
    origin = TrellisVertex(0, -1, tuple([0] * len(traces)), "*")
    while has_sources:
        sources = [v for v, d in trellis.in_degree() if d == 0]
        if len(sources) == 1:
            break
        for source in sources:
            if source != origin:
                logging.debug(f"removing {source}")
                trellis.remove_node(source)


def verify_trellis(trellis: nx.DiGraph):
    if not nx.is_directed_acyclic_graph(trellis):
        raise RuntimeError("trellis is not a DAG!")
    else:
        logging.info("trellis is a DAG")

    # verify graph has a root
    degree_zero_nodes = [n for n, d in trellis.in_degree() if d == 0]
    for node in degree_zero_nodes:
        if len(trellis.edges(node)) == 0:
            logging.warning(f"vertex {node} has no in and out edges! BUG?")
    if len(degree_zero_nodes) > 1:
        logging.warning("trellis has more than one source")
    if not nx.is_weakly_connected(trellis):
        logging.warning("trellis is not weakly connected, so it doesn't have a root!")

    # for v in trellis.nodes:
    #     if len(trellis.edges(v)) == 0:
    #         # ignore absorbing vertices (with no out-edges)
    #         continue
    #     weight_sum = sum(e[2]["weight"] for e in trellis.edges(v, data=True))
    #     # TODO: convert to decimal for precision?
    #     if not math.isclose(weight_sum, 1):
    #         print(f"weight sum for vertex {v} is not 1, but {weight_sum}")
    #         # TODO: remove normalizing?
    #         # normalizing_factor = Decimal(1/weight_sum)
    #         # print("before normalizing:")
    #         # print_edges(trellis.edges(v, data=True))
    #         # new_edges = []
    #         # edges_to_remove = []
    #         # for e in trellis.edges(v, data=True):
    #         #    edges_to_remove.append((e[0], e[1]))
    #         #    new_edges.append((e[0], e[1], e[2]["weight"]*normalizing_factor))
    #         # trellis.remove_edges_from(edges_to_remove)
    #         # trellis.add_weighted_edges_from(new_edges)
    #         # print("after normalizing:")
    #         print_edges(trellis.edges(v, data=True))
