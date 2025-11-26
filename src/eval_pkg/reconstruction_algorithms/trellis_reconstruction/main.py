import sys

from consts import *
import encoder
import algorithms.multi_trace
import algorithms.trellis_bma
from algorithms.trellis_bma import TrellisBMAParams
import random

import logging
import argparse
import time
import re
from typing import NamedTuple, List
from collections import Counter
from enum import Enum
import matplotlib.pyplot as plt


class AlgorithmType(Enum):
    MultiTrace = 1
    TrellisBMA = 2


class ResultEntry(NamedTuple):
    index: int
    original: str
    estimate: str
    hamming: int
    levenstein: int


class ResultData(NamedTuple):
    filename: str
    trace_num: int
    algorithm: AlgorithmType
    results: List[ResultEntry]
    avg_hamming: float
    normalized_hamming: float
    avg_levenstein: float
    normalized_levenstein: float


def read_centers_and_clusters():
    centers = []
    with open("Centers.txt") as f:
        centers = f.read().splitlines()

    with open("Clusters.txt") as f:
        clusters = [[c for c in cluster.splitlines() if c != ""]
                    for cluster in f.read().split("===============================")
                    if cluster != ""]

    return centers, clusters


def parse_results_file(results_file):
    with open(results_file) as f:
        data = f.readlines()
    stripped_data = [[s.strip() for s in line.split(",")] for line in data]
    results = [ResultEntry(int(r[0]), r[4], r[3], int(r[1]), int(r[2])) for r in stripped_data]
    return results


def print_results(results):
    for entry in results:
        print(f"{entry.index}:")
        print(f"hamming: {entry.hamming}, levenstein: {entry.levenstein}")
        print(f"original: {entry.original}")
        print(f"estimate: {entry.estimate}")
        print("----")


if __name__ == '__main__':
    logging.basicConfig(level=LOGGING_LEVEL)
    parser = argparse.ArgumentParser(description='Reconstruct DNA string from noisy traces using trellis.')
    main_options = parser.add_mutually_exclusive_group()
    main_options.add_argument("-p", "--parse", action="store_true", help="parse the input file instead of performing reconstruction")
    main_options.add_argument("-r", "--reconstruct", action="store_true", help="reconstruct DNA string from traces")

    parser.add_argument("--algorithm", default="trellis-bma", choices=["trellis-bma", "multi-trace"], help="Algorithm to be used for trace reconstruction")
    parser.add_argument("--trace-num", type=int, help="Number of traces to use for each cluster. "
                                                      "if a cluster has less than this number, it will be ignored")
    parser.add_argument("--from-idx", type=int, help="index of clusters in dataset to start from")
    parser.add_argument("--to-idx", type=int, help="index of clusters in dataset to finish (non inclusive)")
    # TODO: make the defaults a function of trace num, to be the optimized hyperparams from the article
    parser.add_argument("--beta-b", type=float, default=DEFAULT_BETA_B, help="weight for backward values in trellis BMA")
    parser.add_argument("--beta-i", type=float, default=DEFAULT_BETA_I, help="weight for the current trace in trellis BMA")
    parser.add_argument("--beta-e", type=float, default=DEFAULT_BETA_E, help="weight for the other traces in trellis BMA")
    parser.add_argument("--results-file", type=str, help="write the reconstruction results to this file")
    parser.add_argument("--input-results-file", type=str, nargs='+',
                        help="when --parse is used, parse this results file/s.")
    parser.add_argument("-wh", "--worst-n-hamming", type=int, help="output the worst N reconstructions by hamming distance")
    parser.add_argument("-wl", "--worst-n-levenstein", type=int, help="output the worst N reconstructions by levenstein distance")
    parser.add_argument("-eh", "--error-histogram", action="store_true", help="plot hamming and levenstein error histograms")
    args = parser.parse_args()

    if not args.results_file:
        current_datetime = time.strftime("%Y%m%d-%H_%M_%S")
        algorithm_str = args.algorithm.replace("-", "_")
        args.results_file = f"results_{args.trace_num}_traces_{algorithm_str}_{current_datetime}.txt"

    if args.parse:
        if not args.input_results_file:
            print("missing input results file!")
            sys.exit(1)
        results_by_file = dict()
        for res_file in args.input_results_file:
            results_by_file[res_file] = parse_results_file(res_file)

        for file, results in results_by_file.items():
            print(f"file: {file}")
            # general info
            hammings = [r.hamming for r in results]
            levensteins = [r.levenstein for r in results]
            avg_hamming = sum(hammings) / len(hammings)
            avg_levenstein = sum(levensteins) / len(levensteins)
            normalized_hamming = sum([r.hamming/len(r.original) for r in results]) / len(results)
            normalized_levenstein = sum([r.levenstein / len(r.original) for r in results]) / len(results)

            # parse algorithm type
            if "multi_trace" in file:
                algorithm = AlgorithmType.MultiTrace
            else:
                # Default type is Trellis BMA for convenience
                algorithm = AlgorithmType.TrellisBMA
            # parse trace num
            trace_num = int(re.search("results_(\\d+)_traces", file).group(1))

            results_data = ResultData(file, trace_num, algorithm, results, avg_hamming, normalized_hamming, avg_levenstein, normalized_levenstein)
            results_by_file[file] = results_data

            print(f"trace num: {trace_num}")
            print(f"algorithm type: {algorithm.name}")

            print(f"read {len(results)} samples:")
            print(f"Normalized hamming: {normalized_hamming}, avg hamming: {avg_hamming}")
            print(f"Normalized levenstein: {normalized_levenstein}, avg levenstein: {avg_levenstein}")

        if len(results_by_file) > 1:
            # draw comparison
            trellis_bma_trace_nums = [res.trace_num for res in results_by_file.values()
                                      if res.algorithm == AlgorithmType.TrellisBMA]
            trellis_bma_normalized_hammings = [res.normalized_hamming for res in results_by_file.values()
                                               if res.algorithm == AlgorithmType.TrellisBMA]
            trellis_bma_normalized_levensteins = [res.normalized_levenstein for res in results_by_file.values()
                                                  if res.algorithm == AlgorithmType.TrellisBMA]
            multi_trace_trace_nums = [res.trace_num for res in results_by_file.values()
                                      if res.algorithm == AlgorithmType.MultiTrace]
            multi_trace_normalized_hammings = [res.normalized_hamming for res in results_by_file.values()
                                               if res.algorithm == AlgorithmType.MultiTrace]
            multi_trace_normalized_levensteins = [res.normalized_levenstein for res in results_by_file.values()
                                                  if res.algorithm == AlgorithmType.MultiTrace]

            fig, (sub1, sub2) = plt.subplots(2)
            sub1.plot(trellis_bma_trace_nums, trellis_bma_normalized_hammings, "ro-", label="Trellis BMA")
            sub1.plot(multi_trace_trace_nums, multi_trace_normalized_hammings, "bP", label="Multi Trace")
            sub1.set_ylabel("Normalized Hamming")
            sub1.set_xlabel("Trace Number")
            sub1.legend()

            for x, y in zip(trellis_bma_trace_nums, trellis_bma_normalized_hammings):
                sub1.annotate(f"{y:.4f}", xy=(x+0.25, y))
            for x, y in zip(multi_trace_trace_nums, multi_trace_normalized_hammings):
                sub1.annotate(f"{y:.4f}", xy=(x+0.25, y))

            sub2.plot(trellis_bma_trace_nums, trellis_bma_normalized_levensteins, "ro-", label="Trellis BMA")
            sub2.plot(multi_trace_trace_nums, multi_trace_normalized_levensteins, "bP", label="Multi Trace")
            sub2.set_ylabel("Normalized Levenstein")
            sub2.set_xlabel("Trace Number")
            sub2.legend()

            for x, y in zip(trellis_bma_trace_nums, trellis_bma_normalized_levensteins):
                sub2.annotate(f"{y:.4f}", xy=(x+0.25, y))
            for x, y in zip(multi_trace_trace_nums, multi_trace_normalized_levensteins):
                sub2.annotate(f"{y:.4f}", xy=(x+0.25, y))
            plt.tight_layout()
            sub1.grid()
            sub2.grid()
            plt.show()

        for file, result_data in results_by_file.items():
            results = result_data.results
            print(f"File: {file}")
            if args.worst_n_hamming:
                results.sort(key=lambda res: res.hamming, reverse=True)
                print(f"worst {args.worst_n_hamming} cases by hamming distance:")
                print_results(results[:args.worst_n_hamming])
            if args.worst_n_levenstein:
                results.sort(key=lambda res: res.levenstein, reverse=True)
                print(f"worst {args.worst_n_levenstein} cases by levenstein distance:")
                print_results(results[:args.worst_n_levenstein])
            if args.error_histogram:
                hammings_counter = Counter(hammings)
                levensteins_counter = Counter(levensteins)

                fig, (sub1, sub2) = plt.subplots(2)

                sub1.hist(hammings, bins=max(hammings))
                sub1.set_ylabel("count")
                sub1.set_xlabel("hamming distance")
                sub2.hist(levensteins, bins=max(levensteins))
                sub2.set_ylabel("count")
                sub2.set_xlabel("levenstein distance")
                #plt.hist(levensteins)
                #plt.bar(levensteins_counter.keys(), levensteins_counter.values())
                #fig.show()
                plt.tight_layout()
                sub1.grid()
                sub2.grid()
                plt.show()
    elif args.reconstruct:
        if args.algorithm == "trellis-bma":
            print("Algorithm: trellis BMA")
            trellis_bma_params = TrellisBMAParams(beta_b=args.beta_b, beta_i=args.beta_i, beta_e=args.beta_e)
        elif args.algorithm == "multi-trace":
            print("Algorithm: multi trace")
        if USE_NANOPORE_DATA_FROM_FILE:
            centers, clusters = read_centers_and_clusters()
            results = []
            total_processed = 0
            with open(args.results_file, "w") as f:
                for idx, (original, traces) in enumerate(zip(centers[args.from_idx: args.to_idx], clusters[args.from_idx: args.to_idx])):
                    if len(traces) < args.trace_num:
                        logging.info(f"skipping cluster {args.from_idx+idx}, has {len(traces)} traces and {args.trace_num} required")
                        continue
                    chosen_traces = random.sample(traces, args.trace_num)
                    logging.info(f"original: {original}")
                    logging.info("traces:")
                    logging.info(",\n".join(chosen_traces))
                    total_processed += 1

                    if args.algorithm == "trellis-bma":
                        result = algorithms.trellis_bma.compute_trellis_bma_estimation(chosen_traces, original, trellis_bma_params)
                    elif args.algorithm == "multi-trace":
                        result = algorithms.multi_trace.compute_multi_trace_estimation(chosen_traces, original)
                    else:
                        print("No valid algorithm given")
                        sys.exit(0)
                    results.append(result)
                    print(result)
                    sys.exit()
                    original, estimate, hamm, levenstein = result
                    f.write(f"{args.from_idx+idx}, {hamm}, {levenstein}, {estimate}, {original}\n")
                    f.flush()

        else:
            if USE_CUSTOM_TRACES:
                original = CUSTOM_ORIGINAL
                traces = CUSTOM_TRACES
            else:
                original = ORIGINAL
                traces = [encoder.create_noisy_trace(ORIGINAL) for i in range(TRACE_NUM)]
            logging.info(traces)
            #trellis_graph = trellis_graph.build_new(original, traces)
            #algorithms.multi_trace.compute_marginal_prob_for_each_vertex(trellis_graph, traces, original)
            algorithms.trellis_bma.compute_trellis_bma_estimation(traces, original, trellis_bma_params)
    else:
        print("must choose args or reconstruct!")
