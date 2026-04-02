import sys

#from consts import *
#import encoder
import algorithms.multi_trace
import algorithms.trellis_bma
from algorithms.trellis_bma import TrellisBMAParams
#import random

#import logging
#import argparse
#import time
#import re
#from typing import NamedTuple, List
#from collections import Counter
#from enum import Enum
#import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    algorithm = "trellis-bma" # trellis-bma or multi-trace

    beta_b = 0.1
    beta_i = 0
    beta_e = 1

    original = "ACCATAATGCGTGGGGCCGACCTCGGAATGCGGTCTCCATGCGCGTTTCCTCCAACCTAAGGTAGCCTGTAGTTCATTGGACCTCTGATGGCGCTTATAGAAACCGGGAA"

    chosen_traces = ["ACCATAATGCGTGGGGCTGACCTCGGAATGCGTGGTCTCCATGCGCGTTTCCTCCAACCTAAGGTAGCCTGTAGTTCATTGACCTCTGATGGCGCTTATAGAAACCGGGAA",
                 "ACCATAATGCGTGGGGCCGACCTCGGAATGCGGTCTCCATGCGCGTTTCCTCAACCTAAGGTAGCCTGATTCATTGGACCTCTGATGGCGCTTATAGAAACTGGGGAA"
                 ]

    
    if algorithm == "trellis-bma":
        print("Algorithm: trellis BMA")
        trellis_bma_params = TrellisBMAParams(beta_b=beta_b, beta_i=beta_i, beta_e=beta_e)
    elif algorithm == "multi-trace":
        print("Algorithm: multi trace")

    if algorithm == "trellis-bma":
        #result = algorithms.trellis_bma.compute_trellis_bma_estimation(chosen_traces, original, trellis_bma_params)
        original, final_estimate_str, hamm, levenstein = algorithms.trellis_bma.compute_trellis_bma_estimation(chosen_traces, original, trellis_bma_params)
    elif algorithm == "multi-trace":
        result = algorithms.multi_trace.compute_multi_trace_estimation(chosen_traces, original)
    else:
        print("No valid algorithm given")
        sys.exit(0)
                    
    #print(result)

    #original, final_estimate_str, hamm, levenstein

    print("Original: ", original)
    print("Final estimate: ", final_estimate_str)   
    print("Hamming distance: ", hamm)
    print("Levenstein distance: ", levenstein)
    