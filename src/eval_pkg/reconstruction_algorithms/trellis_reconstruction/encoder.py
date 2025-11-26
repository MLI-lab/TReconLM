from consts import *

from enum import IntEnum
from scipy.stats import rv_discrete
import logging
import random


class EncoderOperation(IntEnum):
    DEL = 0
    SUB = 1
    INS = 2
    COR = 3


PROBABILITIES = [P_DEL, P_SUB, P_INS, P_COR]
encoder_RV = rv_discrete(name="encoder_RV", values=(range(len(PROBABILITIES)), PROBABILITIES))


def create_noisy_trace(s):
    trace = []
    for c in s:
        # insertion loop
        is_insert = True
        while is_insert:
            operation = encoder_RV.rvs()
            if operation == EncoderOperation.COR:
                trace.append(c)
                is_insert = False
                logging.debug("cor")
            elif operation == EncoderOperation.DEL:
                is_insert = False
                logging.debug("del")
            elif operation == EncoderOperation.SUB:
                options = Alphabet.replace(c, "")
                new_char = random.choice(options)
                trace.append(new_char)
                is_insert = False
                logging.debug("sub")
            elif operation == EncoderOperation.INS:
                new_char = random.choice(Alphabet)
                trace.append(new_char)
                logging.debug("ins")
                # stay in loop

    return "".join(trace)
