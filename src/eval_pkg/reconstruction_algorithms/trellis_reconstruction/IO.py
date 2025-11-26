import logging


def dbg_print(s):
    if False:
        print(s)


def print_edges(edges, name=""):
    if not logging.root.isEnabledFor(logging.DEBUG):
        return
    if name:
        logging.debug(f"{name}:")
    for e in edges:
        logging.debug(f"{e[0]})->{e[1]}) ({e[2]})")