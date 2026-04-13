import math
import numpy as np

import pickle
import requests
import itertools
import psutil
import gc

import sys
import os
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

from ..utils.sys_functions import get_available_memory

def initialize(sequence_type):

    print(sequence_type)
    if sequence_type == 'nuc':
        chars = ['A', 'C', 'G', 'T', '-', '|',':','#']
        meta_file = 'meta_nuc.pkl' # Contains the following: {'vocab_size': 8, 'itos': {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: '-', 5: '|', 6: ':', 7: '#'}, 'stoi': {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4, '|': 5, ':': 6, '#': 7}}

    else:
        raise ValueError(f"Unknown sequence type: {sequence_type}")

    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(os.path.dirname(__file__), meta_file), 'wb') as f:
        pickle.dump(meta, f)

    return chars, vocab_size, stoi, itos, meta

def encode(s, stoi):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers

def decode(l, itos):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

def encode_list(s_list, stoi):
    """
    Converts a list of strings (each string being a sequence like DNA) into a list of lists of integers, using a character-to-index mapping (stoi = "string to index").
    """
    return [[stoi[c] for c in s] for s in s_list]

def decode_list(l_list, itos):
    return [''.join([itos[i] for i in l]) for l in l_list]

def pad_encoded_data(data_ids, length, stoi):
    """
    Pads each encoded sequence in data_ids to a fixed length using the special padding token '#', which is mapped to an integer via stoi['#'].
    Input:
        data_ids: a list of lists of integers
        length: the desired final sequence length (int)
        stoi: dictionary mapping characters to token IDs
    """

    data_ids = [list(itertools.chain(elem, itertools.repeat(stoi['#'], length - len(elem)))) for elem in data_ids]
    return data_ids

def encode_and_pad(data_list, name, stoi, block_size, out_dir):
    """
    Encodes a list of DNA sequences, pads them to block_size, creates input/target tensors,
    and saves them to disk as {name}_x.pt and {name}_y.pt.

    Input:
        data_list: list of strings (DNA sequences in format "read1|read2|...:ground_truth")
        name: prefix for output files (e.g., "test", "train")
        stoi: dictionary mapping characters to token IDs
        block_size: the desired sequence length (int)
        out_dir: directory to save the .pt files
    """
    # Encode and pad
    encoded_data = encode_list(data_list, stoi)
    padded_encoded_data = pad_encoded_data(encoded_data, block_size, stoi)

    # Verify padding
    lengths = [len(seq) for seq in padded_encoded_data]
    unique_lengths = set(lengths)
    assert len(unique_lengths) == 1 and list(unique_lengths)[0] == block_size, \
        f"Expected all sequences to have length {block_size}, but found lengths: {unique_lengths}"

    # Convert to numpy array
    np_data = np.array(padded_encoded_data, dtype=np.int64)

    # Create input (x) and target (y) by shifting
    x_temp = np_data[:, 0:block_size-1]
    y_temp = np_data[:, 1:block_size]

    # Convert to torch tensors
    x = torch.from_numpy(x_temp.astype(np.int64))
    y = torch.from_numpy(y_temp.astype(np.int64))

    # Save to disk
    x_path = os.path.join(out_dir, f'{name}_x.pt')
    y_path = os.path.join(out_dir, f'{name}_y.pt')
    torch.save(x, x_path)
    torch.save(y, y_path)

    print(f"Saved {x.shape[0]} examples to {x_path} and {y_path}")
    print(f"Tensor shapes: x={x.shape}, y={y.shape}")

    return x, y


if __name__ == "__main__":

    print('prepare.py')

    initialize('nuc')