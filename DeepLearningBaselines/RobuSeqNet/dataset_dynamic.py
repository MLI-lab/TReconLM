"""
Dynamic dataloader for RobSeqnet that generates data according to the IDS channel. 
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch.utils.data import IterableDataset
from src.data_pkg.data_generation import data_generation
from src.utils.helper_functions import extract_elements
import random
import numpy as np
import random
import torch.distributed as dist


DNA_VOCAB = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
IDX_TO_BASE = {v: k for k, v in DNA_VOCAB.items()}

def decode_prediction(pred_indices):
    """Converts prediction indices into DNA sequence string."""
    return ''.join([IDX_TO_BASE.get(idx, 'N') for idx in pred_indices])  # fallback 'N' for unknown

def one_hot_encode(seq, max_len):
    """
    Each read becomes a tensor of shape [padding_length, 4].
    """
    tensor = torch.zeros((max_len, 4), dtype=torch.float32)
    for idx, base in enumerate(seq):
        if idx >= max_len:
            break
        if base in DNA_VOCAB:
            tensor[idx, DNA_VOCAB[base]] = 1.0
    return tensor

def encode_reads_list(batch_reads_list, padding_length, max_num_reads):
    batch_encoded = []

    for reads in batch_reads_list:
        encoded_reads = []
        for read in reads:
            encoded_read = one_hot_encode(read, padding_length)
            encoded_reads.append(encoded_read)
        
        # Pad with random dummy reads if necessary
        while len(encoded_reads) < max_num_reads:
            dummy_read = torch.zeros(padding_length, 4) # instead of using all A's as dummy reads (original implementation), we use all zero tensor of shape (padding_length, 4) for 4 bases
            encoded_reads.append(dummy_read)

        encoded_reads = torch.stack(encoded_reads, dim=0) # [max_num_reads, padding_length, 4]
        batch_encoded.append(encoded_reads)

    return torch.stack(batch_encoded, dim=0) # [batch_size, max_num_reads, padding_length, 4] or as denoted in Model.py [B, N, L, 4]

def encode_labels_list(labels_list, label_length):
    """
    Each label is a sequence of indices (not one-hot) of shape [batch_size, label_length].
    """
    batch_labels = []
    for label in labels_list:
        label_encoded = [DNA_VOCAB[base] for base in label]
        if len(label_encoded) < label_length:
            label_encoded += [0] * (label_length - len(label_encoded))
        else:
            label_encoded = label_encoded[:label_length]
        batch_labels.append(torch.tensor(label_encoded, dtype=torch.long))
    return torch.stack(batch_labels, dim=0)


class DynamicDataset(IterableDataset):
    """
    IterableDataset for dynamic or infinite data (no indexing or length).
    """
    def __init__(self, batch_size, device, config_params, rng=None, save_generated_data=False, save_dir="generated_data"):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.config_params = config_params
        self.save_generated_data = save_generated_data
        self.save_dir = save_dir
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.rng=rng 

        if self.save_generated_data:
            os.makedirs(self.save_dir, exist_ok=True)

    def __iter__(self):
        idx = 0
        while True:
            if idx % self.world_size == self.rank:
                # Each process generates its own batch of size self.batch_size
                x_batch, y_batch = self.get_batch_dynamic(
                    batch_size=self.batch_size,
                    device=self.device,
                    config_params=self.config_params
                )
                yield x_batch, y_batch
            idx += 1


    def save_data(self, reads, label):
        reads_path = os.path.join(self.save_dir, "reads_dynamic.txt")
        labels_path = os.path.join(self.save_dir, "labels_dynamic.txt")

        # Use append mode so we don’t overwrite on each call
        with open(reads_path, "a") as rf:
            for read in reads:
                rf.write(read + "\n")
            rf.write("=" * 31 + "\n")

        with open(labels_path, "a") as lf:
            lf.write(label + "\n")

    def get_batch_dynamic(self, batch_size, device, config_params):
        data_list = []
        labels_list = []
        num_reads_per_sample = []
        batch_size_counter = 0

        # Check if self.rng is set else fallback to global Python random module
        rng = self.rng if self.rng is not None else random

        while batch_size_counter < batch_size:
            substitution_probability = rng.uniform(config_params['sub_lb'], config_params['sub_ub'])
            insertion_probability = rng.uniform(config_params['ins_lb'], config_params['ins_ub'])
            deletion_probability = rng.uniform(config_params['del_lb'], config_params['del_ub'])

            channel_statistics = {
                'substitution_probability': substitution_probability,
                'insertion_probability': insertion_probability,
                'deletion_probability': deletion_probability
            }

            local_obs_size = rng.randint(config_params['obs_lb'], config_params['obs_ub'])

            data = data_generation(
                data_set_size=1,
                observation_size=local_obs_size,
                length_ground_truth=config_params['gt_length'],
                channel_statistics=channel_statistics,
                target_type=config_params['target_type'],
                data_type=config_params['data_type'], 
                rng=rng
            )

            data_entry = extract_elements(data, config_params['target_type'])[0]

            if isinstance(data_entry, list) and len(data_entry) == 1:
                data_entry = data_entry[0]
            elif not isinstance(data_entry, str):
                raise TypeError(f"[RANK {self.rank}] Unexpected data_entry type: {type(data_entry)}, value: {data_entry}")

            reads_part, label = data_entry.split(":")
            reads = reads_part.split("|")

            #if self.save_generated_data:
            #    self.save_data(reads, label)

            num_reads_per_sample.append(len(reads))
            data_list.append(reads)
            labels_list.append(label)
            batch_size_counter += 1

        max_num_reads = max(num_reads_per_sample)

        x_tensor = encode_reads_list(data_list, config_params['padding_length'], max_num_reads)
        y_tensor = encode_labels_list(labels_list, config_params['label_length'])

        return x_tensor, y_tensor


