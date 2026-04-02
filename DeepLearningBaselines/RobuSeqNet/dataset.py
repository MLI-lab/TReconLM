"""
dataset.py

Modified and extended version of the original RobuSeqNet dataset loader: https://github.com/qinyunnn/RobuSeqNet/blob/master/dataset.py.

Changes from the original:
- One-hot encoding is done manually with torch tensors (no sklearn dependency)
- Read clusters are dynamically subsampled (≤10 reads) per sample

Used for training and finetuning RobuSeqNet on fixed datasets with on-the-fly augmentation.
"""


import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np
import random
from collections import defaultdict
from helper import is_None, getmaxlen, group_shuffle

import torch
from torch.utils.data import Dataset

Separator = '==============================='

class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.x_list, self.y_data = self.load_data_wrapper()

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        X = self.x_list[index]
        Y = self.y_data[index]
        return X, Y

    def load_data_wrapper(self):
        with open(self.root_dir, 'r') as f1, open(self.label_dir, 'r') as f2:
            x_data = []
            y_data = []
            x_list = []
            f1_r = f1.readlines()
            f2_r = f2.readlines()
            id_list = []
            id = 0
            for x_line in f1_r:
                x_line = x_line.strip('\n')
                if x_line != Separator:
                    x_data.append(''.join(x_line))
                elif x_line == Separator and x_data != []:
                    x_list.append(x_data)
                    x_data = []
                    id += 1
                elif x_line == Separator and x_data == []:
                    id_list.append(id)
                    id += 1
            for j, y_line in enumerate(f2_r):
                if j + 1 not in id_list:
                    y_line = y_line.strip('\n')
                    y_data.append(''.join(y_line))

        return x_list, y_data





class CustomSampler(Sampler):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        nums=[]
        for i in range(len(self.data)):
            num=len(self.data[i][0])
            nums.append(num)
        indices = group_shuffle(nums)
        return iter(indices)

    def __len__(self):
        return len(self.data)



class CustomBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

            if (
                    i < len(sampler_list) - 1
                    and len(self.sampler.data[idx][0])
                    != len(self.sampler.data[sampler_list[i + 1]][0])
            ):
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                    batch = []
                #else:
                    #batch = []
            i += 1

        if len(batch) > 0 and not self.drop_last:
            yield batch



    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

DNA_VOCAB = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def one_hot_encode(seq, max_len):
    tensor = torch.zeros((max_len, 4), dtype=torch.float32)
    for idx, base in enumerate(seq):
        if idx >= max_len:
            break
        if base in DNA_VOCAB:
            tensor[idx, DNA_VOCAB[base]] = 1.0
    return tensor


class collater:
    def __init__(self, padding_length, label_length):
        self.padding_length = padding_length
        self.label_length   = label_length

    def __call__(self, batch):
        # batch is a list of (x_item, y_item);
        # x_item is a Python list of raw reads, e.g. ["ACGT", "TGCA", ...].
        fea_batch   = []
        label_batch = []

        # First, for each sample, pick your ≤10 reads.
        all_selected_reads = []
        for reads, lbl in batch:
            if len(reads) > 10:
                k = random.randint(2, 10)
                selected = random.sample(reads, k)
            else:
                selected = list(reads)
            all_selected_reads.append(selected)
            label_batch.append(lbl)

        # Now compute how many rows we need in this batch (all ≤10).
        max_num_reads = max(len(sel) for sel in all_selected_reads)

        # Encode and pad each “selected” list up to max_num_reads:
        all_features = []
        for selected_reads in all_selected_reads:
            encoded_reads = []
            for read in selected_reads:
                encoded = one_hot_encode(read, self.padding_length)
                encoded_reads.append(encoded)

            # Pad with dummy reads until we reach max_num_reads
            while len(encoded_reads) < max_num_reads:
                dummy_read = torch.zeros(self.padding_length, 4)

                encoded_reads.append(dummy_read)

            # Now we have exactly [max_num_reads, padding_length, 4]
            encoded_reads = torch.stack(encoded_reads, dim=0)
            all_features.append(encoded_reads)

        # Stack across the batch -> [batch_size, max_num_reads, padding_length, 4]
        feature_tensor = torch.stack(all_features, dim=0)

        # Build label tensor the same way as before:
        label_tensors = []
        for lbl in label_batch:
            lbl_enc = torch.zeros(self.label_length, dtype=torch.long)
            for i, base in enumerate(lbl):
                if i >= self.label_length:
                    break
                if base in DNA_VOCAB:
                    lbl_enc[i] = DNA_VOCAB[base]
                else:
                    lbl_enc[i] = 0
            label_tensors.append(lbl_enc)
        label_tensor = torch.stack(label_tensors, dim=0)

        return feature_tensor, label_tensor

