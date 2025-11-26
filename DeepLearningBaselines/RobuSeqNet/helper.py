"""
Renamed due to name conflict with utilis folder in TreconLM/src, was `utils.py` in the original implementation: https://github.com/qinyunnn/RobuSeqNet

"""

import torch
import random

def statistics(x, y):
    #print(f"[DEBUG] x shape (predictions): {x.shape}")
    #print(f"[DEBUG] y shape (ground truth): {y.shape}")

    z = x.eq(y).int()  # Element-wise comparison: shape [batch_size, seq_length]
    #print(f"[DEBUG] z shape (matches): {z.shape}")
    #print(f"[DEBUG] z sample:\n{z[:2]}")  # show a few samples

    correct_per_sample = torch.sum(z, dim=1)  # shape: [batch_size]
    #print(f"[DEBUG] correct per sample: {correct_per_sample}")

    total_correct = torch.sum(correct_per_sample)  # scalar
    #print(f"[DEBUG] total correct predictions: {total_correct.item()}")

    total_positions = x.numel()  # total number of predictions = batch_size × seq_length
    #print(f"[DEBUG] total positions (batch_size × seq_length): {total_positions}")

    accuracy = total_correct.float() / total_positions  # Normalize to get per-base accuracy
    #print(f"[DEBUG] accuracy: {accuracy.item()}")

    return accuracy

def is_None(batch):
    new_batch=[]
    for b in batch:
        feature, label=b
        if (feature is not None) and (label is not None):
            new_batch.append((feature,label))
    return new_batch


def getmaxlen(root_dir):
    max_len=0
    with open(root_dir, 'r+') as f:
        readline=f.readlines()
        for i, line in enumerate(readline):
            line=line.strip('\n')
            max_len=max(len(line), max_len)

    return max_len


def group_shuffle(data):
    tensor_data = torch.tensor(data, dtype=torch.long)
    ten, dices = torch.sort(tensor_data, descending=False)
    list1=list(ten)
    list2=list(dices)
    list3 = [list2[:1]]
    [list3[-1].append(t) if x == y else list3.append([t]) for x, y, s, t in zip(list1[:-1], list1[1:], list2[:-1], list2[1:])]
    #random.shuffle(list3)
    list4=[]
    for i in range(len(list3)):
        random.shuffle(list3[i])
        list4.extend(list3[i])
    indices = torch.tensor(list4, dtype=torch.long)

    return indices