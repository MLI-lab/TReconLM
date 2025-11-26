"""
Adjusted data loader for dynamic dataset generation according to IDS channel. 
And for downloading an artifact from W&B that is in format read_1|read_2 etc. and reformating to fit input for DNAformer. 
"""

import random
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader, DistributedSampler, Dataset
import torch.nn.functional as F
from tqdm import tqdm



import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_pkg.data_generation import data_generation
from src.utils.helper_functions import extract_elements
from src.utils.data_functions import load_data_from_file


from torch.utils.data import Dataset
import random

class FileDNAData(Dataset):
    def __init__(self, data_dir, config, mode, fixed_seed=0):  
        if mode =='train': 
            filename= config.filename_train
            self.rng = random  # use global Python RNG 
  
        else:
            filename= config.filename_val
            # set fixed seed
            self.rng = random.Random(fixed_seed)                
        # load data once
        self.data   = load_data_from_file(os.path.join(data_dir, filename)) # list of data examples
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        noisy_part, gt_part = line.split(":", 1)
        reads = [r for r in noisy_part.split("|") if r]
        # subsample if too many
        if len(reads) > self.config.max_number_per_cluster:
            reads = self.rng.sample(reads, self.config.max_number_per_cluster)

        ground_truth = gt_part.strip()
        data = {"noisy_copies": reads, "data": ground_truth}

        model_input, model_input_right, noisy_copy_length, num_noisy_copies = DNAformerData.grab_model_input(self, data)
        noisy_copy_length = torch.tensor(noisy_copy_length)
        
        label = one_hot_encoding(ground_truth).contiguous().int()

        num_noisy_copies = len(data['noisy_copies'])
        if num_noisy_copies<self.config.max_number_per_cluster:
            noisy_copies = data['noisy_copies'].copy()
            for idy in range(self.config.max_number_per_cluster- num_noisy_copies):
                noisy_copies.append('none')
        else:
            noisy_copies = data['noisy_copies'][:self.config.max_number_per_cluster]

        return {
            "model_input": model_input,
            "model_input_right": model_input_right,
            "noisy_copies": noisy_copies,
            "num_noisy_copies": num_noisy_copies,
            "noisy_copy_length": noisy_copy_length,
            "label": label,
            "false_cluster": False,
            "index": idx,
            "cluster_path": self.config.data_folder
        }



class PrecomputedDNAData(Dataset):
    """
    Wraps lists of (noisy_reads_list, ground_truth_str) and
    applies grab_model_input + one_hot_encoding for DNAformerData.
    """
    def __init__(self, x_test, ground_truths, config, meta):
        assert len(x_test) == len(ground_truths)

        self.pairs = list(zip(x_test, ground_truths))
        self.config = config
        # reuse the same helper methods from DNAformerData
        self._dummy = DNAformerData(config, mode="val_fixed")
        self.meta=meta

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        data = {}
        # get the raw tensor + the GT string 
        reads_tensor, gt_string = self.pairs[idx]
        stoi, itos = self.meta['stoi'], self.meta['itos'] # stoi converts characters into token IDs and itos vice versa (token id just from 0 to 7)
        decode = lambda l: ''.join([itos[i] for i in l])

        # decode 
        full_seq = decode(reads_tensor.tolist())
        full_seq = full_seq.split('#', 1)[0]

        # split off noisy reads vs. ground truth
        noisy_part, gt_part = full_seq.split(':', 1)
        # noisy reads on '|' and drop any empty strings
        noisy_reads = [r for r in noisy_part.split('|') if r]
        ground_truth = gt_part.strip()
        
        data = {
            'noisy_copies': noisy_reads,
            'data':         ground_truth
        }
        # get model inputs
        model_input, model_input_right, noisy_copy_length, num_noisy_copies  = self._dummy.grab_model_input(data)
        noisy_copy_length = torch.tensor(noisy_copy_length)

        # build the one-hot label
        label = one_hot_encoding(ground_truth).contiguous().int()

        # Get noisy copies
        # when we create the list of noisy copies (noisy_copies), we pad it by adding 'none' strings if the real cluster has fewer copies than max_number_per_clust
        # Note the model input is already 'padded' by adding all zeros for none noisy copies, so not sure why we need this here but kept so it is consistent with DNAformer dataloader
        num_noisy_copies = len(data['noisy_copies'])
        if num_noisy_copies<self.config.max_number_per_cluster:
            noisy_copies = data['noisy_copies'].copy()
            for idy in range(self.config.max_number_per_cluster- num_noisy_copies):
                noisy_copies.append('none')
        else:
            noisy_copies = data['noisy_copies'][:self.config.max_number_per_cluster]

        return {
            'model_input':      model_input,
            'model_input_right': model_input_right,
            'noisy_copies':     noisy_copies,
            'num_noisy_copies': num_noisy_copies,
            'noisy_copy_length': noisy_copy_length,
            'label':            label,
            'false_cluster':    False,
            'index':            idx,
            'cluster_path':     'downloaded_artifact'
        }


class DNAformerData(IterableDataset):
    """
    Dataset that generates or preloads samples in train or validation modes.
    Mode 'val_fixed' precomputes a fixed-size validation set with a seeded RNG.
    """
    def __init__(self, config, mode="train", fixed_seed=0):
        self.config = config
        self.mode = mode
        if mode == "val_fixed":
            self.rng = random.Random(fixed_seed)         
            # Pre-generate a reproducible fixed validation set
            self.data   = [self.generate_data() for _ in range(config.val_dataset_size)]
        elif mode == "train":
            # Streaming training: use global torch RNG
            self.rng    = random
        else:
            raise ValueError(f"Unsupported mode: {mode}")


    def __iter__(self):
        if self.mode == "val_fixed":
            # finite, precomputed list
            yield from self.data
        else:
            # infinite stream
            while True:
                yield self.generate_data()

    def __len__(self):
        if self.mode == "val_fixed":
            return len(self.data)
        raise TypeError(f"__len__ is undefined for mode '{self.mode}' (streaming)")

    def __getitem__(self, idx):
        if self.mode == "val_fixed":
            return self.data[idx]
        raise TypeError(f"__getitem__ is undefined for mode '{self.mode}' (streaming)")


    def generate_data(self):

        sub_p = self.rng.uniform(
            self.config.substitution_probability_lb,
            self.config.substitution_probability_ub
        )
        ins_p = self.rng.uniform(
            self.config.insertion_probability_lb,
            self.config.insertion_probability_ub
        )
        del_p = self.rng.uniform(
            self.config.deletion_probability_lb,
            self.config.deletion_probability_ub
        )

        channel_statistics = {
            'substitution_probability': sub_p,
            'insertion_probability':    ins_p,
            'deletion_probability':     del_p
        }

        # sample local_obs_size as an integer
        local_obs_size = self.rng.randint(
            self.config.min_number_per_cluster,
            self.config.max_number_per_cluster
        )

        rng=self.rng
        target_type='CPRED'

        
        samples = data_generation(
            data_set_size=1,
            observation_size=local_obs_size,
            length_ground_truth=self.config.label_length,
            channel_statistics=channel_statistics,
            target_type= target_type,
            data_type='ids_data', 
            rng=rng

        )
        # Get train instance as in Equation (2) in paper 
        samples = extract_elements(samples, target_type)[0]
        samples = samples[0]

        # split data by : to get the ground truth 
        noisy_reads, ground_truth = samples.split(':')
        ground_truth = ground_truth.strip()
        # spilit data  by | to get the noisy sequences 
        noisy_reads = noisy_reads.split('|')

        data = {
            'noisy_copies': noisy_reads,
            'data':         ground_truth
        }
        
        # Generate false copies in cluster 
        # We do not do this here
        #data['noisy_copies'] = get_false_copies(self.config, data['noisy_copies'])

        # Get model input (one_hot encoding)
        model_input, model_input_right, noisy_copy_length, num_noisy_copies = self.grab_model_input(data)
        noisy_copy_length = torch.tensor(noisy_copy_length)

        # Torch label
        label = one_hot_encoding(ground_truth).contiguous().int()

        # Place holders
        false_cluster = False
        index         = 'None'
        cluster_path  = 'None' # No path on disk because we generated the data randomly, not read from a file.
        
        # Get noisy copies
        # when we create the list of noisy copies (noisy_copies), we pad it by adding 'none' strings if the real cluster has fewer copies than max_number_per_clust
        # but noisy copies not actually used and we padd in grap input with zeros for empty reads so i am not sure why they do this in the original implementation (keeping this here for consistency with original implementation)
        num_noisy_copies = len(data['noisy_copies'])
        if num_noisy_copies<self.config.max_number_per_cluster:
            noisy_copies = data['noisy_copies'].copy()
            for idy in range(self.config.max_number_per_cluster- num_noisy_copies):
                noisy_copies.append('none')
        else:
            noisy_copies = data['noisy_copies'][:self.config.max_number_per_cluster]
                
        # Build sample
        sample = {'model_input':model_input,
                'model_input_right':model_input_right,
                'noisy_copies':noisy_copies, 
                'num_noisy_copies':num_noisy_copies,
                'noisy_copy_length':noisy_copy_length,
                'label':label,
                'false_cluster':false_cluster,
                'index':index,
                'cluster_path':cluster_path}
        
        return sample

    def grab_model_input(self, data):
        
        noisy_copies = data['noisy_copies']
        num_noisy_copies = len(data['noisy_copies'])
        
        #  Initialize empty tensors 
        # model_input will store the normal copies (as one-hot tensors)
        # model_input_right will store the flipped (reversed) copies also one hot encoded 
        # get two tensors already fully filled with zeros of shape (max_number_per_cluster,4,length).
        # the lengths we will fill with the real lengths of the copies or if no is available anymore with length 0 
        model_input      = torch.zeros([self.config.max_number_per_cluster,4,self.config.noisy_copies_length])
        model_input_right = torch.zeros([self.config.max_number_per_cluster,4,self.config.noisy_copies_length])
        
        noisy_copy_length = []
        
        for idx in range(self.config.max_number_per_cluster):
            
            if idx < len(noisy_copies):
                noisy_copy = noisy_copies[idx]
                            
                # Update copies length list
                noisy_copy_length.append(len(noisy_copy))
                
                # Get one-hot embedding
                one_hot_noisy_copy = one_hot_encoding(noisy_copy)
                
                # Get flipped copy
                one_hot_noisy_copy_right = torch.flip(one_hot_noisy_copy,dims=[1])
                
                # Padding
                #  If the copy is shorter than noisy_copies_length, you pad it with zeros. 
                if one_hot_noisy_copy.shape[-1] < self.config.noisy_copies_length:    
                    model_input[idx,:,:]      = pad_seq(self.config, one_hot_noisy_copy)
                    model_input_right[idx,:,:] = pad_seq(self.config, one_hot_noisy_copy_right)

            else:
                noisy_copy_length.append(0) # zero means no base present in one hot encoding
            
        return model_input, model_input_right, noisy_copy_length, num_noisy_copies


def collate_dna(batch, siamese=True):

    out = {}
    for key in batch[0]:
        # Collects that field (key) from all items in the batch into a list.
        vals = [d[key] for d in batch]
        if isinstance(vals[0], torch.Tensor):
            # If the field contains tensors (same shape), it stacks them along a new dimension (batch dim)
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    if siamese:
        # check if model_input_right is present 
        assert 'model_input_right' in out, "Missing right inputs for siamese mode"
    return out


def make_loader(config, mode="train", batch_size=128, num_workers=4,):
    # if config.data_folder exists and is non‐empty, use FileDNAData
    if hasattr(config, "data_folder") and config.data_folder:
        print(f"Loading data from {config.data_folder}")
        ds = FileDNAData(config.data_folder, config,mode)
        sampler = DistributedSampler(ds, shuffle=True, seed=config.train_seed)
        shuffle = False
    else:
        # fall back to on-the-fly generation
        rank = dist.get_rank() if dist.is_initialized() else 0
        ds = DNAformerData(config, mode)
        shuffle = False
        sampler=None

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=(mode != "train"),
        pin_memory=True,
        collate_fn=lambda batch: collate_dna(
            batch, siamese=(config.model_config == "siamese")
        )
    )



def one_hot_encoding(read):
         
    read = torch.tensor([ord(c) for c in read])

    read[read==65]=0 # A
    read[read==67]=1 # C
    read[read==71]=2 # G
    read[read==84]=3 # T

    return F.one_hot(read,4).transpose(0,1)

def pad_seq(config, one_hot_noisy_copy):
    """
    Pads (with zeros) a one-hot encoded DNA sequence to a fixed length. 
    """
    
    pad = config.noisy_copies_length - one_hot_noisy_copy.shape[-1]

    if config.read_padding=='end':
        pad_start = 0
        pad_end = pad
    elif config.read_padding=='symmetric':
        pad_start = np.floor(pad/2).astype(int)
        pad_end   = np.ceil(pad/2).astype(int)

    one_hot_noisy_copy = F.pad(one_hot_noisy_copy, (pad_start, pad_end), "constant", 0)
     
    return one_hot_noisy_copy