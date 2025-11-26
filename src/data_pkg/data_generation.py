import random
import numpy as np
import torch
import pickle

import string
from collections import Counter

import os
import sys
import gc
import json

import wandb

from ..data_pkg.IDS_channel import IDS_alignment_channel, IDS_channel

from ..utils.data_functions import write_data_to_file
from ..utils.sys_functions import get_available_memory
from ..utils.helper_functions import create_folder
from ..utils.print_functions import print_list
from ..utils.wandb_utils import wandb_kwargs_via_cfg
from ..data_pkg.prepare import encode_list, pad_encoded_data


import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig


def sample_ground_truth_length(length_config, rng=None):
    """
    Sample ground truth length from config. Supports both fixed and interval lengths.

    Args:
        length_config: int (fixed length) or [min, max] (interval for random sampling)
        rng: random number generator (optional, uses global random if None)

    Returns:
        int: sampled length

    Examples:
        sample_ground_truth_length(110) -> 110 (always)
        sample_ground_truth_length([60, 100]) -> random int between 60-100
    """
    if isinstance(length_config, (list, tuple, ListConfig)):
        min_len, max_len = length_config
        if rng is not None:
            return rng.randint(min_len, max_len)
        else:
            return random.randint(min_len, max_len)
    else:
        return length_config


def validate_block_size_for_variable_length(length_config, block_size, observation_size):
    """
    Validate that block_size can accommodate the maximum possible sequence length.

    Args:
        length_config: int or [min, max]
        block_size: available block size
        observation_size: maximum number of reads

    Raises:
        ValueError: if block_size is too small for maximum possible sequence
    """
    # Calculate maximum possible length
    max_length = max(length_config) if isinstance(length_config, (list, tuple, ListConfig)) else length_config

    # Calculate required space: input reads + separators + colon + ground truth
    # Rough estimate: max_length * observation_size (reads) + observation_size (separators) + 1 (:) + max_length (GT)
    required_space = max_length * observation_size + observation_size + 1 + max_length

    if required_space > block_size:
        raise ValueError(
            f"Block size {block_size} too small for maximum sequence length {max_length}. "
            f"Required space: {required_space} (reads: {max_length}×{observation_size}, "
            f"separators: {observation_size}, colon: 1, GT: {max_length}). "
            f"Increase block_size or reduce max ground_truth_length."
        )
        
def nest_strings(input_str: str) -> str: 

    """
    This function takes a string of the form (input_str) 'ABC|DEF|GHI' and returns a string of the form (nested_str) 'ADGBEHCFI'.

    Args:
    input_str (str): The input string.

    Returns:
    nested_string (str): The nested string.
    """ 

    split_strings = input_str.split('|')
    nested_string = ''

    min_length = min(len(s) for s in split_strings)

    for i in range(min_length):
        for s in split_strings:
            nested_string += s[i]

    for s in split_strings:
        if len(s) > min_length:
            nested_string += s[min_length:]

    return nested_string

def unnest_strings(nested_str: str, num_segments: int) -> list:
    """
    This function takes a string of the form (nested_str) 'ADGBEHCFI' and returns a list of segments of the form ['ABC', 'DEF', 'GHI'].

    Args:
    nested_str (str): The nested string.
    num_segments (int): The number of segments.

    Returns:
    segments (list): The list of original segments.
    """
    
    segments = [''] * num_segments

    for i, char in enumerate(nested_str):
        segment_index = i % num_segments
        segments[segment_index] += char

    return segments

def generate_ground_truth_sequence(length, rng=None): 

    """
    Generates a random DNA sequence of length 'length'.
    If a seed is set elsewhere (e.g., random.seed(1337)), this function will produce deterministic results.
    If no seed is set, it generates different sequences on every execution.

    Args:
    length (int): The length of the sequence.

    Returns:
    sequence (str): The generated sequence.
    """

    try:
        # Try the original approach (works with older numpy versions)
        sequence = ''.join(rng.choice('ATGC') for _ in range(length))
    except (TypeError, ValueError):
        # Fallback for newer numpy versions that require 1D arrays
        sequence = ''.join(rng.choice(list('ATGC')) for _ in range(length))
    return sequence

def sample_sequences(file_name, n):
    """
    Randomly samples `n` sequences from a text file.

    Each line in the file is assumed to contain a single sequence (e.g., a DNA string).
    The function reads all lines from the file, samples `n` of them at random (without replacement),
    strips any surrounding whitespace, and returns them as a list of strings.

    Parameters:
        file_name (str): Path to the input text file containing sequences, one per line.
        n (int): Number of sequences to sample.

    Returns:
        List[str]: A list of `n` randomly sampled sequences.
    
    Raises:
        ValueError: If `n` is greater than the number of available sequences in the file.
    """
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    sampled_lines = random.sample(lines, n)
    sampled_sequences = [line.strip() for line in sampled_lines]
    return sampled_sequences


def data_generation(data_set_size, observation_size, length_ground_truth, channel_statistics, target_type, data_type, rng=None, misclustering_config=None):
    """
    Generates synthetic sequence data corrupted by an IDS (Insertion/Deletion/Substitution) channel. 

    Parameters:
        data_set_size (int): Number of ground truth examples to generate.
        observation_size (int): Number of noisy observations (traces) per example.
        length_ground_truth (int): Length of the clean sequence to generate.
        channel_statistics (dict): Dictionary of IDS probabilities (e.g., {'sub': 0.01, 'ins': 0.02, 'del': 0.03}).
        target_type (str): Defines the output format. Options:
            - 'CPRED'        : Clean prediction task (label is original sequence).
            - 'std_MSA'      : Standard multiple sequence alignment. (calls IDS_alignment_channel not IDS channel, label is concatenation of true alignment of each noisy read)
            - 'ext_MSA'      : Extended MSA with additional structure. (calls IDS_alignment_channel not IDS channel)
            - 'std_NESTED'   : Standard nested alignment. (calls IDS_alignment_channel not IDS channel, label is nested alignment with input still concatenation of noisy reads (not nested noisy reads))
            - 'ext_NESTED'   : Extended nested alignment. (calls IDS_alignment_channel not IDS channel)
        data_type (str): Must be 'ids_data'. Defines how data is generated.
        - 'std_*': insertions are initially marked with 'I' and resolved to random nucleotides after alignment in IDS alignment channel.
        - 'ext_*': insertions are realized immediately as random nucleotides in IDS alignment channel.
        rng: Random number generator for reproducibility.
        misclustering_config (dict, optional): Configuration for adding contamination during training.
            - 'enabled': bool, whether to enable misclustering
            - 'contamination_rate_lb': float, lower bound for contamination rate (0.0-1.0)
            - 'contamination_rate_ub': float, upper bound for contamination rate (0.0-1.0)

    Returns:
        List[List[str]]: A list of two entries:
            - ['ground_truth', list of clean sequences] 
            - [target_type, list of training samples in format "<obs_1>|<obs_2>|...:<label>"] # target_type e.g. 'CPRED' 
    """

    rng = rng or random        # falls back to global if rng not provided
        
    ground_truth_sequence_list     = []
    data_list                      = []    
        
    if data_type == 'ids_data':
        sampled_lengths = []
        for i in range(data_set_size):
            # Sample ground truth length (supports both fixed and interval lengths)
            sampled_length = sample_ground_truth_length(length_ground_truth, rng=rng)
            sampled_lengths.append(sampled_length)
            ground_truth_sequence = generate_ground_truth_sequence(sampled_length, rng)
            ground_truth_sequence_list.append(ground_truth_sequence)

        # Log sampled lengths for debugging (only for batches, not single examples)
        if data_set_size > 1 and isinstance(length_ground_truth, (list, tuple, ListConfig)):
            print(f"[DEBUG] Sampled lengths for batch of {data_set_size}: {sampled_lengths}")

    else:
        raise ValueError('Data type not recognized!')


    for i, ground_truth_sequence in enumerate(ground_truth_sequence_list):
        if i % int(1e3) == 0 and i != 0: # Progress log (every 1000 examples)
            print(f'data generation: {i:.2e}')

        observation_sequence_list = [] # Store noisy reads as list for current ground_truth_sequence
        alignment_sequence_list   = []  # Store true alignment of noisy reads ( with '-' for del and 'I' for ins?) ( For CPRED empty as we do not predict alignmemt just ground thruth sequence)

        # Sample contamination rate for this example
        if misclustering_config and misclustering_config.get('enabled', False):
            cont_rate = rng.uniform(
                misclustering_config.get('contamination_rate_lb', 0.0),
                misclustering_config.get('contamination_rate_ub', 0.0)
            )
        else:
            cont_rate = 0.0

        if target_type == 'CPRED':
            for j in range(observation_size):
                # Decide if this observation should be contaminated
                if cont_rate > 0 and rng.random() < cont_rate:
                    # Generate contaminant from a different ground truth sequence
                    # The RNG state has advanced, so this will be different from the original
                    contaminant_length = len(ground_truth_sequence)
                    contaminant_gt = generate_ground_truth_sequence(contaminant_length, rng)
                    observation_sequence = IDS_channel(contaminant_gt, channel_statistics, rng)
                else:
                    # Normal observation from the true ground truth
                    observation_sequence = IDS_channel(ground_truth_sequence, channel_statistics, rng)
                observation_sequence_list.append(observation_sequence)
    
        elif 'MSA' in target_type or 'NESTED' in target_type:
            if not 'std' in target_type and not 'ext' in target_type:
                print('target_type: ', target_type)
                raise ValueError('data_generation.py: target type not fully specified!')

            # For alignment-based targets, we need to handle contamination differently
            if cont_rate > 0:
                # Generate observations, some of which will be contaminants
                observation_sequence_list = []
                alignment_sequence_list = []
                for j in range(observation_size):
                    if rng.random() < cont_rate:
                        # Generate contaminant from different ground truth
                        contaminant_length = len(ground_truth_sequence)
                        contaminant_gt = generate_ground_truth_sequence(contaminant_length, rng)
                        obs_list, align_list = IDS_alignment_channel(ground_truth_sequence = contaminant_gt,
                                                    channel_statistics = channel_statistics,
                                                    observation_size = 1,
                                                    target_type = target_type, print_flag = False, rng=rng)
                        observation_sequence_list.extend(obs_list)
                        # For contaminants, we still align against the original ground truth for consistency
                        alignment_sequence_list.extend(align_list)
                    else:
                        # Normal observation
                        obs_list, align_list = IDS_alignment_channel(ground_truth_sequence = ground_truth_sequence,
                                                    channel_statistics = channel_statistics,
                                                    observation_size = 1,
                                                    target_type = target_type, print_flag = False, rng=rng)
                        observation_sequence_list.extend(obs_list)
                        alignment_sequence_list.extend(align_list)
            else:
                # No contamination, generate all observations normally
                observation_sequence_list, alignment_sequence_list = IDS_alignment_channel(ground_truth_sequence = ground_truth_sequence,
                                            channel_statistics = channel_statistics,
                                            observation_size = observation_size,
                                            target_type = target_type, print_flag = False, rng=rng)
            exists = any('I' in s for s in observation_sequence_list)
            if exists:
                print('I in observation sequence list')
                print_list(observation_sequence_list)
                sys.exit()
        else:
            raise ValueError('Target type not recognized!')
        
        concatenated_observation_sequences = '|'.join(observation_sequence_list)
            
        if target_type == 'CPRED':
            data_example = concatenated_observation_sequences + ":" + ground_truth_sequence
                    
        elif 'MSA' in target_type: 
            concatenated_alignments = '|'.join(alignment_sequence_list)
            data_example = concatenated_observation_sequences + ":" + concatenated_alignments
                
        elif 'NESTED' in target_type:   
            concatenated_alignments = '|'.join(alignment_sequence_list)
            nested_alignment    = nest_strings(concatenated_alignments)
            data_example = concatenated_observation_sequences + ":" + nested_alignment
        else:
            raise ValueError('Target type not recognized!')  
            
        data_list.append(data_example)

        if i % int(1e5) == 0 and i != 0: # For tracking memory 
            print(f'data generation - batch {i:.2e}: finished')
            print('Available RAM (GB):', get_available_memory())
            gc.collect

    data_pairs = [['ground_truth',ground_truth_sequence_list]]
    data_pairs.append([target_type, data_list])

    return data_pairs

def test_data_generation(ground_truth_sequence, observation_size, channel_statistics, target_type, data_type, rng):
    
    """
    Used to generate test data. 
    """

    observation_sequence_list = []
    alignment_sequence_list   = []

        
    if data_type == 'ids_data':
        if target_type == 'CPRED': 
            for j in range(observation_size):
                observation_sequence = IDS_channel(ground_truth_sequence, channel_statistics, rng)
                observation_sequence_list.append(observation_sequence)

        elif 'MSA' in target_type or 'NESTED' in target_type:
            if not 'std' in target_type and not 'ext' in target_type:
                print('target_type: ', target_type)
                raise ValueError('data_generation.py: target type not fully specified!')
            observation_sequence_list, alignment_sequence_list = IDS_alignment_channel(ground_truth_sequence = ground_truth_sequence, 
                                        channel_statistics = channel_statistics, 
                                        observation_size = observation_size,
                                        target_type = target_type, print_flag = False)
            exists = any('I' in s for s in observation_sequence_list)
            if exists:
                print('I in observation sequence list')
                print_list(observation_sequence_list)
                sys.exit()
        else:
            raise ValueError('Target type not recognized!')
    
    concatenated_observation_sequences = '|'.join(observation_sequence_list)
    
    if target_type == 'CPRED':
        data_example = concatenated_observation_sequences + ":" + ground_truth_sequence
            
    elif 'MSA' in target_type: 
        concatenated_alignments = '|'.join(alignment_sequence_list)
        data_example = concatenated_observation_sequences + ":" + concatenated_alignments
        
    elif 'NESTED' in target_type:   
        concatenated_alignments = '|'.join(alignment_sequence_list)
        nested_alignment    = nest_strings(concatenated_alignments)
        data_example = concatenated_observation_sequences + ":" + nested_alignment
    else:
        raise ValueError('Target type not recognized!')  
    
    return data_example

# or from python -m src.data_pkg.data_generation when in TReconLM
@hydra.main(config_path="../hydra/data_config", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.get("sweep", False):
        print("Sweep mode enabled: generating 11 datasets with increasing noise levels...")
        for k in range(11):
            cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))  # make a deep copy
            shift = 0.01 * k

            cfg_copy.substitution_probability_lb = 0.01 + shift
            cfg_copy.substitution_probability_ub = 0.10 + shift

            cfg_copy.insertion_probability_lb = 0.01 + shift
            cfg_copy.insertion_probability_ub = 0.10 + shift

            cfg_copy.deletion_probability_lb = 0.01 + shift
            cfg_copy.deletion_probability_ub = 0.10 + shift

            cfg_copy.folder_name = f"{cfg.folder_name}_sweep_k{k}"
            cfg_copy.seed_number = cfg.seed_number 

            generate_test_dataset(cfg_copy, k)
    else:
        generate_test_dataset(cfg)


def generate_test_dataset(cfg: DictConfig, k=None) -> None:
    # region dir to save data later
    script_dir = os.path.dirname(__file__)
    n = 2
    dir_n_levels_up = script_dir
    for _ in range(n):
        dir_n_levels_up = os.path.dirname(dir_n_levels_up)

    repo_path = dir_n_levels_up
    data_pkg_dir = os.path.join(repo_path,'src','data_pkg')

    config_dict = wandb_kwargs_via_cfg(cfg)

    seed_number = cfg.seed_number
    rng = random.Random(seed_number)

    observation_size = cfg.observation_size
    observation_size_lb=cfg.observation_size_lb
    ground_truth_length = cfg.ground_truth_length
    data_type = cfg.data_type
    test_size = cfg.data_set_size
    target_type = cfg.target_type
    sequence_type = cfg.sequence_type
    data_set_size = cfg.data_set_size
    block_size = cfg.block_size

    substitution_probability_lb = cfg.substitution_probability_lb
    substitution_probability_ub = cfg.substitution_probability_ub
    insertion_probability_lb = cfg.insertion_probability_lb
    insertion_probability_ub = cfg.insertion_probability_ub
    deletion_probability_lb = cfg.deletion_probability_lb
    deletion_probability_ub = cfg.deletion_probability_ub

    data_list = []
    ground_truth_list = []
    reads_list = []
    separator = '==============================='

    for i in range(test_size):
        # Sample ground truth length (supports both fixed and interval lengths)
        sampled_length = sample_ground_truth_length(ground_truth_length, rng=rng)
        ground_truth_sequence = generate_ground_truth_sequence(sampled_length, rng=rng)
        ground_truth_list.append(ground_truth_sequence)

        if i % 1000 == 0 and i != 0:
            gc.collect()
            print('Available RAM (GB):', get_available_memory())
            print(f'ground truth data generation: {i:.2e}')

    for index, ground_truth_sequence in enumerate(ground_truth_list):
        if index % 1000 == 0 and index != 0:
            gc.collect()
            print('Available RAM (GB):', get_available_memory())
            print(f'reads data generation: {index:.2e}')
            #print(f"\n[raw example @{index}] {data_list[index-1] if index>0 else ' (waiting for first)'}")

        substitution_probability = random.uniform(substitution_probability_lb, substitution_probability_ub)
        insertion_probability = random.uniform(insertion_probability_lb, insertion_probability_ub)
        deletion_probability = random.uniform(deletion_probability_lb, deletion_probability_ub)
        observation_size_sampled = random.randint(observation_size_lb, observation_size)

        channel_statistics = {
            'substitution_probability': substitution_probability,
            'insertion_probability': insertion_probability,
            'deletion_probability': deletion_probability
        }

        data_example = test_data_generation(ground_truth_sequence, observation_size_sampled, channel_statistics, target_type, data_type, rng)
        data_list.append(data_example)
        reads_list += data_example.split(':')[0].split('|')
        reads_list.append(separator)
    
    max_len = max(len(s) for s in data_list)
    print(f"Max data example length: {max_len}")

    print("\nExample data_example (noisy reads + label):")
    print(data_list[0])  

    meta_path = os.path.join(os.path.dirname(__file__), f'meta_{sequence_type}.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        config_dict['meta_vocab_size'] = meta_vocab_size
        stoi, itos = meta['stoi'], meta['itos']

    encoded_data = encode_list(data_list, stoi)
    padded_encoded_data = pad_encoded_data(encoded_data, block_size, stoi)
    lengths = [len(seq) for seq in padded_encoded_data]
    unique_lengths = set(lengths)

    assert len(unique_lengths) == 1 and list(unique_lengths)[0] == block_size

    np_data = np.array(padded_encoded_data, dtype=np.int64)
    x_temp = np_data[:, 0:block_size-1]
    y_temp = np_data[:, 1:block_size]
    x = torch.from_numpy(x_temp.astype(np.int64))
    y = torch.from_numpy(y_temp.astype(np.int64))

    max_len = max(len(s) for s in reads_list)
    config_dict['max_len_reads'] = max_len

    
    folder_name = cfg.folder_name 
    folder_path = os.path.join(repo_path, 'data', folder_name)
    print(f"folder_path is {folder_path}")
    create_folder(folder_path)
    if cfg.save_flag:
        write_data_to_file(filepath=f'{folder_path}/{target_type}_data.txt', data=data_list)
        write_data_to_file(filepath=f'{folder_path}/ground_truth.txt', data=ground_truth_list)
        write_data_to_file(filepath=f'{folder_path}/reads.txt', data=reads_list)
        torch.save(x, os.path.join(folder_path, 'test_x.pt'))
        torch.save(y, os.path.join(folder_path, 'test_y.pt'))

        json_file_path = f'{folder_path}/data_generation_config.json'
        with open(json_file_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    if cfg.wandb.wandb_log:
        run = wandb.init(
            project=cfg.wandb.wandb_project,
            entity=cfg.wandb.wandb_entity,
            name="test_data_generation",
            job_type="data_generation",
            config=config_dict,
            dir=folder_path
        )

    if observation_size == observation_size_lb:
        obs_suffix = f"_fixedN{observation_size}"
    else:
        obs_suffix = ""

    if k is not None:
        artifact_name = f"sweep{k}_seed{seed_number}_gl{ground_truth_length}_bs{block_size}_ds{data_set_size}{obs_suffix}"
    else:
        artifact_name = f"{target_type}_seed{seed_number}_gl{ground_truth_length}_bs{block_size}_ds{data_set_size}{obs_suffix}"

    artifact = wandb.Artifact(
        name=artifact_name,
        type="dataset",
        description="Synthetic test dataset for model evaluation",
        metadata={
            "data_type": data_type,
            "target_type": target_type,
            "sequence_type": sequence_type,
            "block_size": block_size,
            "data_set_size": data_set_size,
            "observation_size": observation_size,
            "observation_size_lb": observation_size_lb,
            "ground_truth_length": ground_truth_length
        }
    )
    try:
        artifact.add_file(os.path.join(folder_path, 'test_x.pt'))
        artifact.add_file(os.path.join(folder_path, 'test_y.pt'))
        artifact.add_file(os.path.join(folder_path, f'{target_type}_data.txt'))
        artifact.add_file(os.path.join(folder_path, 'ground_truth.txt'))
        artifact.add_file(os.path.join(folder_path, 'reads.txt'))
        artifact.add_file(os.path.join(folder_path, 'data_generation_config.json'))
    except Exception as e:
        print(f"Exception: {e}")

    print("Logging artifact to W&B...")
    run.log_artifact(artifact)
    run.finish()

if __name__ == "__main__":
    print("data_generation.py")
    main()