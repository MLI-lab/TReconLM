"""Synthetic data generation for TReconLM.

Generates training and test data by corrupting random DNA sequences through an
Insertion-Deletion-Substitution (IDS) channel. Supports multiple target types:

  - CPRED:       label is the original ground truth sequence
  - std_MSA:     label is the concatenated Multiple Sequence Alignment (MSA) of
                 each noisy read. Insertions are first marked with a placeholder
                 'I' in the observation; after all reads are aligned together,
                 the 'I's are replaced with random nucleotides.
  - ext_MSA:     same as std_MSA, but each insertion is realized as a random
                 nucleotide immediately (the observation already contains a real
                 base like 'G' instead of 'I'). The alignment still records
                 which positions were insertions, but the observation differs
                 during alignment construction.
  - std_NESTED:  label is the interleaved (nested) alignment (same insertion
                 handling as std_MSA)
  - ext_NESTED:  same as std_NESTED with immediate insertion realization

Standalone usage (generates a test dataset using Hydra):
    python -m src.data_pkg.data_generation

Hydra config:
    Base config:  src/hydra/data_config/data_config.yaml
    Can be overridden via Hydra CLI or experiment configs in src/hydra/train_config/exps/.
"""

import random
import numpy as np
import torch
import pickle
import warnings

import os
import sys
import gc
import json

import wandb

from ..data_pkg.IDS_channel import IDS_alignment_channel, IDS_channel, error_model_IDS_channel

from ..utils.data_functions import write_data_to_file
from ..utils.sys_functions import get_available_memory
from ..utils.helper_functions import create_folder, compute_homopolymer_map, weighted_choice
from ..utils.print_functions import print_list
from ..utils.wandb_utils import wandb_kwargs_via_cfg
from ..data_pkg.tokenizer import encode_list, pad_encoded_data

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig


################################################################################
# Hydra entry point
################################################################################

@hydra.main(config_path="../hydra/data_config", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point. Run with: python -m src.data_pkg.data_generation

    In sweep mode (sweep=True in config), generates 11 test datasets with
    progressively increasing IDS noise levels (each error probability shifted
    by +0.01 per step). This is used to evaluate model robustness across inreased 
    noise levels different from train distribution.
    """
    if cfg.get("sweep", False):
        print("Sweep mode enabled: generating 11 datasets with increasing noise levels...")
        for k in range(11):
            cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
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


################################################################################
# Test dataset generation (called by main)
################################################################################

def generate_test_dataset(cfg: DictConfig, k=None) -> None:
    """Generate a full test dataset: ground truths, noisy reads, tokenized tensors.

    Saves to <repo_root>/data/<cfg.folder_name>/.
    folder_name is composed from fields in pretraining_data.yaml:
        folder_name: pretraining_test_data/seed_num_${seed_number}/${group_name}/${noise_name}/${target_type}
        group_name:  ${data_type}_observation_size_${observation_size}_ground_truth_${ground_truth_length}
        noise_name:  ${data_type}

    Output files: ground_truth.txt, reads.txt, test_x.pt, test_y.pt,
    data_generation_config.json. Optionally logs the dataset as a W&B artifact.
    """
    # region dir to save data later
    script_dir = os.path.dirname(__file__)
    n = 2
    dir_n_levels_up = script_dir
    for _ in range(n):
        dir_n_levels_up = os.path.dirname(dir_n_levels_up)

    repo_path = dir_n_levels_up

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

        substitution_probability = random.uniform(substitution_probability_lb, substitution_probability_ub)
        insertion_probability = random.uniform(insertion_probability_lb, insertion_probability_ub)
        deletion_probability = random.uniform(deletion_probability_lb, deletion_probability_ub)
        observation_size_sampled = random.randint(observation_size_lb, observation_size)

        channel_statistics = {
            'substitution_probability': substitution_probability,
            'insertion_probability': insertion_probability,
            'deletion_probability': deletion_probability
        }

        data_example = generate_test_data(ground_truth_sequence, observation_size_sampled, channel_statistics, target_type, data_type, rng)
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
            name="generate_test_data",
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


################################################################################
# Core data generation pipelines (called by generate_test_dataset and pretrain)
################################################################################

def data_generation(data_set_size, observation_size, length_ground_truth, channel_statistics, target_type, data_type, rng=None, misclustering_config=None):
    """
    Generates a batch of synthetic training data corrupted by an IDS channel.

    Parameters:
        data_set_size (int): Number of ground truth examples to generate.
        observation_size (int): Number of noisy observations (traces) per example.
        length_ground_truth (int): Length of the clean sequence to generate.
        channel_statistics (dict): Dictionary of IDS probabilities (e.g., {'sub': 0.01, 'ins': 0.02, 'del': 0.03}).
        target_type (str): Defines the output format. Options:
            - 'CPRED'        : Clean prediction task (label is original sequence).
            - 'std_MSA'      : Standard Multiple Sequence Alignment (MSA). (calls IDS_alignment_channel not IDS channel, label is concatenation of true alignment of each noisy read)
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

def generate_test_data(ground_truth_sequence, observation_size, channel_statistics, target_type, data_type, rng):
    """Generate a single test data example from a given ground truth sequence.

    Unlike data_generation() which creates its own ground truths, this takes a
    specific ground truth as input. Used by generate_test_dataset().
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

    return data_example


################################################################################
# Helper functions 
################################################################################

def sample_ground_truth_length(length_config, rng=None):
    """
    Sample ground truth length from config.

    Args:
        length_config: int (fixed length, e.g. 110) or [min, max] (random length
                       uniformly sampled from that range, e.g. [60, 100])
        rng: random number generator (optional, uses global random if None)

    Returns:
        int: sampled length

    Examples:
        sample_ground_truth_length(110) -> 110 (always)
        sample_ground_truth_length([60, 100]) -> random int between 60 and 100
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

def generate_ground_truth_sequence(length, rng=None):

    """
    Generates a random DNA sequence of length 'length'.
    
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


################################################################################
# Error model data generation (not used for results in the paper)
################################################################################

def load_error_model(path):
    """
    Load error model JSON and preprocess for fast lookups during data generation.

    Preprocesses:
    - Burst length distributions into CDFs for sampling
    - Substitution matrix into per-base weighted sampling distributions

    Returns:
        dict: Error model ready for use with error_model_IDS_channel().
    """
    with open(path) as f:
        model = json.load(f)

    return model


def _sample_error_rates(error_model, gt_len, method, rng):
    """
    Sample per-read (sub_rate, del_rate, ins_rate) using the specified method.

    Methods:
        'fixed': Use the overall estimated rates (current default behavior).
        'beta': Sample from fitted Beta distributions (independent per error type).
        'histogram': Sample from the joint empirical histogram (preserves correlations).

    Returns:
        (sub_rate, del_rate, ins_rate) tuple of floats.
    """
    if method == 'beta':
        beta_dist = error_model.get('per_read_beta_distribution', {})
        rates = {}
        for err_type in ('sub', 'del', 'ins'):
            params = beta_dist.get(err_type, {'a': 1.0, 'b': 1.0})
            # Sample from Beta(a, b) and clamp
            val = np.random.default_rng(rng.randint(0, 2**31)).beta(params['a'], params['b'])
            rates[err_type] = max(0.0, float(val))
        return rates['sub'], rates['del'], rates['ins']

    elif method == 'histogram':
        joint_hist = error_model.get('joint_error_histogram', {})
        hist_gt_len = error_model.get('joint_error_histogram_gt_len', gt_len)
        if not joint_hist:
            # Fallback to fixed
            overall = error_model['overall']
            return overall['sub_rate'], overall['del_rate'], overall['ins_rate']

        # Sample a (n_sub, n_del, n_ins) tuple proportional to counts
        keys = list(joint_hist.keys())
        counts = [joint_hist[k] for k in keys]
        total = sum(counts)
        r = rng.random() * total
        cumsum = 0
        chosen_key = keys[0]
        for k, c in zip(keys, counts):
            cumsum += c
            if cumsum >= r:
                chosen_key = k
                break

        n_sub, n_del, n_ins = [int(x) for x in chosen_key.split(',')]
        # Normalize by histogram GT length, then scale to actual GT length
        sub_rate = n_sub / hist_gt_len
        del_rate = n_del / hist_gt_len
        ins_rate = n_ins / hist_gt_len
        return sub_rate, del_rate, ins_rate

    else:  # 'fixed'
        overall = error_model['overall']
        return overall['sub_rate'], overall['del_rate'], overall['ins_rate']


def data_generation_with_error_model(error_model, observation_size, length_ground_truth, target_type, rng=None, error_rate_sampling='fixed'):
    """
    Generate one synthetic training example using a realistic error model estimated
    from real-world data.

    This is the error-model counterpart of data_generation(). It:
    1. Generates a GT sequence using real-world nucleotide frequencies
    2. Corrupts it using context-dependent error rates (nucleotide, homopolymer,
       GC content, position zone)

    Currently supports target_type='CPRED' only.

    Parameters:
        error_model (dict): Preprocessed error model from load_error_model().
        observation_size (int): Number of noisy reads to generate.
        length_ground_truth: Length of GT (int or [min, max] for variable length).
        target_type (str): Must be 'CPRED'.
        rng: Random number generator.
        error_rate_sampling (str): How to sample per-read error rates.
            'fixed': use overall estimated rates (default, current behavior).
            'beta': sample from fitted Beta distributions per read.
            'histogram': sample from joint empirical histogram per read.

    Returns:
        List[List[str]]: Same format as data_generation():
            [['ground_truth', [gt]], [target_type, ['obs1|obs2|...:gt']]]
    """
    rng = rng or random

    if target_type != 'CPRED':
        raise ValueError(f'data_generation_with_error_model only supports CPRED, got {target_type}')

    # Sample GT length
    sampled_length = sample_ground_truth_length(length_ground_truth, rng=rng)

    # Generate realistic GT by sampling each base from real-world frequencies
    freqs = error_model['gt_nucleotide_global']
    bases = list(freqs.keys())
    weights = [freqs[b] for b in bases]
    gt = ''.join(weighted_choice(bases, weights, rng) for _ in range(sampled_length))

    # Precompute context for this GT (once, shared across all reads)
    homo_map = compute_homopolymer_map(gt)

    # Read length bounds from real data (if available in error model)
    # Reads outside these bounds are regenerated, matching real-data filtering
    read_len_min = error_model.get('read_length_min', 0)
    read_len_max = error_model.get('read_length_max', float('inf'))
    max_retries = 10

    # Misclustering rate from error model (0 if not estimated)
    misclustering_rate = error_model.get('misclustering_rate', 0.0)

    # Generate N noisy reads
    observation_list = []
    for _ in range(observation_size):
        if misclustering_rate > 0 and rng.random() < misclustering_rate:
            # Misclustered read: generate from a random different GT
            wrong_gt = ''.join(weighted_choice(bases, weights, rng) for _ in range(sampled_length))
            wrong_homo_map = compute_homopolymer_map(wrong_gt)
            sampled_sub, sampled_del, sampled_ins = _sample_error_rates(
                error_model, sampled_length, error_rate_sampling, rng)
            n_sub = round(sampled_sub * sampled_length)
            n_del = round(sampled_del * sampled_length)
            n_ins = round(sampled_ins * sampled_length)
            for attempt in range(max_retries):
                noisy_read = error_model_IDS_channel(
                    wrong_gt, n_sub, n_del, n_ins, error_model, wrong_homo_map, rng)
                if read_len_min <= len(noisy_read) <= read_len_max:
                    break
        else:
            # Normal read: corrupt from the correct GT
            sampled_sub, sampled_del, sampled_ins = _sample_error_rates(
                error_model, sampled_length, error_rate_sampling, rng)
            n_sub = round(sampled_sub * sampled_length)
            n_del = round(sampled_del * sampled_length)
            n_ins = round(sampled_ins * sampled_length)
            for attempt in range(max_retries):
                noisy_read = error_model_IDS_channel(
                    gt, n_sub, n_del, n_ins, error_model, homo_map, rng)
                if read_len_min <= len(noisy_read) <= read_len_max:
                    break
        if not (read_len_min <= len(noisy_read) <= read_len_max):
            warnings.warn(
                f"Read length {len(noisy_read)} outside bounds [{read_len_min}, {read_len_max}] "
                f"after {max_retries} retries (gt_len={sampled_length}), using anyway",
                stacklevel=2,
            )
        observation_list.append(noisy_read)

    # Format output (same as data_generation for CPRED)
    concatenated = '|'.join(observation_list)
    data_example = concatenated + ':' + gt

    data_pairs = [['ground_truth', [gt]]]
    data_pairs.append([target_type, [data_example]])

    return data_pairs


if __name__ == "__main__":
    print("data_generation.py")
    main()
