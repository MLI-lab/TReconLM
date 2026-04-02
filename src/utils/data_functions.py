import random
import numpy as np

def write_data_to_file(filepath, data): 
    
    """
    This function writes the data to a text file with one data example per line.

    Args:
    filename (str): The path to the file to write the data to.
    data (list): The list of data examples.

    Returns:
    - 
    """
    with open(filepath, 'w') as f:
        for item in data:
            f.write("%s\n" % item)


def load_data_from_file(filepath):

    """
    This function reads the data from a text file with one data example per line.

    Args:
    filepath (str): The path to the file to read the data from.

    Returns:
    list: The list of data examples.
    """
     
    with open(filepath, 'r') as f:
        data = f.readlines()
        data = [x.strip() for x in data]
        return data


def read_clusters(clusters_file_path):
    # Read Clusters.txt
    clusters = []
    with open(clusters_file_path, 'r') as f:
        next(f)  # Skip the first line
        cluster = []
        for line in f:
            line = line.strip()
            if line == "===============================":
                clusters.append(cluster)
                cluster = []
            else:
                cluster.append(line)
        # Append the last cluster
        clusters.append(cluster)
    return clusters

def filter_string(s):
    return ''.join(c for c in s if c in 'ACTG')


def shuffle_data(orig_seqs, clusters):

    if len(orig_seqs) != len(clusters):
        raise ValueError('The lengths of the original sequences and the clusters are not the same.')
    
    c = list(zip(orig_seqs, clusters))
    random.shuffle(c)
    orig_seqs, clusters = zip(*c)

    return orig_seqs, clusters

def get_cluster_size_distribution(clusters):

    cluster_size_dict = {}
    for obs_seqs in clusters:

        #obs_seqs = cluster.split(':')[0].split('|')
        cluster_size = len(obs_seqs)
        if cluster_size in cluster_size_dict:
            cluster_size_dict[cluster_size] += 1
        else:
            cluster_size_dict[cluster_size] = 1

    return cluster_size_dict

def get_cluster_size_distribution_cpred(cpred_data):

    cluster_size_dict = {}
    for cluster in cpred_data:

        obs_seqs = cluster.split(':')[0].split('|')
        cluster_size = len(obs_seqs)
        if cluster_size in cluster_size_dict:
            cluster_size_dict[cluster_size] += 1
        else:
            cluster_size_dict[cluster_size] = 1

    return cluster_size_dict

def count_obs_seqs(cpred_data):

    num = 0
    for ex in cpred_data:
        obs_seqs = ex.split(':')[0].split('|')
        num += len(obs_seqs)

    return num

def save_np_array(data_path, np_array):

    np.save(data_path, np_array)
    return True

def save_list(list, file_path):
    with open(file_path, 'w') as f:
        for item in list:
            f.write("%s\n" % item)

def load_np_array(data_path):

    return np.load(data_path)



if __name__ == '__main__':

    test_sequence = 'ACTNGTACgTACG'
    test_sequence = filter_string(test_sequence)
    print(test_sequence) # ACTGTACTACG

    # 'ACTNGTACgTACG'
    # 'ACTGTACTACG'
