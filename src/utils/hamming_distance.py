import random  

def hamming_distance_postprocessed(ground_truth, reconstructed):
    """
    Computes normalized Hamming distance between ground_truth and reconstructed sequence.
    
    If reconstructed is shorter than ground_truth, randomly fill with A/C/G/T.
    If reconstructed is longer, cut it to the ground_truth length.
    """

    L = len(ground_truth)
    reconstructed = list(reconstructed)

    if len(reconstructed) < L:
        # Pad randomly until reaching length L
        bases = ['A', 'C', 'G', 'T']
        while len(reconstructed) < L:
            reconstructed.append(random.choice(bases))
    elif len(reconstructed) > L:
        # Cut down to L
        reconstructed = reconstructed[:L]

    # Now compute Hamming distance
    mismatches = sum(c1 != c2 for c1, c2 in zip(ground_truth, reconstructed))
    normalized_distance = mismatches / L

    return normalized_distance