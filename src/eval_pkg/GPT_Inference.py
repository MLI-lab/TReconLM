import numpy as np
import torch
import subprocess
import os
import sys
import time
from typing import List, Dict, Any


from src.data_pkg.data_generation import unnest_strings
from src.utils.helper_functions import filter_string
from src.utils.print_functions import print_list
from src.gpt_pkg.beam_search import beam_search, beam_search_cached
from src.eval_pkg.majority_vote import majority_merge, extended_majority_vote


def gpt_prediction(test_examples, attn_mask, **kwargs) -> Dict[str, Any]:
    # configs
    model        = kwargs['model']
    ctx          = kwargs['ctx']
    device       = kwargs['device']
    decode       = kwargs['decode']
    max_new_tok  = kwargs['ground_truth_length'] # want to generate ground truth many sequneces
    temperature  = kwargs['temperature']
    sampling     = 'greedy' if kwargs['greedy'] else 'beam_search'
    itos         = kwargs['itos']
    stoi         = kwargs['stoi']
    constrained  = kwargs['constrained_generation'] # only generate ACTG
    beam_width   = kwargs.get('beam_width', 5)
    use_kv_cache = kwargs.get('use_kv_cache', True)  # KV caching always enabled (35x faster!)
    debug_beam   = kwargs.get('debug_beam', False)   # Debug logging
    cluster_size = kwargs.get('cluster_size', None)  # Cluster size for diagnostics

    # Entropy and attention tracking parameters
    track_entropy = kwargs.get('track_entropy', False)
    track_attention = kwargs.get('track_attention', False)
    track_all_layers = kwargs.get('track_all_layers', False)
    return_hidden_states = kwargs.get('return_hidden_states', False)

    pad_id   = stoi['#']
    token_entropies = None
    attention_data = None
    logits = None  # Initialize logits variable
    hidden_states = None  # Initialize hidden states variable
    model_inference_time = None  # Initialize model-only timing

    # generate
    # Parse read boundaries if tracking attention
    read_boundaries_batch = None
    if track_attention:
        read_boundaries_batch = []
        for example in test_examples:
            # Parse example to find read boundaries
            decoded_example = decode(example.tolist())
            reads_part = decoded_example.split(':')[0]
            reads = reads_part.split('|')

            boundaries = []
            current_pos = 0
            for read in reads:
                start_pos = current_pos
                end_pos = current_pos + len(read) - 1
                boundaries.append((start_pos, end_pos))
                current_pos += len(read) + 1  # +1 for '|' separator

            read_boundaries_batch.append(boundaries)

    if sampling == 'greedy':
        with torch.no_grad(), ctx:
            if (track_entropy or track_attention) and hasattr(model, 'generate_cpred_with_entropy_and_attention'):
                # Process each example individually to get proper read boundaries
                results = []
                for i in range(len(test_examples)):
                    single_example = test_examples[i:i+1]
                    single_mask = attn_mask[i:i+1]
                    single_boundaries = read_boundaries_batch[i] if read_boundaries_batch else None

                    Y_single, entropies_single, attention_single = model.generate_cpred_with_entropy_and_attention(
                        idx=single_example,
                        attn_mask=single_mask,
                        max_new_tokens=max_new_tok,
                        temperature=temperature,
                        top_k=None,
                        sampling='greedy',
                        constrained_generation=constrained,
                        itos=itos,
                        track_entropy=track_entropy,
                        track_attention=track_attention,
                        track_all_layers=track_all_layers,
                        read_boundaries=single_boundaries
                    )

                    results.append({
                        'Y': Y_single,
                        'entropies': entropies_single,
                        'attention': attention_single,
                        'read_boundaries': single_boundaries
                    })

                # Combine results
                Y = torch.cat([r['Y'] for r in results], dim=0)
                token_entropies = [r['entropies'] for r in results] if track_entropy else None
                attention_data = results if track_attention else None
                logits = None  # No logits in entropy/attention path

            elif track_entropy and hasattr(model, 'generate_cpred_with_entropy'):
                Y, token_entropies = model.generate_cpred_with_entropy(
                    idx=test_examples,
                    attn_mask=attn_mask,
                    max_new_tokens=max_new_tok,
                    temperature=temperature,
                    top_k=None,
                    sampling='greedy',
                    constrained_generation=constrained,
                    itos=itos,
                    track_entropy=track_entropy
                )
                logits = None  # No logits in entropy-only path
            else:
                # Check if we need logits and/or hidden states
                analyze_vote_confidence = kwargs.get('analyze_vote_confidence', False)

                if analyze_vote_confidence and return_hidden_states:
                    # Collect both logits and hidden states
                    Y, logits, hidden_states = model.generate_cpred(
                        idx=test_examples,
                        attn_mask=attn_mask,
                        max_new_tokens=max_new_tok,
                        temperature=temperature,
                        top_k=None,
                        sampling='greedy',
                        constrained_generation=constrained,
                        itos=itos,
                        collect_logits=True,
                        return_hidden_states=True
                    )
                    print(f"[DEBUG] Collected logits: {logits.shape}, hidden_states: {hidden_states.shape}")
                elif analyze_vote_confidence:
                    # Use new logits collection feature
                    Y, logits = model.generate_cpred(
                        idx=test_examples,
                        attn_mask=attn_mask,
                        max_new_tokens=max_new_tok,
                        temperature=temperature,
                        top_k=None,
                        sampling='greedy',
                        constrained_generation=constrained,
                        itos=itos,
                        collect_logits=True
                    )
                    print(f"[DEBUG] Collected logits during generation: {logits.shape}")
                elif return_hidden_states:
                    # Collect only hidden states
                    Y, hidden_states = model.generate_cpred(
                        idx=test_examples,
                        attn_mask=attn_mask,
                        max_new_tokens=max_new_tok,
                        temperature=temperature,
                        top_k=None,
                        sampling='greedy',
                        constrained_generation=constrained,
                        itos=itos,
                        return_hidden_states=True
                    )
                    print(f"[DEBUG] Collected hidden_states: {hidden_states.shape}")
                else:
                    # Standard generation without logits or hidden states
                    # Time only the pure model inference (no tracking overhead)
                    if str(device).startswith('cuda'):
                        torch.cuda.synchronize()

                    t_model_start = time.perf_counter()

                    Y = model.generate_cpred(
                        idx=test_examples,
                        attn_mask=attn_mask,
                        max_new_tokens=max_new_tok,
                        temperature=temperature,
                        top_k=None,
                        sampling='greedy',
                        constrained_generation=constrained,
                        itos=itos
                    )

                    if str(device).startswith('cuda'):
                        torch.cuda.synchronize()

                    model_inference_time = time.perf_counter() - t_model_start
        

    else:  # beam_search
        if debug_beam:
            print(f"[DEBUG] Starting beam search - use_kv_cache={use_kv_cache}, beam_width={beam_width}")

        with torch.no_grad(), ctx:
            if use_kv_cache:
                # Use the new cached beam search
                Y_beams = beam_search_cached(
                    model=model,
                    beam_width=beam_width,
                    sequence_length=max_new_tok,
                    x=test_examples,
                    attn_mask=attn_mask,
                    device=device,
                    pad_id=pad_id,
                    debug=debug_beam
                )
            else:
                # Use the original non-cached version
                Y_beams = beam_search(
                    model=model,
                    beam_width=beam_width,
                    sequence_length=max_new_tok,
                    x=test_examples,
                    attn_mask=attn_mask,
                    device=device,
                    pad_id=pad_id
                )

        if debug_beam:
            print(f"[DEBUG] Beam search completed. Y_beams shape: {Y_beams.shape}")

        # Y_beams has shape [B, beam_width, T+max_new_tok]
        # Take the best beam (index 0) for each example
        Y = Y_beams[:, 0, :]
        logits = None  # Beam search doesn't collect logits yet

    decoded = [decode(row) for row in Y.tolist()]

    # trim to label length
    candidates = [txt.split(':', 1)[1][:max_new_tok] for txt in decoded]

    # Logits are now collected efficiently during generation when needed
    # No additional processing required here

    result = {
        'candidate_sequences': candidates,
        'token_entropies': token_entropies,
        'attention_data': attention_data,
        'cluster_size': cluster_size,
        'logits': logits,
        'model_inference_time': model_inference_time  # Pure model time (no preprocessing/tracking)
    }

    # Add hidden states if collected (key expected by probe.py)
    if hidden_states is not None:
        result['dec_hidden_last'] = hidden_states

    return result



def gpt_alignment(test_examples, attn_mask, alignment_size, **kwargs):
    """
    Generates an alignment (MSA or nested) using the GPT model and applies postprocessing to prepare the output for majority voting.
    Alignment_size (int) is expected number of aligned sequences (equal to cluster size).

    MSA decoding (when 'MSA' in target_type):
        - If output contains '|' tokens, they are treated as alignment separators.
        - If not, the output is evenly split into alignment_size chunks as fallback.
        - All sequences are truncated to the length of the shortest valid one.

    Nested decoding (when 'NESTED' in target_type):
        - Output is un-nested into alignment_size (i.e. cluster size) equal-length parts.
        - All parts are truncated to the length of the shortest one.

    Merging:
        - std uses majority_merge
        - ex uses extended_majority_vote
    """

    model = kwargs['model']
    decode = kwargs['decode']
    ctx = kwargs['ctx']
    block_size = kwargs['block_size']
    temperature = kwargs['temperature']
    top_k = kwargs['top_k']
    device = kwargs['device']
    target_type = kwargs['target_type']
    itos = kwargs['itos']
    stoi = kwargs['stoi']

    pad_id   = stoi['#']

    try:
        # generate max new tokens until out of block length
        if 'MSA' in target_type or 'NESTED' in target_type:
            max_new_tokens = block_size-len(test_example.split(':')[0])
    except Exception as e:
        print(f"Error in assigning max new tokens {e}")

    sampling = kwargs['sampling'] # 'greedy' 'beam_search'

    # generate
    if sampling == 'greedy':    
        with torch.no_grad(), ctx:
            Y = model.generate(test_examples, attn_mask ,max_new_tokens, temperature = temperature, top_k = top_k, itos=itos)
        

    elif sampling == 'beam_search':
        beam_width = kwargs['beam_width']
        with torch.no_grad(), ctx:
            Y_beams = beam_search(model = model, beam_width = beam_width, sequence_length = max_new_tokens, x = test_examples, attn_mask=attn_mask, device = device, pad_id = pad_id)
        # Y_beams has shape [B, beam_width, T+max_new_tokens]
        # Take the best beam (index 0) for each example
        Y = Y_beams[:, 0, :] 
    
    decoded = [decode(row) for row in Y.tolist()]

    out_candidates = []
    out_alignments = []

    for ex, txt in zip(test_examples, decoded):
        # strip prefix and EOS
        try:
            body = txt.split(':', 1)[1].split('#', 1)[0]
        except IndexError:
            body = txt.split('#', 1)[0]

        # split into alignment pieces
        if 'MSA' in target_type:
            if '|' in body:
                pieces = body.split('|')
            else:
                # even split fallback
                chunk = len(body) // alignment_size
                pieces = [body[i*chunk:(i+1)*chunk] for i in range(alignment_size)]
        else:  # nested
            pieces = unnest_strings(nested_str=body, num_segments=alignment_size)

        # truncate all to shortest
        min_len = min(len(s) for s in pieces)
        pieces = [s[:min_len] for s in pieces]

        # merge
        if 'ex' in target_type:
            candidate = extended_majority_vote(pieces, length=min_len, check_length=alignment_size)
        else:
            candidate = majority_merge(pieces, weight=0.4)

        out_candidates.append(candidate)
        out_alignments.append(pieces)

    return {
        'candidate_sequences':       out_candidates,
        'predicted_alignment_list':  out_alignments
    }



class GPT_Inference:
    def __init__(self, inference_params):
        self.inference_params = inference_params

    def inference(self, test_examples, alignment_size=None):
        """
        test_examples: List[str] of examples that will be batched.
        returns a dict with keys: candidate_sequences (List[str]).
        """
        target_type = self.inference_params['target_type']
        stoi        = self.inference_params['stoi']
        encode      = self.inference_params['encode']
        decode      = self.inference_params['decode']
        device      = self.inference_params['device']

        pad_id   = stoi['#']
        colon_id = stoi[':']

        # batching
        prefixes      = [ex.split(':')[0] for ex in test_examples]
        enc_prefixes  = [encode(p) for p in prefixes]

        B              = len(enc_prefixes)
        max_prefix_len = max(len(e) for e in enc_prefixes)
        T              = max_prefix_len + 1           # +1 for the colon

        # Check if any input exceeds block size
        block_size = self.inference_params.get('block_size', 1500)
        for i, (prefix, enc) in enumerate(zip(prefixes, enc_prefixes)):
            if len(enc) + 1 > block_size:  # +1 for colon
                cluster_size = len(prefix.split('|'))
                print(f"\n  WARNING: Example {i} exceeds block_size!")
                print(f"   Cluster size: {cluster_size}")
                print(f"   Encoded length: {len(enc) + 1} tokens (including colon)")
                print(f"   Block size: {block_size}")
                print(f"   TRUNCATION WILL OCCUR - results may be unreliable!")

        # left-padded tensor of input IDs
        X = torch.tensor([[pad_id] * (max_prefix_len - len(e))  + e + [colon_id] for e in enc_prefixes], dtype=torch.long, device=device)

        # matching attention mask  (False = pad, True = real)
        attn_mask = torch.tensor([[False] * (max_prefix_len - len(e)) + [True] * (len(e) + 1) for e in enc_prefixes], dtype=torch.bool, device=device)

        if 'MSA' in target_type or 'NESTED' in target_type:
            return gpt_alignment(X, attn_mask, alignment_size, **self.inference_params)

        elif 'CPRED' in target_type:
            # Pass cluster size info for tracking/diagnostics
            # alignment_size can be a single int or a list - take max for safety
            cluster_size = max(alignment_size) if isinstance(alignment_size, list) else alignment_size
            return gpt_prediction(X, attn_mask, cluster_size=cluster_size, **self.inference_params)

        else:
            raise ValueError(f"Unknown target_type: {target_type}")
