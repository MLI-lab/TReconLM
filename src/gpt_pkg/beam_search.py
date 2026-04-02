import torch
import numpy as np
import os
from typing import Optional, List, Tuple

def _duplicate_beam_caches(beam_caches: List[Tuple[torch.Tensor, torch.Tensor]],
                          beam_indices: torch.Tensor,
                          device: torch.device,
                          debug: bool = False) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Duplicate KV caches based on beam selection.

    Args:
        beam_caches: List of (k, v) tuples for each layer
        beam_indices: [B, beam_width] indices of which beams to keep
        device: Device to place tensors on
        debug: Enable debug logging

    Returns:
        duplicated_caches: New caches with duplicated entries
    """
    if beam_caches is None:
        return None

    if debug:
        print(f"[DEBUG] _duplicate_beam_caches: beam_indices shape {beam_indices.shape}")

    duplicated_caches = []
    B, beam_width = beam_indices.shape

    for layer_idx, (k, v) in enumerate(beam_caches):
        # k, v shapes: [B*K_old, num_heads, seq_len, head_dim]
        B_K_old, num_heads, seq_len, head_dim = k.shape
        K_old = B_K_old // B

        # Reshape to [B, K_old, num_heads, seq_len, head_dim]
        k_reshaped = k.view(B, K_old, num_heads, seq_len, head_dim)
        v_reshaped = v.view(B, K_old, num_heads, seq_len, head_dim)

        # Gather based on beam_indices
        # beam_indices: [B, beam_width] -> [B, beam_width, 1, 1, 1]
        gather_indices = beam_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gather_indices = gather_indices.expand(-1, -1, num_heads, seq_len, head_dim)

        k_selected = k_reshaped.gather(1, gather_indices)  # [B, beam_width, num_heads, seq_len, head_dim]
        v_selected = v_reshaped.gather(1, gather_indices)  # [B, beam_width, num_heads, seq_len, head_dim]

        # Flatten back to [B*beam_width, num_heads, seq_len, head_dim]
        k_flat = k_selected.view(B * beam_width, num_heads, seq_len, head_dim)
        v_flat = v_selected.view(B * beam_width, num_heads, seq_len, head_dim)

        duplicated_caches.append((k_flat, v_flat))

        if debug and layer_idx == 0:
            print(f"[DEBUG] Layer {layer_idx}: k {k.shape} -> {k_flat.shape}, v {v.shape} -> {v_flat.shape}")

    return duplicated_caches

def beam_search(
        model,
        beam_width: int,
        sequence_length: int,
        x: torch.LongTensor,
        attn_mask: Optional[torch.BoolTensor],
        device: torch.device,
        pad_id: int | None = None,
) -> torch.LongTensor:
    """
    Performs batched beam search through the "GPT tree".

    Args:
    model: A transformer model.
    beam_width (int): The number of sequences to keep at each level.
    sequence_length (int): The total length of the sequence to be generated.
    x (tensor): The initial sequence [B, T].
    attn_mask: The attention mask [B, T].
    device: cuda device
    pad_id: Token ID for padding (optional, for constrained generation)

    Returns:
    tensor: All beam sequences [B, beam_width, T+sequence_length]
    """
    # Initialize batch size
    B = x.size(0)

    # Start with K=1 beam per example
    # sequences: [B, 1, T]
    sequences = x.unsqueeze(1)
    # scores: [B, 1]
    scores = torch.zeros(B, 1, device=device)

    # Initialize beam attention masks if provided
    if attn_mask is not None:
        # [B, T] -> [B, 1, T]
        beam_masks = attn_mask.unsqueeze(1)
    else:
        beam_masks = None

    # Log beam search start for multi-GPU debugging
    if 'RANK' in os.environ and B > 0:
        rank = int(os.environ['RANK'])
        print(f"Rank {rank}: Starting beam search - B={B}, beam_width={beam_width}, seq_len={sequence_length}, device={device}")

    for step in range(sequence_length):
        # Current shape
        B, K, t = sequences.shape

        # Flatten beams into batch dimension: [B*K, t]
        flat_seqs = sequences.view(B * K, t)
        flat_mask = (beam_masks.view(B * K, t) if beam_masks is not None else None)

        # One forward pass for all beams
        # Returns probs of shape [B*K, V]
        probs = model.generate_for_beam_search(flat_seqs, max_new_tokens=1, attn_mask=flat_mask)  # (B*K, V)
        logp = torch.log(probs.clamp(min=1e-10))  # safer clamping for log probabilities

        # Log progress for debugging (only occasionally to avoid spam)
        if 'RANK' in os.environ and step % 20 == 0 and step > 0:
            rank = int(os.environ['RANK'])
            print(f"Rank {rank}: Beam search progress - step {step}/{sequence_length}")

        # Reshape back to [B, K, V]
        V = logp.size(-1)
        logp = logp.view(B, K, V)

        # Compute new scores: [B, K, 1] + [B, K, V] = [B, K, V]
        new_scores = scores.unsqueeze(-1) + logp

        # For first step, only expand from first beam to avoid duplicates
        if step == 0 and K == 1:
            # Only consider expansions from the first beam
            new_scores[:, 1:, :] = float('-inf')

        # Flatten last two dims to select top beams: [B, K*V]
        flat_scores = new_scores.view(B, K * V)
        top_scores, top_idxs = flat_scores.topk(beam_width, dim=-1)

        # Decode beam and token indices
        beam_idxs = top_idxs // V  # [B, beam_width]
        token_idxs = top_idxs % V   # [B, beam_width]

        # Gather beam sequences
        sequences = sequences.gather(1, beam_idxs.unsqueeze(-1).expand(-1, -1, t))  # [B, beam_width, t]

        # Append the next token
        sequences = torch.cat([sequences, token_idxs.unsqueeze(-1)], dim=-1)  # [B, beam_width, t+1]

        # Update scores
        scores = top_scores

        # Update attention mask if used
        if beam_masks is not None:
            # Gather the selected beam masks
            beam_masks = beam_masks.gather(1, beam_idxs.unsqueeze(-1).expand(-1, -1, t))
            # Append True for new token
            new_mask = torch.ones(B, beam_width, 1, dtype=torch.bool, device=device)
            beam_masks = torch.cat([beam_masks, new_mask], dim=-1)

    # Return all beams (sorted by score, best first)
    return sequences


def beam_search_cached(
        model,
        beam_width: int,
        sequence_length: int,
        x: torch.LongTensor,
        attn_mask: Optional[torch.BoolTensor],
        device: torch.device,
        pad_id: int | None = None,
        debug: bool = False,
) -> torch.LongTensor:
    """
    Performs batched beam search with KV caching for efficiency.

    Args:
    model: A transformer model with generate_for_beam_search_cached method
    beam_width (int): The number of sequences to keep at each level.
    sequence_length (int): The total length of the sequence to be generated.
    x (tensor): The initial sequence [B, T].
    attn_mask: The attention mask [B, T].
    device: cuda device
    pad_id: Token ID for padding (optional, for constrained generation)
    debug: Enable debug logging

    Returns:
    tensor: All beam sequences [B, beam_width, T+sequence_length]
    """
    # Initialize batch size
    B = x.size(0)

    # Start with K=1 beam per example
    # sequences: [B, 1, T]
    sequences = x.unsqueeze(1)
    # scores: [B, 1]
    scores = torch.zeros(B, 1, device=device)

    # Initialize beam attention masks if provided
    if attn_mask is not None:
        # [B, T] -> [B, 1, T]
        beam_masks = attn_mask.unsqueeze(1)
    else:
        beam_masks = None

    # Initialize KV caches
    beam_caches = None

    # Log beam search start for multi-GPU debugging
    if 'RANK' in os.environ and B > 0:
        rank = int(os.environ['RANK'])
        print(f"Rank {rank}: Starting CACHED beam search - B={B}, beam_width={beam_width}, seq_len={sequence_length}, device={device}")

    for step in range(sequence_length):
        # Current shape
        B, K, t = sequences.shape

        if debug:
            print(f"[DEBUG] Step {step}/{sequence_length}: B={B}, K={K}, t={t}")

        # Flatten beams into batch dimension: [B*K, t]
        flat_seqs = sequences.view(B * K, t)
        flat_mask = (beam_masks.view(B * K, t) if beam_masks is not None else None)

        # Use cached generation method
        if debug:
            print(f"[DEBUG] Calling generate_for_beam_search_cached with flat_seqs shape {flat_seqs.shape}")

        probs, beam_caches = model.generate_for_beam_search_cached(
            flat_seqs,
            beam_caches=beam_caches,
            max_new_tokens=1,
            attn_mask=flat_mask,
            debug=debug
        )

        logp = torch.log(probs.clamp(min=1e-10))

        # Log progress for debugging (only occasionally to avoid spam)
        if 'RANK' in os.environ and step % 20 == 0 and step > 0:
            rank = int(os.environ['RANK'])
            print(f"Rank {rank}: CACHED beam search progress - step {step}/{sequence_length}")

        # Reshape back to [B, K, V]
        V = logp.size(-1)
        logp = logp.view(B, K, V)

        # Compute new scores: [B, K, 1] + [B, K, V] = [B, K, V]
        new_scores = scores.unsqueeze(-1) + logp

        # For first step, only expand from first beam to avoid duplicates
        if step == 0 and K == 1:
            # Only consider expansions from the first beam
            new_scores[:, 1:, :] = float('-inf')

        # Flatten last two dims to select top beams: [B, K*V]
        flat_scores = new_scores.view(B, K * V)
        top_scores, top_idxs = flat_scores.topk(beam_width, dim=-1)

        # Decode beam and token indices
        beam_idxs = top_idxs // V  # [B, beam_width]
        token_idxs = top_idxs % V   # [B, beam_width]

        if debug:
            print(f"[DEBUG] Step {step}: selected beam_idxs {beam_idxs[0]} for batch 0")

        # Critical: Update KV caches based on beam selection
        if beam_caches is not None:
            if debug:
                print(f"[DEBUG] Step {step}: Updating caches based on beam selection")
            beam_caches = _duplicate_beam_caches(beam_caches, beam_idxs, device, debug=debug)

        # Gather beam sequences
        sequences = sequences.gather(1, beam_idxs.unsqueeze(-1).expand(-1, -1, t))  # [B, beam_width, t]

        # Append the next token
        sequences = torch.cat([sequences, token_idxs.unsqueeze(-1)], dim=-1)  # [B, beam_width, t+1]

        # Update scores
        scores = top_scores

        # Update attention mask if used
        if beam_masks is not None:
            # Gather the selected beam masks
            beam_masks = beam_masks.gather(1, beam_idxs.unsqueeze(-1).expand(-1, -1, t))
            # Append True for new token
            new_mask = torch.ones(B, beam_width, 1, dtype=torch.bool, device=device)
            beam_masks = torch.cat([beam_masks, new_mask], dim=-1)

    if debug:
        print(f"[DEBUG] CACHED beam search completed. Final sequences shape: {sequences.shape}")

    # Return all beams (sorted by score, best first)
    return sequences
