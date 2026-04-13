import torch
import pytest
from src.gpt_pkg.model import GPT, GPTConfig
from src.gpt_pkg.beam_search import beam_search
from src.eval_pkg.majority_vote import majority_vote
from src.utils.permutation_utils import (
    generate_unique_permutations,
    permute_reads,
)


@pytest.fixture
def tiny_config():
    return GPTConfig(
        block_size=64,
        vocab_size=8,
        n_layer=1,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True,
    )


# Majority vote

def test_majority_vote_clear_winner():
    result = majority_vote(["ACGT", "ACGT", "TTTT"], length=4, check_length=3)
    assert result[0] == "A"  # 2 vs 1
    assert len(result) == 4


# Permutation ensemble

def test_permutation_pipeline_three_reads():
    """Generate all permutations of 3 reads and verify we get all 6 expected inputs."""
    src = "AA|BB|CC:GT"
    perms = generate_unique_permutations(3, 10, seed=42)
    assert len(perms) == 6  # 3! = 6
    results = {permute_reads(src, p) for p in perms}
    assert results == {
        "AA|BB|CC:GT",
        "AA|CC|BB:GT",
        "BB|AA|CC:GT",
        "BB|CC|AA:GT",
        "CC|AA|BB:GT",
        "CC|BB|AA:GT",
    }




# Model generation and masking (inference-time behavior)

def test_padded_generate_matches_unpadded(tiny_config):
    """Left-padded prompt should produce identical tokens as unpadded, with and without KV cache."""
    torch.manual_seed(0)
    model = GPT(tiny_config)
    model.eval()

    # unpadded
    prompt = torch.tensor([[0, 1, 2, 3, 6]])
    mask = torch.ones(1, 5, dtype=torch.bool)

    # left-padded (same real content)
    padded_prompt = torch.tensor([[7, 7, 7, 0, 1, 2, 3, 6]])
    padded_mask = torch.tensor([[False, False, False, True, True, True, True, True]])

    with torch.no_grad():
        out_cached = model.generate(prompt.clone(), mask.clone(), max_new_tokens=5, use_kv_cache=True)
        out_padded = model.generate(padded_prompt, padded_mask, max_new_tokens=5, use_kv_cache=True)
        out_nocache = model.generate(prompt.clone(), mask.clone(), max_new_tokens=5, use_kv_cache=False)

    generated_cached = out_cached[0, 5:].tolist()
    generated_padded = out_padded[0, 8:].tolist()
    generated_nocache = out_nocache[0, 5:].tolist()

    # all three should produce the same tokens
    assert generated_cached == generated_padded, "padding changed output"
    assert generated_cached == generated_nocache, "KV cache diverged from full forward"
    # tokens should be valid
    assert (out_cached >= 0).all() and (out_cached < 8).all()


def test_beam_search_width1_matches_greedy(tiny_config):
    """Beam search with width=1 should produce the same tokens as greedy generation."""
    torch.manual_seed(0)
    model = GPT(tiny_config)
    model.eval()

    prompt = torch.tensor([[0, 1, 2, 3, 6]])  # "ACGT:"
    mask = torch.ones(1, 5, dtype=torch.bool)

    with torch.no_grad():
        greedy_out = model.generate(prompt.clone(), mask.clone(), max_new_tokens=5)
        beam_out = beam_search(model, beam_width=1, sequence_length=5, x=prompt.clone(), attn_mask=mask.clone(), device=torch.device('cpu'))

    # beam_out is (B, beam_width, T+seq_len) -> take best beam
    assert greedy_out[0, 5:].tolist() == beam_out[0, 0, 5:].tolist()
