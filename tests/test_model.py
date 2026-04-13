import torch
import pytest
from src.gpt_pkg.model import GPT, GPTConfig


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


def test_forward_inference_shape(tiny_config):
    model = GPT(tiny_config)
    model.eval()
    x = torch.randint(0, 8, (2, 10))  # batch=2, seq_len=10
    with torch.no_grad():
        logits, loss = model(x)
    # inference: returns logits at last position -> (B, 1, vocab_size)
    assert logits.shape == (2, 1, 8)
    assert loss is None


def test_forward_respects_block_size(tiny_config):
    model = GPT(tiny_config)
    model.eval()
    # sequence longer than block_size should raise
    x = torch.randint(0, 8, (1, 65))  # block_size=64
    with pytest.raises(AssertionError):
        model(x)


# Training

def test_training_loop_loss_decreases(tiny_config):
    """A few training steps on CPU should reduce the loss."""
    stoi = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4, '|': 5, ':': 6, '#': 7}

    torch.manual_seed(42)
    model = GPT(tiny_config)
    model.train()

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type='cpu',
    )

    # minimal batch: 4 samples of "reads:ground_truth#" pattern
    batch = torch.tensor([
        [0, 1, 2, 3, 5, 3, 2, 1, 0, 6, 0, 1, 2, 3, 7],  # ACGT|TGCA:ACGT#
        [2, 2, 1, 0, 5, 0, 1, 2, 3, 6, 2, 2, 1, 0, 7],  # GGCA|ACGT:GGCA#
        [3, 3, 3, 3, 5, 0, 0, 0, 0, 6, 3, 3, 3, 3, 7],  # TTTT|AAAA:TTTT#
        [1, 0, 2, 3, 5, 1, 0, 2, 3, 6, 1, 0, 2, 3, 7],  # CAGT|CAGT:CAGT#
    ])

    losses = []
    for _ in range(20):
        _, loss = model(batch, targets=batch, stoi=stoi)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
