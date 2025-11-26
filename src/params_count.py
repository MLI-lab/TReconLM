import os
from gpt_pkg.model import GPTConfig, GPT
from rnn_pkg.lstm_model import LSTMConfig, LSTMConsensus


###################### GPT PARAM COUNT ######################

def exact_param_count_gpt(n_layer, n_head, n_embd, non_embedding=True):
    cfg = GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd)
    model = GPT(cfg)
    return model.get_num_params(non_embedding=non_embedding)


###################### LSTM PARAM COUNT ######################

def exact_lstm_param_count(n_layer, n_embd, vocab_size=8, dropout=0.0, non_embedding=True):
    cfg = LSTMConfig(vocab_size=vocab_size, n_layer=n_layer, n_embd=n_embd, dropout=dropout)
    pad_token_id = vocab_size - 1  # Use last token as padding
    model = LSTMConsensus(cfg, pad_token_id)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if non_embedding:
        embedding_params = sum(p.numel() for p in model.embed.parameters() if p.requires_grad)
        return total_params - embedding_params
    return total_params


###################### MAMBA PARAM COUNT ######################

def exact_mamba_param_count(
    d_model,
    n_layer,
    vocab_size,
    ssm_cfg,
    d_intermediate=0,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    pad_vocab_size_multiple=8,
    tie_embeddings=True,
    non_embedding=True,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Avoid loading CUDA kernels
    os.environ["MAMBA_DISABLE_TRITON"] = "1"  # Force CPU fallback

    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_pkg.my_mamba_model import MambaLMHeadModel

    config = MambaConfig(
        d_model=d_model,
        n_layer=n_layer,
        d_intermediate=d_intermediate,
        vocab_size=vocab_size,
        ssm_cfg=ssm_cfg,
        rms_norm=rms_norm,
        residual_in_fp32=residual_in_fp32,
        fused_add_norm=fused_add_norm,
        pad_vocab_size_multiple=pad_vocab_size_multiple,
        tie_embeddings=tie_embeddings,
    )

    model = MambaLMHeadModel(config)
    return model.get_num_params(non_embedding=non_embedding)


###################### GPT/MAMBA TRAINING FLOPs ######################

def compute_training_flops(param_count, batch_size, context_length, num_iterations):
    """
    Compute training FLOPs using Kaplan et al. (2020) scaling law:
    
        C = 6 × N × D
        N = number of parameters (non-embedding)
        D = total number of tokens processed
    
    Reference:
        "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)
    """
    tokens = batch_size * context_length * num_iterations
    return 6 * param_count * tokens


###################### LSTM TRAINING FLOPs ######################

def compute_lstm_training_flops(
    n_layer, input_dim, hidden_dim, vocab_size,
    batch_size, seq_len, num_iterations
):
    """
    Compute total training FLOPs for an LSTM language model.
    
    Formula from Zhang et al. (2018), 
    'Fast Neural Network Decoding for Sequence Models' (NeurIPS 2018).

    Forward FLOPs per LSTM step:
        F_LSTM = 2 × n_layer × (input_dim + hidden_dim) × hidden_dim × 4 × 2
    Softmax FLOPs per step:
        F_softmax = hidden_dim × vocab_size × 2

    Training FLOPs:
        F_total = 3 × (F_LSTM + F_softmax) × batch_size × seq_len × num_iterations
    """
    F_LSTM = 2 * n_layer * (input_dim + hidden_dim) * hidden_dim * 4 * 2
    F_softmax = hidden_dim * vocab_size * 2
    F_per_step = 3 * (F_LSTM + F_softmax)
    return F_per_step * batch_size * seq_len * num_iterations


###################### MAIN ######################

def format_scientific(flops):
    """Format a number as a×10^b style for FLOPs."""
    import math
    if flops == 0:
        return "0"
    exponent = int(math.floor(math.log10(flops)))
    mantissa = flops / 10**exponent
    return f"{mantissa:.2f}×10^{exponent} FLOPs"


if __name__ == "__main__":

    # Hyperparameters
    batch_size = 16
    context_length = 1500
    num_iterations = 98114
    vocab_size = 8

    # === GPT ===
    gpt_params = exact_param_count_gpt(n_layer=4, n_head=6, n_embd=384, non_embedding=True)
    gpt_flops = compute_training_flops(gpt_params, batch_size, context_length, num_iterations)

    # === LSTM ===
    lstm_params = exact_lstm_param_count(vocab_size=vocab_size, n_layer=4, n_embd=384, dropout=0.0, non_embedding=True)
    lstm_flops = compute_lstm_training_flops(
        n_layer=4, input_dim=384, hidden_dim=384,
        vocab_size=vocab_size, batch_size=batch_size,
        seq_len=context_length, num_iterations=num_iterations
    )

    # === MAMBA ===
    mamba_cfg = {
        "d_state": 32,
        "n_layer": 4,
        "d_model": 384,
        "d_intermediate": 1536,
        "rms_norm": True,
        "residual_in_fp32": False,
        "fused_add_norm": True,
        "pad_vocab_size_multiple": 8,
        "tie_embeddings": True,
        "ssm_cfg": {
            "d_state": 32,
            "d_conv": 4,
            "expand": 2,
            "dt_min": 0.001,
            "dt_max": 0.1,
            "dt_init_floor": 1e-4,
            "conv_bias": True,
            "bias": False,
            "layer": "Mamba2"
        }
    }

    mamba_params = exact_mamba_param_count(
        d_model=mamba_cfg["d_model"],
        n_layer=mamba_cfg["n_layer"],
        ssm_cfg=mamba_cfg["ssm_cfg"],
        d_intermediate=mamba_cfg["d_intermediate"],
        rms_norm=mamba_cfg["rms_norm"],
        residual_in_fp32=mamba_cfg["residual_in_fp32"],
        fused_add_norm=mamba_cfg["fused_add_norm"],
        pad_vocab_size_multiple=mamba_cfg["pad_vocab_size_multiple"],
        tie_embeddings=mamba_cfg["tie_embeddings"],
        non_embedding=True,
        vocab_size=vocab_size
    )

    mamba_flops = compute_training_flops(mamba_params, batch_size, context_length, num_iterations)

    # === PRINT RESULTS ===
    print("\n===== PARAM COUNT & TRAINING FLOPs =====")
    print(f"GPT   | Params: {gpt_params/1e6:.2f}M | FLOPs: {format_scientific(gpt_flops)}")
    print(f"LSTM  | Params: {lstm_params/1e6:.2f}M | FLOPs: {format_scientific(lstm_flops)}")
    print(f"Mamba | Params: {mamba_params/1e6:.2f}M | FLOPs: {format_scientific(mamba_flops)}")
