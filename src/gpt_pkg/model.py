"""
Full definition of a GPT Language Model, all of it in this single file with batched inference via masking out padding tokens. 
Also supports caching attention scores for faster inference. 
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import os

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):

        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.config = config
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # always register the causal mask buffer
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size),
            persistent=False          
        )
        
        if not hasattr(CausalSelfAttention, "warned"):
            if self.flash:
                print("FlashAttention available (will be used during training and inference with KV cache)")
            else:
                print("FlashAttention not available - requires PyTorch >= 2.0 (using manual attention)")
            CausalSelfAttention.warned = True

    def forward(self, x, attn_mask=None, cache=None, store_attention=False):
        """
        x: (B, T, C)
        attn_mask: (B, T)  or None
        cache: None or (k_prev, v_prev) each (B, h, T_prev, d)
        store_attention: bool, whether to store attention weights for analysis
        """
        B, T, C = x.size()
        # project QKV
        qkv = self.c_attn(x)                   # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # reshape into heads
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)  # (B,h,T,d)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)

        # prepend cache if present
        if cache is not None:
            k_prev, v_prev = cache
            k = torch.cat([k_prev, k], dim=2)  # (B,h,T_prev+T,d)
            v = torch.cat([v_prev, v], dim=2)

        # crop to block_size so mask/bias never overflow
        block_size = self.config.block_size
        if k.size(2) > block_size:
            k = k[:, :, -block_size:, :]
            v = v[:, :, -block_size:, :]
            if attn_mask is not None:
                attn_mask = attn_mask[:, -block_size:]

        new_cache = (k, v)
        T_all     = k.size(2)

        assert k.size(2) <= block_size and v.size(2) == k.size(2)
        assert q.size(2) == T  # B,h,T,d

        # Use Flash Attention when available (unless we need to store attention weights)
        if self.flash and not store_attention:
            if attn_mask is None:
                # No padding mask - just use causal masking
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True
                )
            else:
                # Have padding mask - need to combine with causal mask
                # Build boolean mask: True = attend, False = masked
                # Start with causal mask
                causal_mask = self.bias[:, :, T_all - T : T_all, :T_all]  # (1, 1, T, T_all), 1=attend 0=masked

                # Expand padding mask: True for real tokens, False for padding
                pad_mask = attn_mask[:, None, None, :T_all]  # (B, 1, 1, T_all)

                # Combine: can attend if BOTH causal allows AND key is not padding
                combined_mask = (causal_mask.bool()) & (pad_mask)  # (B, 1, T, T_all)

                # Important: ensure each query can attend to at least one position
                # to avoid NaN in softmax. For padding queries, allow self-attention.
                # This doesn't affect real tokens since their outputs are used, but
                # prevents NaN for padding positions.
                diagonal_mask = torch.eye(T_all, dtype=torch.bool, device=combined_mask.device)
                diagonal_mask = diagonal_mask[T_all - T : T_all, :T_all]  # Grab last T rows
                diagonal_mask = diagonal_mask[None, None, :, :]  # Add batch and head dims
                combined_mask = combined_mask | diagonal_mask  # Allow diagonal

                # Convert to additive mask: -inf for masked, 0.0 for attend
                flash_mask = torch.zeros_like(combined_mask, dtype=q.dtype)
                flash_mask.masked_fill_(~combined_mask, float('-inf'))

                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=flash_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False  # Already handled in flash_mask
                )
        else:
            # Manual attention (fallback when storing attention weights)
            att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))

            # grab the last T rows of the block‐size mask
            # so that even when T=1 you attend to all previous keys:
            bias = self.bias[:, :, T_all - T : T_all, :T_all]    # (1,1,T,T_all)
            att  = att.masked_fill(bias == 0, float('-inf'))

            # then apply any padding mask over those T_all keys
            if attn_mask is not None:
                pad_mask = (~attn_mask)[:, None, None, :T_all]
                att = att.masked_fill(pad_mask, float('-inf'))

            att = F.softmax(att, dim=-1)
            att = torch.nan_to_num(att, nan=0.0)

            # Store attention weights if requested (before dropout)
            if store_attention:
                self.last_attention_weights = att.detach().clone()

            att = self.attn_dropout(att)
            y   = att @ v   # (B,h,T,d)

        # combine heads
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_cache


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x, attn_mask=None, cache=None, store_attention=False):
        # layer‐norm and attention with caching
        attn_input = self.ln_1(x)
        attn_out, new_cache = self.attn(attn_input, attn_mask=attn_mask, cache=cache, store_attention=store_attention)
        x = x + attn_out

        # feed-forward
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache

@dataclass
class GPTConfig:
    """
    Defines default GPT model hyperparameters, which are typically overridden by values provided via Hydra configs.
    """
    block_size: int = 600 # in gpt-2 original 1024
    vocab_size: int = 8 # Here its 8, in GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    label_smoothing: float = 0.0

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__() #initialize parent class to function as PyTorch module
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd), # Token embeddings
            wpe  = nn.Embedding(config.block_size, config.n_embd), # Positional embeddings
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Block contains Self-attention, multi-head attention, feedforward and residual connections
            ln_f = LayerNorm(config.n_embd, bias=config.bias), # Final LayerNorm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # Maps hidden states back to token vocabulary (logits over vocab)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying.  Since embeddings map tokens to vectors, and lm_head maps vectors back to tokens, they can be the same matrix transposed.

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        #print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            #print('positional embeddings:', self.transformer.wpe.weight.numel())
            #n_params -= self.transformer.wte.weight.numel()
            #print('token embeddings:', self.transformer.wte.weight.numel())
        return n_params

    def _init_weights(self, module):
        """ Initalizes weights and embedding vectors from a normal distribution with biases initialized to zero."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attn_mask=None, targets=None, stoi=None):
        """
        input_ids : (B, T) LongTensor
        attn_mask : (B, T) BoolTensor, True = real token, False = pad
        targets   : optional (B, T) LongTensor for loss
        """
        idx = input_ids
        device = idx.device
        b, t = idx.size()

        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only "
            f"{self.config.block_size}"
        )

        # embeddings
        if attn_mask is None:                                # right-padding case or training
            pos = torch.arange(t, device=device)[None, :].expand(b, -1)
        else:                                                # left-padding aware
            pos = (attn_mask.long().cumsum(-1) - 1).clamp(min=0)


        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)


        # transformer stack 
        for block in self.transformer.h:
            x, _ = block(x, attn_mask=attn_mask) 
        x = self.transformer.ln_f(x)                       # (B, T, E)

        # training branch 
        if targets is not None and stoi is not None:
            logits_full = self.lm_head(x)                  # (B, T, V)

            losses = []
            colon_indices = (targets == stoi[":"]).nonzero(as_tuple=True)
            for batch_index, colon_position in zip(*colon_indices):
                try:
                    end_position = (
                        (targets[batch_index] == stoi["#"]).nonzero(as_tuple=True)[0][0]
                        + 1
                    )
                except IndexError:
                    end_position = targets.size(1)

                logits_slice  = logits_full[batch_index, colon_position + 1 : end_position]
                targets_slice = targets      [batch_index, colon_position + 1 : end_position]

                losses.append(
                    F.cross_entropy(
                        logits_slice,
                        targets_slice,
                        label_smoothing=self.config.label_smoothing,
                    )
                )

            loss = torch.stack(losses).mean()
            return logits_full, loss          

        #  inference
        logits_full = self.lm_head(x)                          # (B, T, V)

        if attn_mask is not None:
            # count real tokens per sample, last valid index
            last_idx = attn_mask.long().sum(1) - 1             # (B,)
        else:
            # no mask: assume every sequence is full length
            last_idx = torch.full((b,), t - 1, device=device)  # (B,)

        # gather the logits at those indices, keep time-dim of size 1
        logits = logits_full[torch.arange(b, device=device), last_idx][:, None, :]
        return logits, None


    def crop_block_size(self, block_size):
        """
        Adjusts the context length (maximum number of tokens the model can handle).
        Useful when loading a large model but needing a smaller sequence length.
        Allows the model to process shorter sequences efficiently while retaining the rest of the pretrained weights.
        """
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size]) # Truncates the positional embeddings to fit the new block size.
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size] # Truncates attention bias tensors (used for masking).

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Loads a Hugging Face GPT-2 model and adapts it to a custom GPT class.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        # Forces model configurations to match OpenAI’s original GPT-2 settings.
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # Creates a new instance of the custom GPT model using the same configuration as GPT-2. 
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model # Returns the fully initialized and pretrained GPT model.

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configures the optimizer for training the model. 
        It defines which parameters should have weight decay, sets up optimization groups, and initializes AdamW with the correct settings.
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate Model FLOPs Utilization (MFU) as a fraction of A100 peak FLOPs. See PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311"""
        flops_achieved = self.estimate_flops_per_iter(fwdbwd_per_iter, dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak FLOPS is 312 TFLOPS
        return flops_achieved / flops_promised

    
    def estimate_flops_per_iter(self, fwdbwd_per_iter,dt):

        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)

        return flops_achieved

    @torch.no_grad()
    def generate(self, idx: torch.LongTensor, attn_mask: torch.BoolTensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None, itos=None) -> torch.LongTensor:
        """
        Greedy (or top-k) token generation using per-layer KV cache.
        Only the new token is fed through each transformer block.
        """
        device = idx.device
        # initialize empty cache for each transformer layer
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.transformer.h)

        for _ in range(max_new_tokens):
            # crop to block_size if needed
            if idx.size(1) > self.config.block_size:
                idx       = idx[:, -self.config.block_size :]
                attn_mask = attn_mask[:, -self.config.block_size :]
                for i in range(len(caches)):
                    caches[i] = _crop_kv_cache(caches[i], self.config.block_size)

            # isolate last token and its mask
            last_token = idx[:, -1:].to(device)       # (B,1)

            # compute positional index for that token
            pos = (attn_mask.long().cumsum(-1) - 1).clamp(min=0)
            last_pos = pos[:, -1:]          # (B,1,1)

            # embed and dropout
            x = self.transformer.wte(last_token) + self.transformer.wpe(last_pos)
            x = self.transformer.drop(x)

            # pass through each block with its cache
            for i, block in enumerate(self.transformer.h):
                x, caches[i] = block(x, attn_mask=attn_mask, cache=caches[i]) # not sure if correctly update 
            # final layer norm
            x = self.transformer.ln_f(x)  # (B,1,C)

            logits = self.lm_head(x)[:, 0, :] / temperature  # (B, V)

            # optional top-k pruning
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)

            # greedy pick
            next_idx = torch.argmax(probs, dim=-1, keepdim=True)

            # append token & update mask
            idx       = torch.cat((idx, next_idx), dim=1)
            # append a “real”-token flag for that new token
            attn_mask = torch.cat([
                attn_mask,
                torch.ones((attn_mask.size(0), 1), dtype=torch.bool, device=attn_mask.device)
            ], dim=1)

        return idx



    @torch.no_grad()
    def generate_cpred(
        self,
        idx:        torch.LongTensor,     # (B, T_ctx)   prompt incl. ':'
        attn_mask:  torch.BoolTensor,     # (B, T_ctx)   True = real token
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        sampling: str = "greedy",         # or "sample"
        constrained_generation: bool = False,
        itos=None,
        collect_logits: bool = False,    # NEW: collect logits during generation
        return_hidden_states: bool = False,  # NEW: return hidden states from last layer
        debug: bool = False,  # Enable detailed logging
    ):
        device, B = idx.device, idx.size(0)

        if debug:
            print(f"\n[DEBUG generate_cpred] Starting generation")
            print(f"  Initial idx shape: {idx.shape}")
            print(f"  max_new_tokens: {max_new_tokens}")
            print(f"  constrained_generation: {constrained_generation}")

        # Initialize logits collection if requested
        collected_logits = [] if collect_logits else None

        # Initialize hidden state collection if requested
        collected_hidden_states = [] if return_hidden_states else None

        # truncate prompt if too long
        if idx.size(1) > self.config.block_size:
            idx       = idx[:, -self.config.block_size :]
            attn_mask = attn_mask[:, -self.config.block_size :]

        # compute positions, embed, and do a full forward to build caches
        pos = (attn_mask.long().cumsum(-1) - 1).clamp(min=0)          # (B, T_ctx)
        x   = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x   = self.transformer.drop(x)

        caches = []
        for block in self.transformer.h:
            x, cache = block(x, attn_mask=attn_mask, cache=None)
            caches.append(cache)
        x = self.transformer.ln_f(x)                                   # (B, T_ctx, C)

        # Autoregressive loop
        for step_idx in range(max_new_tokens):
            # collect hidden states before lm_head (last layer representations)
            if return_hidden_states:
                # x is the output after layer norm, shape (B, T, C)
                # Take the last position's hidden state
                hidden_state = x[:, -1, :]                              # (B, C)
                collected_hidden_states.append(hidden_state.unsqueeze(1))  # (B, 1, C)

            # compute logits on the last position
            logits = self.lm_head(x)[:, -1, :] / temperature            # (B, V)

            if debug and step_idx < 3:  # Only log first 3 steps
                print(f"\n[DEBUG] Step {step_idx}:")
                print(f"  x shape: {x.shape}")
                print(f"  logits: {logits[0]}")
                print(f"  logits argmax: {logits.argmax(dim=-1).item()} ({itos[logits.argmax(dim=-1).item()] if itos else '?'})")

            # collect logits before any modifications (for vote confidence analysis)
            if collect_logits:
                # Store the raw logits (before temperature scaling for consistency)
                raw_logits = self.lm_head(x)[:, -1, :]                  # (B, V)
                collected_logits.append(raw_logits.unsqueeze(1))        # (B, 1, V)

            # top-k pruning, if in config
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            # pick next token
            if sampling == "greedy":
                next_idx = torch.argmax(probs, dim=-1, keepdim=True)
                # optionally enforce DNA letters only
                if constrained_generation and itos is not None:
                    flat = next_idx.squeeze(1)
                    for i in range(B):
                        if itos[flat[i].item()] not in {"A", "C", "G", "T"}:
                            sorted_probs, sorted_idx = torch.sort(probs[i], descending=True)
                            for alt in sorted_idx[1:]:
                                if itos[alt.item()] in {"A", "C", "G", "T"}:
                                    flat[i] = alt
                                    break
                    next_idx = flat.unsqueeze(1)
            else:
                next_idx = torch.multinomial(probs, num_samples=1)

            # append new token and update mask
            idx       = torch.cat([idx, next_idx], dim=1)               # (B, T+1)
            # append a “real”-token flag for that new token
            attn_mask = torch.cat([
                attn_mask,
                torch.ones((attn_mask.size(0), 1), dtype=torch.bool, device=attn_mask.device)
            ], dim=1)

            # handle overflow 
            if idx.size(1) > self.config.block_size:
                idx       = idx[:, -self.config.block_size :]
                attn_mask = attn_mask[:, -self.config.block_size :]
                for i in range(len(caches)):
                    caches[i] = _crop_kv_cache(caches[i], self.config.block_size)

            # build embedding for just that new token
            last_pos = (attn_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -1:]
            x = self.transformer.wte(next_idx) + self.transformer.wpe(last_pos)
            x = self.transformer.drop(x)

            # step through each block with its cache
            for i, block in enumerate(self.transformer.h):
                x, caches[i] = block(x, attn_mask=attn_mask, cache=caches[i])

            # final layer-norm readied for the next iteration
            x = self.transformer.ln_f(x)                               # (B, 1, C)

        # Return results
        if collect_logits and return_hidden_states:
            # Return everything
            logits_tensor = torch.cat(collected_logits, dim=1)
            hidden_states_tensor = torch.cat(collected_hidden_states, dim=1)
            return idx, logits_tensor, hidden_states_tensor
        elif collect_logits:
            # Stack collected logits: (B, max_new_tokens, V)
            logits_tensor = torch.cat(collected_logits, dim=1)
            return idx, logits_tensor
        elif return_hidden_states:
            # Stack collected hidden states: (B, max_new_tokens, C)
            hidden_states_tensor = torch.cat(collected_hidden_states, dim=1)
            return idx, hidden_states_tensor
        else:
            return idx

    @torch.no_grad()
    def generate_cpred_with_entropy(
        self,
        idx:        torch.LongTensor,     # (B, T_ctx)   prompt incl. ':'
        attn_mask:  torch.BoolTensor,     # (B, T_ctx)   True = real token
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        sampling: str = "greedy",         # or "sample"
        constrained_generation: bool = False,
        itos=None,
        track_entropy: bool = False
    ):
        """
        Generate tokens with optional entropy tracking.
        Returns: (generated_ids, token_entropies)
        """
        device, B = idx.device, idx.size(0)

        # Storage for entropy if tracking
        all_entropies = [] if track_entropy else None

        # truncate prompt if too long
        if idx.size(1) > self.config.block_size:
            idx       = idx[:, -self.config.block_size :]
            attn_mask = attn_mask[:, -self.config.block_size :]

        # compute positions, embed, and do a full forward to build caches
        pos = (attn_mask.long().cumsum(-1) - 1).clamp(min=0)          # (B, T_ctx)
        x   = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x   = self.transformer.drop(x)

        caches = []
        for block in self.transformer.h:
            x, cache = block(x, attn_mask=attn_mask, cache=None)
            caches.append(cache)
        x = self.transformer.ln_f(x)                                   # (B, T_ctx, C)

        # Autoregressive loop
        for _ in range(max_new_tokens):
            # compute logits on the last position
            logits = self.lm_head(x)[:, -1, :] / temperature            # (B, V)

            # Calculate entropy before sampling (this is key!)
            if track_entropy:
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                all_entropies.append(entropy.cpu().numpy())

            # top-k pruning, if in config
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            # pick next token
            if sampling == "greedy":
                next_idx = torch.argmax(probs, dim=-1, keepdim=True)
                # optionally enforce DNA letters only
                if constrained_generation and itos is not None:
                    flat = next_idx.squeeze(1)
                    for i in range(B):
                        if itos[flat[i].item()] not in {"A", "C", "G", "T"}:
                            sorted_probs, sorted_idx = torch.sort(probs[i], descending=True)
                            for alt in sorted_idx[1:]:
                                if itos[alt.item()] in {"A", "C", "G", "T"}:
                                    flat[i] = alt
                                    break
                    next_idx = flat.unsqueeze(1)
            else:
                next_idx = torch.multinomial(probs, num_samples=1)

            # append new token and update mask
            idx       = torch.cat([idx, next_idx], dim=1)               # (B, T+1)
            # append a "real"-token flag for that new token
            attn_mask = torch.cat([
                attn_mask,
                torch.ones((attn_mask.size(0), 1), dtype=torch.bool, device=attn_mask.device)
            ], dim=1)

            # handle overflow
            if idx.size(1) > self.config.block_size:
                idx       = idx[:, -self.config.block_size :]
                attn_mask = attn_mask[:, -self.config.block_size :]
                for i in range(len(caches)):
                    caches[i] = _crop_kv_cache(caches[i], self.config.block_size)

            # build embedding for just that new token
            last_pos = (attn_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -1:]
            x = self.transformer.wte(next_idx) + self.transformer.wpe(last_pos)
            x = self.transformer.drop(x)

            # step through each block with its cache
            for i, block in enumerate(self.transformer.h):
                x, caches[i] = block(x, attn_mask=attn_mask, cache=caches[i])

            # final layer-norm readied for the next iteration
            x = self.transformer.ln_f(x)                               # (B, 1, C)

        # Process entropy data
        if track_entropy:
            # Shape: [max_new_tokens, B] -> [B, max_new_tokens]
            all_entropies = np.array(all_entropies).T

        return idx, all_entropies

    @torch.no_grad()
    def generate_cpred_with_entropy_and_attention(
        self,
        idx: torch.LongTensor,     # (B, T_ctx)   prompt incl. ':'
        attn_mask: torch.BoolTensor,     # (B, T_ctx)   True = real token
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        sampling: str = "greedy",
        constrained_generation: bool = False,
        itos=None,
        track_entropy: bool = False,
        track_attention: bool = False,
        track_all_layers: bool = False,
        read_boundaries: list = None  # Only used for validation, not in generation
    ):
        """
        Generate tokens with entropy and detailed attention tracking.

        Note on attention timing:
        The attention weights captured show what the model attends to when PROCESSING
        each newly generated token, not when producing its logits. 

        Note on memory:
        Attention weights are moved to CPU immediately to prevent GPU memory accumulation
        during long generation sequences.

        Args:
            read_boundaries: List of (start, end) positions - used only for postprocessing validation
        Returns:
            (generated_ids, token_entropies, attention_sequence)
            - attention_sequence: List of [B, num_heads, 1, seq_len] CPU tensors, one per generated token
        """
        device, B = idx.device, idx.size(0)

        # Storage for tracking
        all_entropies = [] if track_entropy else None
        # If tracking all layers, use dict: {layer_idx: [attention_weights_per_step]}
        # Otherwise, just list: [attention_weights_per_step] for last layer only
        if track_attention and track_all_layers:
            attention_sequence = {i: [] for i in range(len(self.transformer.h))}
        elif track_attention:
            attention_sequence = []
        else:
            attention_sequence = None

        # truncate prompt if too long
        if idx.size(1) > self.config.block_size:
            idx       = idx[:, -self.config.block_size :]
            attn_mask = attn_mask[:, -self.config.block_size :]

        # compute positions, embed, and do a full forward to build caches
        pos = (attn_mask.long().cumsum(-1) - 1).clamp(min=0)          # (B, T_ctx)
        x   = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x   = self.transformer.drop(x)

        caches = []
        for block in self.transformer.h:
            x, cache = block(x, attn_mask=attn_mask, cache=None)
            caches.append(cache)
        x = self.transformer.ln_f(x)                                   # (B, T_ctx, C)

        # Autoregressive loop
        for step in range(max_new_tokens):
            # compute logits on the last position
            logits = self.lm_head(x)[:, -1, :] / temperature            # (B, V)

            # Calculate entropy before sampling
            if track_entropy:
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                all_entropies.append(entropy.cpu().numpy())

            # top-k pruning, if in config
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            # pick next token
            if sampling == "greedy":
                next_idx = torch.argmax(probs, dim=-1, keepdim=True)
                # optionally enforce DNA letters only
                if constrained_generation and itos is not None:
                    flat = next_idx.squeeze(1)
                    for i in range(B):
                        if itos[flat[i].item()] not in {"A", "C", "G", "T"}:
                            sorted_probs, sorted_idx = torch.sort(probs[i], descending=True)
                            for alt in sorted_idx[1:]:
                                if itos[alt.item()] in {"A", "C", "G", "T"}:
                                    flat[i] = alt
                                    break
                    next_idx = flat.unsqueeze(1)
            else:
                next_idx = torch.multinomial(probs, num_samples=1)

            # append new token and update mask
            idx       = torch.cat([idx, next_idx], dim=1)               # (B, T+1)
            attn_mask = torch.cat([
                attn_mask,
                torch.ones((attn_mask.size(0), 1), dtype=torch.bool, device=attn_mask.device)
            ], dim=1)

            # handle overflow
            if idx.size(1) > self.config.block_size:
                idx       = idx[:, -self.config.block_size :]
                attn_mask = attn_mask[:, -self.config.block_size :]
                for i in range(len(caches)):
                    caches[i] = _crop_kv_cache(caches[i], self.config.block_size)

            # build embedding for just that new token
            last_pos = (attn_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -1:]
            x = self.transformer.wte(next_idx) + self.transformer.wpe(last_pos)
            x = self.transformer.drop(x)

            # step through each block with its cache
            for i, block in enumerate(self.transformer.h):
                # Capture attention from all layers or just last layer
                if track_all_layers:
                    store_attention = track_attention
                else:
                    store_attention = track_attention and (i == len(self.transformer.h) - 1)

                x, caches[i] = block(x, attn_mask=attn_mask, cache=caches[i], store_attention=store_attention)

                # Store attention weights for processing this newly added token
                if store_attention and hasattr(block.attn, 'last_attention_weights'):
                    # This shows what the model attends to when processing the newly generated token
                    # Move to CPU immediately to avoid GPU memory accumulation
                    attn_weights = block.attn.last_attention_weights.clone().cpu()

                    if track_all_layers:
                        # Store in dict with layer index
                        attention_sequence[i].append(attn_weights)
                    else:
                        # Store in list (last layer only)
                        attention_sequence.append(attn_weights)

            # final layer-norm readied for the next iteration
            x = self.transformer.ln_f(x)                               # (B, 1, C)

        # Process tracking data
        if track_entropy:
            all_entropies = np.array(all_entropies).T

        return idx, all_entropies, attention_sequence


    @torch.no_grad()
    def generate_for_beam_search(self, idx, max_new_tokens=1, temperature=1.0, top_k=None, attn_mask=None):
        for _ in range(max_new_tokens):
            if idx.size(1) > self.config.block_size:
                idx_cond = idx[:, -self.config.block_size:]
                am_cond  = attn_mask[:, -self.config.block_size:] if attn_mask is not None else None
            else:
                idx_cond, am_cond = idx, attn_mask

            pos = (am_cond.long().cumsum(-1) - 1).clamp(min=0) if am_cond is not None else torch.arange(idx_cond.size(1), device=idx.device)[None,:].expand(idx_cond.size(0), -1)

            x = self.transformer.wte(idx_cond) + self.transformer.wpe(pos)
            x = self.transformer.drop(x)
            for block in self.transformer.h:
                x, _ = block(x, attn_mask=am_cond, cache=None)
            x = self.transformer.ln_f(x)

            logits_full = self.lm_head(x)
            logits = logits_full[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
        return probs

    @torch.no_grad()
    def generate_for_beam_search_cached(self, idx, beam_caches=None, max_new_tokens=1, temperature=1.0, top_k=None, attn_mask=None, debug=False):
        """
        Generate one token for beam search with KV caching support.

        Args:
            idx: Input tensor [B*K, T] where B=batch_size, K=beam_width
            beam_caches: List of caches for each beam [beam_idx][layer_idx] = (k, v)
            max_new_tokens: Should always be 1 for beam search
            temperature: Temperature for sampling
            top_k: Top-k filtering
            attn_mask: Attention mask [B*K, T]
            debug: Enable debug logging

        Returns:
            probs: Token probabilities [B*K, V]
            updated_beam_caches: Updated caches for each beam
        """
        assert max_new_tokens == 1, "Beam search should only generate 1 token at a time"

        device = idx.device
        batch_beam_size, seq_len = idx.shape

        if debug:
            print(f"[DEBUG] generate_for_beam_search_cached: input shape {idx.shape}, beam_caches is {'None' if beam_caches is None else 'provided'}")

        # Handle block size limits
        if seq_len > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            attn_mask = attn_mask[:, -self.config.block_size:] if attn_mask is not None else None
            seq_len = self.config.block_size
            if debug:
                print(f"[DEBUG] Cropped to block_size: new seq_len = {seq_len}")

        # Initialize caches if first call
        if beam_caches is None:
            if debug:
                print("[DEBUG] Initializing caches - doing full forward pass")
            # Full forward pass to initialize caches
            pos = (attn_mask.long().cumsum(-1) - 1).clamp(min=0) if attn_mask is not None else torch.arange(seq_len, device=device)[None,:].expand(batch_beam_size, -1)
            x = self.transformer.wte(idx) + self.transformer.wpe(pos)
            x = self.transformer.drop(x)

            beam_caches = []
            for i, block in enumerate(self.transformer.h):
                x, cache = block(x, attn_mask=attn_mask, cache=None)
                beam_caches.append(cache)
                if debug and i == 0:
                    k, v = cache
                    print(f"[DEBUG] Layer {i} cache shapes: k={k.shape}, v={v.shape}")

        else:
            if debug:
                print("[DEBUG] Using existing caches - processing only last token")
            # Process only the last token using existing caches
            last_token = idx[:, -1:]
            pos = (attn_mask.long().cumsum(-1) - 1).clamp(min=0) if attn_mask is not None else torch.arange(seq_len, device=device)[None,:].expand(batch_beam_size, -1)
            last_pos = pos[:, -1:]

            x = self.transformer.wte(last_token) + self.transformer.wpe(last_pos)
            x = self.transformer.drop(x)

            # Update caches with new token
            updated_caches = []
            for i, (block, cache) in enumerate(zip(self.transformer.h, beam_caches)):
                x, updated_cache = block(x, attn_mask=attn_mask, cache=cache)
                updated_caches.append(updated_cache)
                if debug and i == 0:
                    k, v = updated_cache
                    print(f"[DEBUG] Layer {i} updated cache shapes: k={k.shape}, v={v.shape}")
            beam_caches = updated_caches

        # Final layer norm and head
        x = self.transformer.ln_f(x)
        logits_full = self.lm_head(x)
        logits = logits_full[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)

        if debug:
            print(f"[DEBUG] Generated probabilities shape: {probs.shape}, max prob: {probs.max().item():.4f}")

        return probs, beam_caches


def _crop_kv_cache(cache, block_size):
    if cache is None:
        return None
    k, v = cache
    if k.size(2) > block_size:
        k = k[:, :, -block_size:, :]
        v = v[:, :, -block_size:, :]
    return (k, v)
