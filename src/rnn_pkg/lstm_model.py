import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Union


class LSTMConfig:
    def __init__(self, vocab_size, n_layer=2, n_embd=256, dropout=0.1):
        self.vocab_size, self.n_layer, self.n_embd, self.dropout = \
            vocab_size, n_layer, n_embd, dropout

class LSTMConsensus(nn.Module):
    """
    Unidirectional L‑STM that mimics GPT’s forward() signature:
        logits: (B, T, vocab)
        loss  : scalar x‑entropy against targets (shifted inside this module)
    """
    def __init__(self, cfg: LSTMConfig, pad_token_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = pad_token_id
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.lstm  = nn.LSTM(cfg.n_embd, cfg.n_embd,
                             num_layers=cfg.n_layer,
                             batch_first=True)
        self.drop  = nn.Dropout(cfg.dropout)
        self.proj  = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, input_ids, lengths=None, targets=None, **ignored):
        """
        input_ids : (B, T)  padded on the right with pad_token_id
        lengths   : (B,)    real lengths WITHOUT pad; if None, infer
        """
        if lengths is None:
            lengths = (input_ids != self.pad_token_id).sum(1)      # (B,)

        x = self.embed(input_ids)                                  # (B,T,E)

        # Store original sequence length to pad output correctly
        original_seq_len = x.size(1)

        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        x, _ = pad_packed_sequence(packed_out, batch_first=True)   # (B,actual_T,E)

        # Pad output to match original input length for language modeling
        if x.size(1) < original_seq_len:
            B, T, E = x.shape
            padding_needed = original_seq_len - T
            padding = torch.zeros(B, padding_needed, E, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)  # (B, original_T, E)

        logits = self.proj(self.drop(x))                           # (B,T,V)

        loss = None
        if targets is not None:
            # Ensure logits and targets have same sequence length
            T_logits = logits.size(1)
            T_targets = targets.size(1)

            if T_logits > T_targets:
                logits = logits[:, :T_targets]
            elif T_targets > T_logits:
                targets = targets[:, :T_logits]

            # Reshape for loss computation
            B, T, V = logits.shape
            logits_flat = logits.reshape(B * T, V)
            targets_flat = targets.reshape(B * T)

            # Create mask and compute loss
            mask = targets_flat != self.pad_token_id
            if mask.any():
                loss = F.cross_entropy(
                    logits_flat[mask], targets_flat[mask], reduction="mean"
                )
            else:
                loss = torch.tensor(0.0, device=logits.device)
        return logits, loss


    # GPT has this method, keep so train loop works
    def configure_optimizers(self, wd, lr, betas, device_type):
        return torch.optim.AdamW(self.parameters(), lr=lr, betas=betas, weight_decay=wd)

    # Mimic MFU estimate so log line does not crash
    def estimate_mfu(self, *a, **kw): return 0.  # dummy

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,      # (B, T0) right-padded
        lengths:   torch.Tensor,      # (B,)   initial real lengths
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Union[int, None] = None,
        eos_token_id: Union[int, None] = None,
    ) -> torch.Tensor:                # returns right-padded (B, T0+∗)
        pad = self.pad_token_id
        unfinished = torch.ones_like(lengths, dtype=torch.bool)  # (B,)

        for _ in range(max_new_tokens):
            logits, _ = self(input_ids, lengths)                 # (B,T,V)

            last = lengths - 1                                   # (B,)
            next_logits = logits[torch.arange(len(last), device=input_ids.device),
                                last]                           # (B,V)
            next_logits = next_logits / temperature

            if top_k is not None:
                k = min(top_k, next_logits.size(-1))
                v, _ = torch.topk(next_logits, k)
                next_logits[next_logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(next_logits, dim=-1)               # (B,V)
            next_token = torch.multinomial(probs, 1).squeeze(1)  # (B,)

            if eos_token_id is not None:
                newly_finished = (next_token == eos_token_id)
                unfinished &= ~newly_finished
            next_token = torch.where(unfinished, next_token,
                                    torch.full_like(next_token, pad))

            # append and update lengths
            input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)  # (B,T+1)
            lengths   = lengths + unfinished

            if not unfinished.any():
                break
        return input_ids
