"""Bahdanau (additive) attention over encoder outputs.

Owner: TV4 - Attention LSTM & Interpretability.

This module is intentionally decoupled from any specific encoder/decoder
implementation so it can be unit tested in isolation and reused by other
attention-based architectures if needed later.

Contract (internal to this module, does not replace the shared model
contract in docs/WORKFLOW.md / agents/rules/MODEL_RULES.md):

    encoder_outputs: float tensor [B, T_enc, encoder_hidden_size]
    decoder_hidden:  float tensor [B, decoder_hidden_size]
        (the *top layer* decoder hidden state for the current step, i.e.
        hidden[-1] if hidden has shape [num_layers, B, decoder_hidden_size])
    mask (optional):  bool/float tensor [B, T_enc], True/1 = keep, False/0 = ignore.
        Positions that are masked out receive an attention weight of
        (numerically) 0 after softmax, and never contribute to the context
        vector or to gradients w.r.t. those encoder positions.

Returns:
    context: float tensor [B, encoder_hidden_size]
    weights: float tensor [B, T_enc], each row sums to ~1.0 (or exactly 0.0
        for a hypothetical fully-masked row - see notes below).
"""
from __future__ import annotations

import torch
from torch import nn

# Large-but-finite negative number used to mask out logits before softmax.
# We intentionally avoid float("-inf") because softmax over a row that is
# *entirely* -inf produces NaN (0/0). Using a large finite negative value
# keeps softmax numerically well-defined (see _NEG_INF docstring below).
_NEG_INF = -1e9


class BahdanauAttention(nn.Module):
    """Additive attention: score(h_dec, h_enc) = v^T tanh(W1 h_enc + W2 h_dec).

    This is attention computed fresh at *every decoder step* (as required by
    the mission spec), not a single attention pass over the whole sequence.
    """

    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        attention_dim: int | None = None,
    ) -> None:
        super().__init__()

        if encoder_hidden_size <= 0:
            raise ValueError("encoder_hidden_size must be positive")
        if decoder_hidden_size <= 0:
            raise ValueError("decoder_hidden_size must be positive")

        # Default: reuse hidden_size as the attention projection dim. Kept
        # configurable in case a reviewer wants to decouple it later.
        self.attention_dim = attention_dim or encoder_hidden_size

        self.encoder_proj = nn.Linear(encoder_hidden_size, self.attention_dim, bias=False)
        self.decoder_proj = nn.Linear(decoder_hidden_size, self.attention_dim, bias=False)
        self.energy_proj = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if decoder_hidden.ndim != 2:
            raise ValueError(
                f"decoder_hidden must be [B, decoder_hidden_size], got {list(decoder_hidden.shape)}"
            )
        if encoder_outputs.ndim != 3:
            raise ValueError(
                f"encoder_outputs must be [B, T_enc, encoder_hidden_size], got {list(encoder_outputs.shape)}"
            )
        if decoder_hidden.shape[0] != encoder_outputs.shape[0]:
            raise ValueError("decoder_hidden and encoder_outputs must share batch size")

        # [B, 1, attn_dim] + [B, T_enc, attn_dim] -> broadcast to [B, T_enc, attn_dim]
        decoder_term = self.decoder_proj(decoder_hidden).unsqueeze(1)
        encoder_term = self.encoder_proj(encoder_outputs)
        energy = torch.tanh(decoder_term + encoder_term)

        # [B, T_enc, 1] -> [B, T_enc]
        scores = self.energy_proj(energy).squeeze(-1)

        if mask is not None:
            if mask.shape != scores.shape:
                raise ValueError(
                    f"mask must have shape {list(scores.shape)}, got {list(mask.shape)}"
                )
            bool_mask = mask.to(dtype=torch.bool)
            scores = scores.masked_fill(~bool_mask, _NEG_INF)

        weights = torch.softmax(scores, dim=-1)

        if mask is not None:
            # Belt-and-suspenders: zero out any residual weight on masked
            # positions (softmax with _NEG_INF already makes these ~0, but
            # this keeps the invariant exact rather than approximate, and
            # protects against a fully-masked row producing a uniform
            # distribution instead of all-zero).
            weights = weights * bool_mask.to(dtype=weights.dtype)

        # [B, 1, T_enc] @ [B, T_enc, H_enc] -> [B, 1, H_enc] -> [B, H_enc]
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, weights
