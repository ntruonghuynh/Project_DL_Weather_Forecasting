"""Attention Seq2Seq model implementation.

Owner: TV4 - Attention LSTM & Interpretability.
"""

from __future__ import annotations

from .attention import BahdanauAttention
from .attention_lstm_seq2seq import (
    AttentionLSTMDecoder,
    AttentionLSTMEncoder,
    AttentionLSTMSeq2Seq,
)

# Standard alias conforming to project scaffold and contracts
Seq2SeqAttention = AttentionLSTMSeq2Seq

__all__ = [
    "BahdanauAttention",
    "AttentionLSTMEncoder",
    "AttentionLSTMDecoder",
    "AttentionLSTMSeq2Seq",
    "Seq2SeqAttention",
]
