"""Seq2Seq LSTM with Bahdanau attention over encoder outputs.

Owner: TV4 - Attention LSTM & Interpretability.

Mirrors the shared contract implemented by TV3's src/models/seq2seq_lstm.py:
    forward(x, y=None, teacher_forcing_ratio=0.0) -> float tensor [B, 72, 1]

Key difference vs. TV3's plain LSTM: TV3's LSTMEncoder only returns the
final (hidden, cell) - it does not expose per-timestep outputs, so there is
nothing for an attention mechanism to attend over. This module therefore
defines its own encoder that also returns the full output sequence
(``encoder_outputs``, shape [B, 168, hidden_size]). This is a private
architectural detail of this model only; it does NOT change the shared
data/model contract in docs/WORKFLOW.md, and does not require any change
to TV3's or TV5's code.

Attention weights are exposed via ``get_last_attention_weights()``, a
separate method, per MODEL_RULES.md ("Attention weights SHOULD go through
a separate method/interface; MUST NOT change the prediction contract").
They are NOT returned from ``forward()``.

Validators ``validate_forward_arguments`` and ``validate_model_output``
are imported from the shared ``src/models/base.py``.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import BahdanauAttention
from .base import validate_forward_arguments, validate_model_output


class AttentionLSTMEncoder(nn.Module):
    """Encode the 168-hour multivariate input sequence, keeping all outputs."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs, (hidden, cell) = self.lstm(x)
        # outputs: [B, T_enc, hidden_size] - this is what attention attends over.
        return outputs, hidden, cell


class AttentionLSTMDecoder(nn.Module):
    """Decode one target value at a time, attending over encoder outputs."""

    def __init__(
        self,
        encoder_hidden_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        attention_dim: int | None = None,
    ) -> None:
        super().__init__()

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.attention = BahdanauAttention(
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=hidden_size,
            attention_dim=attention_dim,
        )

        # Input to the decoder LSTM cell at each step: previous prediction
        # (or teacher-forced target) concatenated with the attention
        # context vector computed from the *previous* decoder hidden state.
        self.lstm = nn.LSTM(
            input_size=1 + encoder_hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        # Output projection sees the decoder output fused with the context
        # vector again ("context fusion"), not just the raw LSTM output.
        self.output_layer = nn.Linear(hidden_size + encoder_hidden_size, 1)

    def forward(
        self,
        decoder_input: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Attention is computed from the top-layer hidden state produced by
        # the *previous* step (or the encoder's final hidden state for the
        # first step), per "Bahdanau attention at every decoder step".
        context, weights = self.attention(
            decoder_hidden=hidden[-1], encoder_outputs=encoder_outputs, mask=mask
        )

        # decoder_input: [B, 1, 1] -> concat context: [B, 1, encoder_hidden_size]
        lstm_input = torch.cat([decoder_input, context.unsqueeze(1)], dim=-1)

        decoder_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        fused = torch.cat([decoder_output[:, -1, :], context], dim=-1)
        prediction = self.output_layer(fused)

        return prediction, hidden, cell, weights


class AttentionLSTMSeq2Seq(nn.Module):
    """Encoder-decoder LSTM with Bahdanau attention for 168h -> 72h forecasting.

    Conforms to the shared forecast contract (see docs/WORKFLOW.md and
    agents/rules/MODEL_RULES.md):
      - forward(x, y=None, teacher_forcing_ratio=0.0) -> [B, horizon, 1]
      - x: float [B, 168, n_features]; output on same device/dtype family as x
      - supports batch_size == 1
      - does not hard-code n_features or target_feature_index
      - teacher forcing only usable during training, ratio in [0, 1]
      - eval mode rejects y != None or teacher_forcing_ratio != 0.0
      - does not read data, inverse-scale, checkpoint, or compute metrics
    """

    def __init__(
        self,
        n_features: int,
        target_feature_index: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        horizon: int = 72,
        attention_dim: int | None = None,
    ) -> None:
        super().__init__()

        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if not 0 <= target_feature_index < n_features:
            raise ValueError("target_feature_index must be within the feature range")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be within [0.0, 1.0)")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        self.n_features = n_features
        self.target_feature_index = target_feature_index
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        self.encoder = AttentionLSTMEncoder(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.decoder = AttentionLSTMDecoder(
            encoder_hidden_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            attention_dim=attention_dim,
        )

        # Populated on every forward() call; exposed read-only via
        # get_last_attention_weights(). Never fed back into forward's
        # return value, per the "MUST NOT change prediction contract" rule.
        self._last_attention_weights: torch.Tensor | None = None

    def get_last_attention_weights(self) -> torch.Tensor:
        """Return attention weights from the most recent forward() call.

        Shape: [B, horizon, T_enc]. Row [b, t, :] sums to ~1.0 (or exactly
        0.0 only in the degenerate case of a fully-masked encoder sequence
        for that batch element).

        Raises RuntimeError if forward() has not been called yet.
        """
        if self._last_attention_weights is None:
            raise RuntimeError(
                "No attention weights available yet - call forward() first."
            )
        return self._last_attention_weights

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        encoder_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_forward_arguments(
            training=self.training, y=y, teacher_forcing_ratio=teacher_forcing_ratio
        )

        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [B, input_len, n_features], got {list(x.shape)}"
            )
        if x.shape[-1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.shape[-1]}")
        if not x.is_floating_point():
            raise TypeError("x must have a floating-point dtype")

        if self.training and teacher_forcing_ratio > 0.0 and y is None:
            raise ValueError(
                "y is required when teacher_forcing_ratio > 0 during training"
            )

        if y is not None:
            expected_y_shape = (x.shape[0], self.horizon, 1)
            if y.shape != expected_y_shape:
                raise ValueError(
                    f"y must have shape {list(expected_y_shape)}, got {list(y.shape)}"
                )
            if y.device != x.device:
                raise ValueError("y must be on the same device as x")
            if y.dtype != x.dtype:
                raise ValueError("y must have the same dtype as x")

        if encoder_mask is not None:
            expected_mask_shape = (x.shape[0], x.shape[1])
            if encoder_mask.shape != expected_mask_shape:
                raise ValueError(
                    f"encoder_mask must have shape {list(expected_mask_shape)}, "
                    f"got {list(encoder_mask.shape)}"
                )
            if encoder_mask.device != x.device:
                raise ValueError("encoder_mask must be on the same device as x")

        encoder_outputs, hidden, cell = self.encoder(x)

        # Decoder starts from the final observed target value (same
        # convention as TV3's plain LSTM).
        decoder_input = x[:, -1, self.target_feature_index].reshape(x.shape[0], 1, 1)

        predictions: list[torch.Tensor] = []
        attention_weights: list[torch.Tensor] = []

        for step in range(self.horizon):
            prediction, hidden, cell, weights = self.decoder(
                decoder_input, hidden, cell, encoder_outputs, encoder_mask
            )

            predictions.append(prediction.unsqueeze(1))
            attention_weights.append(weights.unsqueeze(1))

            if step == self.horizon - 1:
                continue

            next_input = prediction

            if self.training and y is not None and teacher_forcing_ratio > 0.0:
                if teacher_forcing_ratio == 1.0:
                    next_input = y[:, step, :]
                else:
                    teacher_mask = (
                        torch.rand(x.shape[0], 1, device=x.device) < teacher_forcing_ratio
                    )
                    next_input = torch.where(teacher_mask, y[:, step, :], prediction)

            decoder_input = next_input.unsqueeze(1)

        output = torch.cat(predictions, dim=1)
        self._last_attention_weights = torch.cat(attention_weights, dim=1).detach()

        validate_model_output(output, x, horizon=self.horizon, target_dim=1)

        return output
