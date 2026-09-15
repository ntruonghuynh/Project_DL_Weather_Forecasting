"""Encoder-decoder Transformer for multistep temperature forecasting.

The decoder consumes one scalar target value per step. Its first token is the
last observed target in ``x``; later tokens are either previous predictions or
training targets selected by teacher forcing. A causal mask prevents every
decoder position from attending to later decoder tokens.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import validate_forward_arguments, validate_model_output


class SinusoidalPositionalEncoding(nn.Module):
    """Add deterministic sinusoidal positions to batch-first embeddings."""

    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        # Odd d_model values have one fewer cosine channel than sine channel.
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``values`` with positions added without changing its shape."""
        if values.ndim != 3:
            raise ValueError("positional encoding expects [B, sequence, d_model]")
        if values.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                f"sequence length {values.shape[1]} exceeds configured maximum "
                f"{self.encoding.shape[1]}"
            )
        positions = self.encoding[:, : values.shape[1]].to(
            device=values.device, dtype=values.dtype
        )
        return values + positions


class WeatherTransformer(nn.Module):
    """Transformer Encoder-Decoder implementing Model Contract v1."""

    def __init__(
        self,
        n_features: int,
        target_feature_index: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        horizon: int = 72,
        input_length: int = 168,
    ) -> None:
        super().__init__()
        self._validate_init_arguments(
            n_features=n_features,
            target_feature_index=target_feature_index,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            horizon=horizon,
            input_length=input_length,
        )

        self.n_features = n_features
        self.target_feature_index = target_feature_index
        self.d_model = d_model
        self.horizon = horizon
        self.input_length = input_length

        self.input_projection = nn.Linear(n_features, d_model)
        self.target_projection = nn.Linear(1, d_model)
        self.encoder_positions = SinusoidalPositionalEncoding(d_model, input_length)
        self.decoder_positions = SinusoidalPositionalEncoding(d_model, horizon)
        self.embedding_scale = math.sqrt(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, norm=nn.LayerNorm(d_model)
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(d_model)
        )
        self.output_projection = nn.Linear(d_model, 1)

    @staticmethod
    def _validate_init_arguments(**arguments: int | float) -> None:
        integer_names = (
            "n_features",
            "d_model",
            "nhead",
            "num_encoder_layers",
            "num_decoder_layers",
            "dim_feedforward",
            "horizon",
            "input_length",
        )
        for name in integer_names:
            value = arguments[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        n_features = int(arguments["n_features"])
        target_index = arguments["target_feature_index"]
        if (
            isinstance(target_index, bool)
            or not isinstance(target_index, int)
            or not 0 <= target_index < n_features
        ):
            raise ValueError("target_feature_index must be within the feature range")
        if int(arguments["d_model"]) % int(arguments["nhead"]) != 0:
            raise ValueError("d_model must be divisible by nhead")
        dropout = arguments["dropout"]
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a number")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be within [0.0, 1.0)")

    @staticmethod
    def make_causal_mask(length: int, device: torch.device | str) -> torch.Tensor:
        """Return a boolean mask whose upper triangle blocks future tokens."""
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError("length must be a positive integer")
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one validated input batch into Transformer memory."""
        embedded = self.input_projection(x) * self.embedding_scale
        return self.encoder(self.encoder_positions(embedded))

    def decode(self, memory: torch.Tensor, decoder_values: torch.Tensor) -> torch.Tensor:
        """Decode a known target prefix and return a prediction at every position.

        This method is exposed for focused mask tests and serving diagnostics.
        Normal callers should use :meth:`forward`.
        """
        if decoder_values.ndim != 3 or decoder_values.shape[-1] != 1:
            raise ValueError("decoder_values must have shape [B, sequence, 1]")
        if memory.ndim != 3 or memory.shape[0] != decoder_values.shape[0]:
            raise ValueError("memory and decoder_values must have aligned batch dimensions")
        if memory.device != decoder_values.device:
            raise ValueError("memory and decoder_values must be on the same device")
        if decoder_values.shape[1] > self.horizon:
            raise ValueError("decoder sequence exceeds configured horizon")

        embedded = self.target_projection(decoder_values) * self.embedding_scale
        embedded = self.decoder_positions(embedded)
        mask = self.make_causal_mask(decoder_values.shape[1], decoder_values.device)
        decoded = self.decoder(tgt=embedded, memory=memory, tgt_mask=mask)
        return self.output_projection(decoded)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Forecast ``horizon`` target values from an input history."""
        validate_forward_arguments(
            training=self.training, y=y, teacher_forcing_ratio=teacher_forcing_ratio
        )
        self._validate_inputs(x, y, teacher_forcing_ratio)

        memory = self.encode(x)
        start = x[:, -1, self.target_feature_index].reshape(x.shape[0], 1, 1)

        # Full teacher forcing has a known shifted decoder sequence and can be
        # computed in one masked pass. This greatly reduces training memory and
        # latency compared with rebuilding the decoder prefix at every step.
        if self.training and y is not None and teacher_forcing_ratio == 1.0:
            decoder_values = torch.cat((start, y[:, :-1, :]), dim=1)
            output = self.decode(memory, decoder_values)
        else:
            output = self._decode_autoregressively(
                memory=memory,
                start=start,
                y=y,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )

        validate_model_output(output, x, horizon=self.horizon, target_dim=1)
        return output

    def _decode_autoregressively(
        self,
        memory: torch.Tensor,
        start: torch.Tensor,
        y: torch.Tensor | None,
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        decoder_values = start
        predictions: list[torch.Tensor] = []

        for step in range(self.horizon):
            prediction = self.decode(memory, decoder_values)[:, -1:, :]
            predictions.append(prediction)
            if step == self.horizon - 1:
                continue

            next_value = prediction
            if self.training and y is not None and teacher_forcing_ratio > 0.0:
                teacher_mask = torch.rand(
                    start.shape[0], 1, 1, device=start.device
                ) < teacher_forcing_ratio
                next_value = torch.where(teacher_mask, y[:, step : step + 1, :], prediction)
            decoder_values = torch.cat((decoder_values, next_value), dim=1)

        return torch.cat(predictions, dim=1)

    def _validate_inputs(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        teacher_forcing_ratio: float,
    ) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError("x must have shape [B, input_length, n_features]")
        if x.shape[0] <= 0:
            raise ValueError("x batch size must be positive")
        if x.shape[1] != self.input_length:
            raise ValueError(f"Expected input length {self.input_length}, got {x.shape[1]}")
        if x.shape[2] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.shape[2]}")
        if not x.is_floating_point():
            raise TypeError("x must have a floating-point dtype")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain only finite values")

        if self.training and teacher_forcing_ratio > 0.0 and y is None:
            raise ValueError("y is required when teacher_forcing_ratio > 0 during training")
        if y is None:
            return
        if not isinstance(y, torch.Tensor):
            raise TypeError("y must be a torch.Tensor")
        expected_shape = (x.shape[0], self.horizon, 1)
        if y.shape != expected_shape:
            raise ValueError(f"y must have shape {list(expected_shape)}, got {list(y.shape)}")
        if y.device != x.device:
            raise ValueError("y must be on the same device as x")
        if y.dtype != x.dtype:
            raise ValueError("y must have the same dtype as x")
        if not torch.isfinite(y).all():
            raise ValueError("y must contain only finite values")
