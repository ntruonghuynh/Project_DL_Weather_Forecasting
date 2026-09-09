from __future__ import annotations

import torch
from torch import nn

from .base import validate_forward_arguments, validate_model_output


class LSTMEncoder(nn.Module):
    """Encode the 168-hour multivariate input sequence."""

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
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class LSTMDecoder(nn.Module):
    """Decode one target value at a time."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(
        self,
        decoder_input: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder_output, (hidden, cell) = self.lstm(
            decoder_input,
            (hidden, cell),
        )

        prediction = self.output_layer(decoder_output[:, -1, :])

        return prediction, hidden, cell


class Seq2SeqLSTM(nn.Module):
    """Encoder-decoder LSTM for 168h -> 72h temperature forecasting."""

    def __init__(
        self,
        n_features: int,
        target_feature_index: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        horizon: int = 72,
    ) -> None:
        super().__init__()

        if n_features <= 0:
            raise ValueError("n_features must be positive")

        if not 0 <= target_feature_index < n_features:
            raise ValueError(
                "target_feature_index must be within the feature range"
            )

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

        self.encoder = LSTMEncoder(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.decoder = LSTMDecoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        validate_forward_arguments(
            training=self.training,
            y=y,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")

        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [B, input_len, n_features], got {list(x.shape)}"
            )

        if x.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {x.shape[-1]}"
            )

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
                    f"y must have shape {list(expected_y_shape)}, "
                    f"got {list(y.shape)}"
                )

            if y.device != x.device:
                raise ValueError("y must be on the same device as x")

            if y.dtype != x.dtype:
                raise ValueError("y must have the same dtype as x")

        hidden, cell = self.encoder(x)

        # Decoder starts from the final observed target value.
        decoder_input = x[
            :, -1, self.target_feature_index
        ].reshape(x.shape[0], 1, 1)

        predictions: list[torch.Tensor] = []

        for step in range(self.horizon):
            prediction, hidden, cell = self.decoder(
                decoder_input,
                hidden,
                cell,
            )

            predictions.append(prediction.unsqueeze(1))

            if step == self.horizon - 1:
                continue

            next_input = prediction

            if (
                self.training
                and y is not None
                and teacher_forcing_ratio > 0.0
            ):
                if teacher_forcing_ratio == 1.0:
                    next_input = y[:, step, :]
                else:
                    teacher_mask = (
                        torch.rand(
                            x.shape[0],
                            1,
                            device=x.device,
                        )
                        < teacher_forcing_ratio
                    )

                    next_input = torch.where(
                        teacher_mask,
                        y[:, step, :],
                        prediction,
                    )

            decoder_input = next_input.unsqueeze(1)

        output = torch.cat(predictions, dim=1)

        validate_model_output(
            output,
            x,
            horizon=self.horizon,
            target_dim=1,
        )

        return output