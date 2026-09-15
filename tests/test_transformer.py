"""Contract, masking, learning, and checkpoint tests for TV5's Transformer."""

from __future__ import annotations

import copy

import pytest
import torch

from src.models.transformer import WeatherTransformer

N_FEATURES = 5
TARGET_INDEX = 2
INPUT_LENGTH = 12
HORIZON = 6


def make_model(**overrides: object) -> WeatherTransformer:
    arguments: dict[str, object] = {
        "n_features": N_FEATURES,
        "target_feature_index": TARGET_INDEX,
        "d_model": 8,
        "nhead": 2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "horizon": HORIZON,
        "input_length": INPUT_LENGTH,
    }
    arguments.update(overrides)
    return WeatherTransformer(**arguments)  # type: ignore[arg-type]


def make_batch(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randn(batch_size, INPUT_LENGTH, N_FEATURES),
        torch.randn(batch_size, HORIZON, 1),
    )


@pytest.mark.parametrize("batch_size", [1, 3])
def test_eval_output_follows_prediction_contract(batch_size: int) -> None:
    model = make_model().eval()
    x, _ = make_batch(batch_size)
    with torch.no_grad():
        prediction = model(x)
    assert prediction.shape == (batch_size, HORIZON, 1)
    assert prediction.is_floating_point()
    assert prediction.dtype == x.dtype
    assert prediction.device == x.device
    assert torch.isfinite(prediction).all()


def test_locked_168_to_72_shape() -> None:
    model = make_model(input_length=168, horizon=72).eval()
    x = torch.randn(1, 168, N_FEATURES)
    with torch.no_grad():
        prediction = model(x)
    assert prediction.shape == (1, 72, 1)


def test_causal_mask_blocks_future_positions() -> None:
    mask = WeatherTransformer.make_causal_mask(4, "cpu")
    expected = torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    assert torch.equal(mask, expected)


def test_future_decoder_values_do_not_change_earlier_predictions() -> None:
    torch.manual_seed(0)
    model = make_model().eval()
    x, _ = make_batch(2)
    decoder_values = torch.randn(2, HORIZON, 1)
    changed = decoder_values.clone()
    changed[:, -1, :] += 100.0
    with torch.no_grad():
        memory = model.encode(x)
        original = model.decode(memory, decoder_values)
        modified = model.decode(memory, changed)
    assert torch.allclose(original[:, :-1], modified[:, :-1], atol=1e-6)


def test_decoder_starts_from_last_observed_target() -> None:
    model = make_model().eval()
    x, _ = make_batch(2)
    x[:, -1, TARGET_INDEX] = torch.tensor([0.25, -0.75])
    with torch.no_grad():
        memory = model.encode(x)
        expected = model.decode(memory, x[:, -1, TARGET_INDEX].reshape(2, 1, 1))
        actual = model(x)
    assert torch.allclose(actual[:, :1], expected, atol=1e-6)


def test_eval_rejects_future_targets_and_teacher_forcing() -> None:
    model = make_model().eval()
    x, y = make_batch()
    with pytest.raises(ValueError, match="y=None"):
        model(x, y=y)
    with pytest.raises(ValueError, match="must be 0.0"):
        model(x, teacher_forcing_ratio=0.5)


@pytest.mark.parametrize("ratio", [-0.1, 1.1])
def test_invalid_teacher_forcing_ratio_is_rejected(ratio: float) -> None:
    model = make_model().train()
    x, y = make_batch()
    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        model(x, y=y, teacher_forcing_ratio=ratio)


def test_positive_teacher_forcing_requires_target() -> None:
    model = make_model().train()
    x, _ = make_batch()
    with pytest.raises(ValueError, match="y is required"):
        model(x, teacher_forcing_ratio=0.5)


def test_full_teacher_forcing_forward_backward_has_finite_gradients() -> None:
    model = make_model().train()
    x, y = make_batch()
    prediction = model(x, y=y, teacher_forcing_ratio=1.0)
    torch.nn.functional.mse_loss(prediction, y).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(0)
    model = make_model().eval()
    x, _ = make_batch()
    with torch.no_grad():
        expected = model(x)
    checkpoint = tmp_path / "transformer.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = make_model().eval()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    with torch.no_grad():
        actual = restored(x)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_tiny_batch_can_overfit() -> None:
    torch.manual_seed(0)
    model = make_model().train()
    x = torch.randn(1, INPUT_LENGTH, N_FEATURES)
    x[:, -1, TARGET_INDEX] = 0.4
    y = torch.full((1, HORIZON, 1), 0.4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    with torch.no_grad():
        initial_loss = torch.nn.functional.mse_loss(
            model(x, y=y, teacher_forcing_ratio=1.0), y
        ).item()
    for _ in range(40):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x, y=y, teacher_forcing_ratio=1.0), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = torch.nn.functional.mse_loss(
            model(x, y=y, teacher_forcing_ratio=1.0), y
        ).item()
    assert final_loss < initial_loss * 0.2


def test_model_copy_does_not_share_parameters() -> None:
    model = make_model()
    cloned = copy.deepcopy(model)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    assert not torch.equal(next(model.parameters()), next(cloned.parameters()))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_features": 0}, "n_features"),
        ({"target_feature_index": N_FEATURES}, "target_feature_index"),
        ({"d_model": 7, "nhead": 2}, "divisible"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_invalid_constructor_arguments(overrides: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        make_model(**overrides)
