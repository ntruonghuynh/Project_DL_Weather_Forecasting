"""Tests for the attention LSTM seq2seq model (owner: TV4).

Covers the required GATE checks from MODEL_RULES.md plus the
attention-specific checks from the mission spec:
  - shape (normal batch + batch_size=1)
  - inference-without-target
  - invalid teacher-forcing rejected outside training
  - forward/backward smoke test with finite gradients
  - attention weight normalization (sum ~= 1 over encoder time)
  - attention weights contain no NaN, respect mask
  - one-batch overfit
  - checkpoint save/load
"""
from __future__ import annotations

import copy

import pytest
import torch

from src.models.attention_lstm_seq2seq import AttentionLSTMSeq2Seq

N_FEATURES = 5
TARGET_FEATURE_INDEX = 1
INPUT_LEN = 168
HORIZON = 72
HIDDEN_SIZE = 16  # small for fast tests


def make_model(**overrides) -> AttentionLSTMSeq2Seq:
    kwargs = dict(
        n_features=N_FEATURES,
        target_feature_index=TARGET_FEATURE_INDEX,
        hidden_size=HIDDEN_SIZE,
        num_layers=1,
        dropout=0.0,
        horizon=HORIZON,
    )
    kwargs.update(overrides)
    return AttentionLSTMSeq2Seq(**kwargs)


def make_batch(batch_size: int, dtype: torch.dtype = torch.float32):
    x = torch.randn(batch_size, INPUT_LEN, N_FEATURES, dtype=dtype)
    y = torch.randn(batch_size, HORIZON, 1, dtype=dtype)
    return x, y


class TestShapes:
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_output_shape_eval(self, batch_size):
        model = make_model()
        model.eval()
        x, _ = make_batch(batch_size)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (batch_size, HORIZON, 1)
        assert output.dtype == x.dtype
        assert output.device == x.device

    def test_batch_size_one_keeps_batch_dim(self):
        model = make_model()
        model.eval()
        x, _ = make_batch(1)
        with torch.no_grad():
            output = model(x)
        assert output.ndim == 3
        assert output.shape[0] == 1


class TestInferenceContract:
    def test_inference_without_target(self):
        model = make_model()
        model.eval()
        x, _ = make_batch(2)
        with torch.no_grad():
            output = model(x, y=None, teacher_forcing_ratio=0.0)
        assert output.shape == (2, HORIZON, 1)

    def test_eval_rejects_target(self):
        model = make_model()
        model.eval()
        x, y = make_batch(2)
        with pytest.raises(ValueError):
            model(x, y=y, teacher_forcing_ratio=0.0)

    def test_eval_rejects_nonzero_teacher_forcing(self):
        model = make_model()
        model.eval()
        x, _ = make_batch(2)
        with pytest.raises(ValueError):
            model(x, y=None, teacher_forcing_ratio=0.5)

    def test_invalid_teacher_forcing_ratio_range(self):
        model = make_model()
        model.train()
        x, y = make_batch(2)
        with pytest.raises(ValueError):
            model(x, y=y, teacher_forcing_ratio=1.5)
        with pytest.raises(ValueError):
            model(x, y=y, teacher_forcing_ratio=-0.1)

    def test_training_without_target_but_positive_ratio_raises(self):
        model = make_model()
        model.train()
        x, _ = make_batch(2)
        with pytest.raises(ValueError):
            model(x, y=None, teacher_forcing_ratio=0.5)


class TestAttentionWeights:
    def test_weights_sum_to_one(self):
        model = make_model()
        model.eval()
        x, _ = make_batch(3)
        with torch.no_grad():
            model(x)
        weights = model.get_last_attention_weights()
        assert weights.shape == (3, HORIZON, INPUT_LEN)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    def test_weights_no_nan(self):
        model = make_model()
        model.eval()
        x, _ = make_batch(2)
        with torch.no_grad():
            model(x)
        weights = model.get_last_attention_weights()
        assert not torch.isnan(weights).any()

    def test_weights_unavailable_before_forward(self):
        model = make_model()
        with pytest.raises(RuntimeError):
            model.get_last_attention_weights()

    def test_mask_zeroes_out_masked_positions(self):
        model = make_model()
        model.eval()
        x, _ = make_batch(2)
        mask = torch.ones(2, INPUT_LEN, dtype=torch.bool)
        # Mask out the first half of the encoder timeline.
        mask[:, : INPUT_LEN // 2] = False
        with torch.no_grad():
            model(x, encoder_mask=mask)
        weights = model.get_last_attention_weights()
        masked_region = weights[:, :, : INPUT_LEN // 2]
        assert torch.allclose(masked_region, torch.zeros_like(masked_region), atol=1e-6)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


class TestGradients:
    def test_forward_backward_smoke(self):
        model = make_model()
        model.train()
        x, y = make_batch(2)
        output = model(x, y=y, teacher_forcing_ratio=0.5)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"

    def test_no_nan_in_output(self):
        model = make_model()
        model.eval()
        x, _ = make_batch(2)
        with torch.no_grad():
            output = model(x)
        assert not torch.isnan(output).any()


class TestOverfit:
    def test_one_batch_overfit(self):
        # Full teacher forcing (ratio=1.0) here on purpose: this is a sanity
        # check that the model *can* drive loss down on a single batch, not
        # a test of scheduled sampling - randomness from ratio<1.0 would
        # make convergence noisy and this test flaky.
        torch.manual_seed(0)
        model = make_model(hidden_size=32)
        model.train()
        x, y = make_batch(4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        initial_loss = None
        final_loss = None
        for step in range(400):
            optimizer.zero_grad()
            output = model(x, y=y, teacher_forcing_ratio=1.0)
            loss = torch.nn.functional.mse_loss(output, y)
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        assert final_loss < initial_loss * 0.7, (
            f"expected substantial overfit on a single batch, "
            f"got initial={initial_loss:.4f} final={final_loss:.4f}"
        )


class TestCheckpoint:
    def test_state_dict_roundtrip(self, tmp_path):
        model = make_model()
        checkpoint_path = tmp_path / "attention_lstm.pt"
        torch.save(model.state_dict(), checkpoint_path)

        reloaded = make_model()
        reloaded.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        reloaded.eval()

        model.eval()
        x, _ = make_batch(2)
        with torch.no_grad():
            out_a = model(x)
            out_b = reloaded(x)
        assert torch.allclose(out_a, out_b, atol=1e-6)

    def test_reloaded_model_is_independent_copy(self):
        model = make_model()
        cloned = copy.deepcopy(model)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        model.eval()
        cloned.eval()
        x, _ = make_batch(1)
        with torch.no_grad():
            out_a = model(x)
            out_b = cloned(x)
        assert not torch.allclose(out_a, out_b)
