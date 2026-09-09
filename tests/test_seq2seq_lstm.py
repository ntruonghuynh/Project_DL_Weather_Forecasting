import pytest
import torch

from src.models.seq2seq_lstm import Seq2SeqLSTM


def make_model() -> Seq2SeqLSTM:
    return Seq2SeqLSTM(
        n_features=18,
        target_feature_index=1,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        horizon=72,
    )


def test_seq2seq_lstm_output_shape():
    model = make_model()
    model.eval()

    x = torch.randn(4, 168, 18)

    with torch.no_grad():
        prediction = model(x)

    assert prediction.shape == (4, 72, 1)


def test_seq2seq_lstm_batch_size_one():
    model = make_model()
    model.eval()

    x = torch.randn(1, 168, 18)

    with torch.no_grad():
        prediction = model(x)

    assert prediction.shape == (1, 72, 1)


def test_seq2seq_lstm_output_is_finite():
    model = make_model()
    model.eval()

    x = torch.randn(2, 168, 18)

    with torch.no_grad():
        prediction = model(x)

    assert torch.isfinite(prediction).all()


def test_eval_rejects_future_target():
    model = make_model()
    model.eval()

    x = torch.randn(2, 168, 18)
    y = torch.randn(2, 72, 1)

    with pytest.raises(ValueError, match="y=None"):
        model(x, y=y)


def test_eval_rejects_teacher_forcing():
    model = make_model()
    model.eval()

    x = torch.randn(2, 168, 18)

    with pytest.raises(
        ValueError,
        match="teacher_forcing_ratio",
    ):
        model(
            x,
            teacher_forcing_ratio=0.5,
        )


def test_training_with_teacher_forcing():
    model = make_model()
    model.train()

    x = torch.randn(2, 168, 18)
    y = torch.randn(2, 72, 1)

    prediction = model(
        x,
        y=y,
        teacher_forcing_ratio=1.0,
    )

    assert prediction.shape == (2, 72, 1)

def test_seq2seq_lstm_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)

    model = make_model()
    model.eval()

    x = torch.randn(2, 168, 18)

    with torch.no_grad():
        prediction_before = model(x)

    checkpoint_path = tmp_path / "seq2seq_lstm.pt"
    torch.save(model.state_dict(), checkpoint_path)

    restored_model = make_model()
    restored_model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    )
    restored_model.eval()

    with torch.no_grad():
        prediction_after = restored_model(x)

    assert torch.allclose(
        prediction_before,
        prediction_after,
        atol=1e-6,
    )



    


def test_seq2seq_lstm_tiny_batch_overfit():
    torch.manual_seed(0)

    model = Seq2SeqLSTM(
        n_features=18,
        target_feature_index=1,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        horizon=72,
    )
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-2,
    )

    x = torch.randn(1, 168, 18)

    # Tạo một tiny batch đơn giản để kiểm tra model có học được hay không.
    x[:, -1, 1] = 0.5
    y = torch.full((1, 72, 1), 0.5)

    with torch.no_grad():
        initial_prediction = model(
            x,
            y=y,
            teacher_forcing_ratio=1.0,
        )
        initial_loss = torch.mean(
            (initial_prediction - y) ** 2
        ).item()

    for _ in range(40):
        optimizer.zero_grad()

        prediction = model(
            x,
            y=y,
            teacher_forcing_ratio=1.0,
        )

        loss = torch.mean(
            (prediction - y) ** 2
        )

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_prediction = model(
            x,
            y=y,
            teacher_forcing_ratio=1.0,
        )
        final_loss = torch.mean(
            (final_prediction - y) ** 2
        ).item()

    assert final_loss < initial_loss