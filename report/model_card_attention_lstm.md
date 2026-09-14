# Model Card — Attention LSTM Seq2Seq (TV4)

## Owner
Attention LSTM & Interpretability Owner (TV4).

## Summary
Encoder-decoder LSTM for 168h -> 72h Jena temperature forecasting, augmented
with Bahdanau (additive) attention computed at every decoder step over the
full encoder output sequence. Architecturally independent from TV3's plain
`Seq2SeqLSTM` (which only exposes final hidden/cell state, not per-step
outputs), but conforms to the same shared forecast contract.

## Interface contract
- `forward(x, y=None, teacher_forcing_ratio=0.0, encoder_mask=None) -> [B, 72, 1]`
- `x`: float32 `[B, 168, n_features]`
- Supports `batch_size == 1`
- `n_features` / `target_feature_index` passed in via config, never hard-coded
- `y` and `teacher_forcing_ratio` must be `None` / `0.0` outside training
- Attention weights are **not** part of the return value; retrieve via
  `model.get_last_attention_weights()` after calling `forward()` — shape
  `[B, 72, 168]`, each row sums to ~1.0 (or exactly 0.0 for a fully-masked
  encoder row).
- `encoder_mask` (optional): bool `[B, 168]`, `True`/`1` = attend, `False`/`0`
  = ignore. Not used by default (this dataset has no padding — fixed 168h
  windows), included for completeness per MODEL_RULES.md's "optional mask".

## Files
| File | Purpose |
|---|---|
| `src/models/attention.py` | `BahdanauAttention` module, decoupled/unit-testable |
| `src/models/attention_lstm_seq2seq.py` | `AttentionLSTMSeq2Seq` model (main implementation) |
| `src/models/seq2seq_attention.py` | Re-exports as `Seq2SeqAttention` alias (scaffold compatibility) |
| `src/viz/attention_heatmap.py` | heatmap plotting, sample selection, demo JSON export |
| `src/viz/__init__.py` | Package init with public API exports |
| `configs/attention.yaml` | Hyperparameters (primary config, scaffold-compatible name) |
| `configs/attention_lstm.yaml` | Hyperparameters (detailed config with notes) |
| `tests/test_attention_lstm_seq2seq.py` | Unit/smoke tests (17 test cases) |
| `report/model_card_attention_lstm.md` | This document |

## Hyperparameters (default, see `configs/attention.yaml`)
`hidden_size=128`, `num_layers=1`, `dropout=0.0`, `horizon=72`,
`attention_dim=hidden_size` — chosen to match TV3's plain LSTM defaults for
a like-for-like comparison, per the "same training budget" constraint.

## Architecture notes
1. **Encoder is reimplemented locally** (`AttentionLSTMEncoder`) rather
   than reusing TV3's `LSTMEncoder`, because TV3's encoder discards
   per-timestep outputs (`_, (hidden, cell) = self.lstm(x)`), which
   attention needs. This is a private detail of this model only and does
   not require changes to the shared data/model contract or to TV3/TV5's
   code.
2. **Validators** (`validate_forward_arguments`, `validate_model_output`)
   are imported directly from `src/models/base.py`, ensuring identical
   behavior with TV3's plain LSTM and any future models.
3. **`src/models/seq2seq_attention.py`** re-exports the model as
   `Seq2SeqAttention = AttentionLSTMSeq2Seq` for scaffold compatibility,
   so both import paths work:
   - `from src.models.seq2seq_attention import Seq2SeqAttention`
   - `from src.models.attention_lstm_seq2seq import AttentionLSTMSeq2Seq`

## Validation status
- All unit/smoke tests pass locally (17/17): shape (incl. `batch_size=1`),
  inference-without-target, invalid teacher-forcing rejected outside
  training, forward/backward finite-gradient smoke test, attention weight
  normalization (`sum ≈ 1`, no NaN), mask correctness, one-batch overfit,
  checkpoint save/load roundtrip.
- `python -m compileall` passes with zero errors on all source files.
- **Not yet run**: training/validation on the real Jena dataset (blocked
  on the shared trainer / `scripts/train.py` integration), and the
  official evaluation metric. Per EXPERIMENT_RULES.md, no test-set numbers
  exist yet and none will be produced until this candidate is promoted.

## Reminder
Heatmaps from `src/viz/attention_heatmap.py` are an explanation/debug tool
only. They do not factor into model selection — the shared evaluation
metric on the validation set does.

## VISUALIZATION_RULES.md / EVALUATION_RULES.md compliance
- `plot_attention_heatmap` requires `run_id`, `model_id`, `data_split` and
  puts them in the figure title; axis labels state units (hours) and the
  colorbar label states the weights are unitless and sum to 1 per row.
- `save_report_figure` is the only sanctioned way to add a heatmap to
  `report/figures/` — it forces a `Finding -> Impact -> Decision` caption
  (raises if any is empty) and appends a row to `report/figure_manifest.csv`
  in the same call, so every report figure stays traceable to a run and
  reproducible from code (no hand-edited images/rows).
- `export_attention_demo_data` carries `run_id`/`model_id`/`data_split`
  for the same traceability reason.
- This model's `forward()` already enforces EVALUATION_RULES.md's inference
  contract (`y=None`, `teacher_forcing_ratio=0.0` outside training), so no
  extra work is needed there for TV6's evaluation pipeline to call it
  correctly. Persistence-baseline computation, inverse-scaling to °C, and
  per-horizon (72-step) metric aggregation are TV6's responsibility, not
  this model's — this model only ever returns standardized-scale
  predictions, per MODEL_RULES.md ("Model MUST NOT inverse scale
  prediction").
