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
| `src/models/attention_lstm_seq2seq.py` | `AttentionLSTMSeq2Seq` model |
| `src/viz/attention_heatmap.py` | heatmap plotting, sample selection, demo JSON export |
| `configs/attention_lstm.yaml` | hyperparameters |
| `tests/test_attention_lstm_seq2seq.py` | unit/smoke tests |

## Hyperparameters (default, see `configs/attention_lstm.yaml`)
`hidden_size=128`, `num_layers=1`, `dropout=0.0`, `horizon=72`,
`attention_dim=hidden_size` — chosen to match TV3's plain LSTM defaults for
a like-for-like comparison, per the "same training budget" constraint.

## Known open items / ASSUMPTIONS (flag for review)
1. **`src/models/base.py` was not available** when this was written. This
   model imports `validate_forward_arguments` / `validate_model_output`
   from `.base` if present, and falls back to a locally re-implemented
   version (matching MODEL_RULES.md's description) otherwise. **TV1/TV3
   should confirm the fallback matches the real `base.py` exactly**, or
   this file should be updated once `base.py` is available.
2. **No example config for TV3's plain LSTM was available**, so
   `configs/attention_lstm.yaml`'s key structure is inferred from
   `Seq2SeqLSTM.__init__` and `docs/WORKFLOW.md`'s
   `scripts/train.py --config configs/<model>.yaml` convention. Align key
   names with the team's real config schema once visible.
3. **Encoder is reimplemented locally** (`AttentionLSTMEncoder`) rather
   than reusing TV3's `LSTMEncoder`, because TV3's encoder discards
   per-timestep outputs (`_, (hidden, cell) = self.lstm(x)`), which
   attention needs. This is a private detail of this model only and does
   not require changes to the shared data/model contract or to TV3/TV5's
   code.
4. **Directory placement of `src/viz/`** is a suggestion — team may prefer
   `src/interpretability/` or `scripts/`; adjust the import path in tests
   if moved.

## Validation status
- All unit/smoke tests pass locally (17/17): shape (incl. `batch_size=1`),
  inference-without-target, invalid teacher-forcing rejected outside
  training, forward/backward finite-gradient smoke test, attention weight
  normalization (`sum ≈ 1`, no NaN), mask correctness, one-batch overfit,
  checkpoint save/load roundtrip.
- **Not yet run**: training/validation on the real Jena dataset (blocked
  on the shared data pipeline / `configs/data.yaml` access), and the
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
- `export_attention_demo_data` now also carries `run_id`/`model_id`/`data_split`
  for the same traceability reason.
- This model's `forward()` already enforces EVALUATION_RULES.md's inference
  contract (`y=None`, `teacher_forcing_ratio=0.0` outside training), so no
  extra work is needed there for TV6's evaluation pipeline to call it
  correctly. Persistence-baseline computation, inverse-scaling to °C, and
  per-horizon (72-step) metric aggregation are TV6's responsibility, not
  this model's — this model only ever returns standardized-scale
  predictions, per MODEL_RULES.md ("Model MUST NOT inverse scale
  prediction").
