# TV6 Evaluation and Demo Guide

Tài liệu này mô tả luồng TV6 từ prediction artifact thật đến đánh giá, khóa model, đóng gói và demo. Không có bước nào huấn luyện model hoặc tự tạo kết quả thay thế.

## 1. Prediction artifact

Model owner bàn giao một file NPZ có `y_true`, `y_pred`, `target_timestamps` và `last_observed_target`. Hai mảng đầu có shape `[N,72]` hoặc `[N,72,1]`. Metadata JSON phải chứa:

```json
{
  "run_id": "<immutable-run-id>",
  "model_name": "<model-name>",
  "model_version": "<version>",
  "split": "validation",
  "population_id": "<same-population-id-for-all-models>",
  "checkpoint": "<checkpoint-path-or-id>",
  "resolved_config": "<config-path-or-id>",
  "schema_hash": "<feature-schema-sha256>",
  "scaler_sha256": "<scaler-sha256>"
}
```

## 2. Đánh giá validation

```bash
python scripts/evaluate.py --predictions <predictions.npz> --metadata <metadata.json> --output-dir <evaluation-dir> --scale standardized --scaler <scaler.joblib> --target-index <schema-target-index>
```

Đầu ra gồm `metrics.json`, `metrics_per_horizon.csv`, `worst_cases.csv`, phân tích theo horizon/mùa/giờ, hai figure và manifest. Với prediction đã ở °C, dùng `--scale degC` và bỏ scaler/target index.

## 3. So sánh và khóa candidate

```bash
python scripts/compare_models.py <lstm-metrics.json> <attention-metrics.json> <transformer-metrics.json> --output <comparison.csv> --baseline-rmse <validation-baseline-rmse> --population-id <population-id> --selection-manifest <selection_manifest.json> --approved-by TV6 --approved-by TV1
```

Lệnh từ chối test metric, population không đồng nhất và candidate không vượt validation baseline gate 3%.

## 4. Final test

Chỉ candidate trong manifest đã khóa được chạy:

```bash
python scripts/evaluate.py --predictions <test-predictions.npz> --metadata <test-metadata.json> --output-dir <final-test-dir> --scale standardized --scaler <scaler.joblib> --target-index <index> --selection-manifest <selection_manifest.json> --final-test-audit <final_test_audit.json>
```

Audit được tạo trước khi đọc kết quả và chặn lần chạy test thứ hai.

## 5. Model bundle

```bash
python scripts/build_bundle.py --destination <bundle-dir> --checkpoint <best.pt> --scaler <scaler.joblib> --feature-schema <feature_schema.json> --resolved-config <resolved_config.json> --metadata <run_metadata.json> --model-class src.models.seq2seq_lstm.Seq2SeqLSTM --model-kwargs "{\"n_features\":18,\"target_feature_index\":1,\"horizon\":72}" --model-name seq2seq_lstm --model-version <version> --run-id <run-id>
```

Bundle chứa checkpoint, scaler, schema, config, metadata và checksum của từng file. Loader từ chối file thiếu, bị sửa hoặc không cùng run.

## 6. API và Streamlit

```bash
make stack BUNDLE=<bundle-dir>
```

Hoặc chạy riêng:

```bash
set JENA_MODEL_BUNDLE=<bundle-dir>
python -m uvicorn api.main:app --port 8000
python -m streamlit run app/streamlit_app.py
```

Payload JSON cho demo có `timestamps` và `observations` đúng 168 giờ. Có thể thêm `future_actual` để hiển thị đối chiếu; trường này không được gửi vào model. API trả 72 dự báo °C, persistence baseline, provenance, latency và attention weights nếu model hỗ trợ.

## Acceptance checklist

- Metric tính tay, inverse scale, baseline và per-horizon tests pass.
- Selection chỉ dùng validation và final test one-shot.
- Bundle checksum/load/model parity pass.
- Request 168 giờ, cadence, feature order, missing/extra feature tests pass.
- Model inference nhận `y=None`, teacher forcing 0.0.
- Streamlit chỉ gọi HTTP API và không import/load model bundle.
