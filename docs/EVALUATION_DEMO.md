# TV6 Evaluation and Demo Guide

Tài liệu này mô tả luồng từ prediction artifact thật đến đánh giá, khóa model, đóng gói và demo. Không có bước nào huấn luyện model hoặc tự tạo kết quả thay thế.

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
python scripts/export_bundle.py \
  --output-dir bundle/seq2seq_attention \
  --selection-manifest experiments/selection_manifest.json \
  --final-test-audit experiments/final_test_audit.json \
  --preprocessing-config configs/data.yaml \
  --model-version 1.0.0
```

Exporter chỉ chấp nhận candidate trong locked selection manifest, tự suy ra model
constructor từ resolved config/schema và kiểm tra lại final-test audit. Bundle v2
chứa model, scaler, schema, resolved/preprocessing config, selection manifest,
final-test audit, metadata và checksum của từng file. Exporter strict-load model
trước khi báo thành công; loader từ chối file thiếu, bị sửa, sai schema/version
hoặc không cùng provenance.

## 6. Payload demo từ dữ liệu thật

Không commit sample tổng hợp. Xuất một cửa sổ thật từ train hoặc validation bằng đúng pipeline hourly trước bước scaling:

```bash
python scripts/generate_demo_payload.py --raw-csv data/raw/jena_climate_2009_2016.csv --schema artifacts/preprocessing/feature_schema.json --split validation --output <local-demo-payload.json>
```

Payload có SHA-256 của source/schema và `synthetic=false`. Script cố ý không cho phép lấy sample từ test split. File đầu ra là artifact cục bộ, không commit vào repository.

Đặt `JENA_DEMO_PAYLOAD_DIR` tới thư mục chứa các payload thật để giao diện hiển thị danh sách chọn sample; nếu không đặt, người dùng vẫn có thể upload JSON trực tiếp. Sample được cấu hình phải có `provenance.synthetic=false`.

## 7. API và Streamlit

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

## 8. Trạng thái bàn giao

Candidate production là Attention LSTM run
`seq2seq_attention_20260917_100234_95dc80`, được chọn độc quyền bằng validation
RMSE. Cả ba validation artifacts dùng cùng population
`jena_validation_fac7f994f7f54901_in168_out72`, 10.026 samples, schema và
train-fitted scaler.

| Model | MAE °C | MSE °C² | RMSE °C | Improvement vs persistence |
|---|---:|---:|---:|---:|
| Attention LSTM | 2.620241 | 11.487950 | 3.389388 | 35.3546% |
| Seq2Seq LSTM | 2.617508 | 11.530652 | 3.395681 | 35.2346% |
| Transformer | 3.593824 | 22.563722 | 4.750129 | 9.4014% |

Validation persistence RMSE là 5.243047 °C. Final test của locked candidate:
MAE 2.721700 °C, MSE 12.300567 °C² và RMSE 3.507216 °C.

Training compute provenance: cả ba run dùng train stride 6; validation stride 1;
input 168 giờ; horizon 72 giờ. Đây không phải full-overlapping-window training.
Transformer dùng cấu hình compute-feasible `d_model=64`, 2 encoder/decoder layers
và teacher forcing 1.0 khi training; validation/inference vẫn dùng 0.0.

The current candidate was selected exclusively using validation results. The
test split had been accessed during an earlier invalid development iteration
and therefore is not considered a pristine unseen holdout. Those earlier test
results were not used to select or modify the final candidate.

Release evidence:

- Bundle: `bundle/seq2seq_attention/`, manifest SHA-256
  `3fa89932b9d60fd596a9e8329fb694078b7daadca1b51854f76c88572efc5460`.
- Locked selection: `experiments/selection_manifest.json`.
- Final-test audit: `experiments/final_test_audit.json`.
- Real E2E payload/exchange: local artifacts under `artifacts/demo/`; these are
  intentionally ignored by Git and must not be replaced with synthetic data.
- Production bundle `model.pt` is a separately delivered release artifact and
  remains ignored by Git under the project artifact policy.

## Acceptance checklist

- Metric tính tay, inverse scale, baseline và per-horizon tests pass.
- Selection chỉ dùng validation và final test one-shot.
- Bundle checksum/load/model parity pass.
- Request 168 giờ, cadence, feature order, missing/extra feature tests pass.
- Model inference nhận `y=None`, teacher forcing 0.0.
- Streamlit chỉ gọi HTTP API và không import/load model bundle.
- Clean-start requires the separately delivered production bundle at the
  documented `bundle/seq2seq_attention/` path; it does not require raw data,
  processed data, runs or retraining.
