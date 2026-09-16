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
python scripts/export_bundle.py --run-dir <immutable-run-dir> --output-dir <bundle-dir> --model-class <fully-qualified-model-class> --model-kwargs '<constructor-kwargs-from-locked-config-and-schema>' --model-name <model-name> --model-version <version> --run-id <run-id>
```

`--model-kwargs` là JSON object lấy từ resolved config và feature schema của chính run; không nhập số đoán hoặc ghi cứng theo ví dụ. Bundle chứa checkpoint, scaler, schema, config, metadata và checksum của từng file. Exporter tải thử model với strict state-dict trước khi báo thành công; loader từ chối file thiếu, bị sửa hoặc không cùng run.

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

Code, contract tests và runbook đã hoàn tất. Các con số thực nghiệm, model được chọn, final-test report và bundle production chỉ được tạo khi nhóm huấn luyện bàn giao prediction/checkpoint/config/schema/scaler thật cùng provenance hợp lệ. Trạng thái “chưa có kết quả thật” không phải là kết quả thất bại và không được thay bằng số liệu minh họa.

## Acceptance checklist

- Metric tính tay, inverse scale, baseline và per-horizon tests pass.
- Selection chỉ dùng validation và final test one-shot.
- Bundle checksum/load/model parity pass.
- Request 168 giờ, cadence, feature order, missing/extra feature tests pass.
- Model inference nhận `y=None`, teacher forcing 0.0.
- Streamlit chỉ gọi HTTP API và không import/load model bundle.
