# Data Rules

Áp dụng cho `configs/data.yaml`, `src/data/`, `scripts/prepare_data.py`, data tests, data notebooks và preprocessing artifacts.

## Split và leakage

- `MUST` chia train/validation/test theo thời gian: train cũ nhất, test mới nhất.
- `MUST NOT` random split row hoặc để window vượt qua split boundary.
- `MUST` fit scaler và mọi preprocessing statistics chỉ trên train; validation/test chỉ được transform.
- `MUST NOT` backward-fill, dùng future row để tạo past feature hoặc dùng test để điều chỉnh preprocessing.
- `MUST` lấy split ratio và policy từ `configs/data.yaml` hoặc single source of truth được phê duyệt.

## Missing data và gap

- `MUST` xử lý sentinel trước aggregation/resampling.
- `MUST` giữ target `T (degC)` chưa quan sát là missing; `MUST NOT` impute future target.
- `MUST` lấy gap threshold, fill limit và window rejection policy từ config.
- `MUST` loại window có NaN/Inf trong input hoặc target theo policy hiện tại.
- `MUST NOT` dùng `missing_*` làm model feature khi ordered schema đã loại chúng; chúng chỉ phục vụ diagnostic.

## Schema và khả năng tái tạo

- `MUST` lấy feature order, target name và target index từ schema/config.
- `MUST NOT` hard-code `n_features=18` hoặc `target_feature_index=1` trong consumer; các số này chỉ là trạng thái schema hiện tại.
- `MUST` giữ processed data và artifacts có thể tái tạo từ raw provenance, config và code.
- `MUST` lưu metadata về timestamp unit/timezone, split boundary, scaler fit scope và policy version.
- Notebook `MUST NOT` là nơi duy nhất chứa preprocessing logic; canonical logic `MUST` nằm trong `src/data/`.
- Raw/processed dataset `MUST NOT` được commit.

## Data contract

```text
x:                 float32 [B, 168, n_features]
y:                 float32 [B, 72, 1]
input_timestamps:  int64   [B, 168]
target_timestamps: int64   [B, 72]
```

- `MUST` giữ timestamp căn chỉnh với đúng row của `x` và `y`.
- `MUST` tạo dataset riêng cho từng split.
- Train DataLoader `MAY` shuffle window; nội dung và boundary của mỗi window `MUST` giữ nguyên.

## GATE — Data Quality

- `GATE` schema: ordered features, target và policy khớp config/artifact.
- `GATE` dtype/shape: batch đúng dtype, rank và kích thước contract.
- `GATE` timestamp: tăng dần, cadence theo config và metadata đơn vị/timezone có mặt.
- `GATE` split: boundary không overlap; scaler chứng minh fit train-only.
- `GATE` window: không crossing split, không NaN/Inf, input 168 và target 72.
- `GATE` test/assertion phải thực thi được; nhận xét Markdown không đủ bằng chứng.
