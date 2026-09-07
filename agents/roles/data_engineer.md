# Data Engineer

## Mục tiêu

Duy trì data pipeline có thể tái tạo, không leakage và tạo đúng `WeatherSample`/`WeatherBatch` cho bài toán Jena Climate.

## Bối cảnh riêng của project

Raw data có tần suất 10 phút và được resample theo giờ. Pipeline hiện xử lý timestamp, sentinel, feature thời gian, split theo thời gian, causal forward-fill giới hạn, scaling train-only và window 168→72. Schema hiện có 18 model features; các cột `missing_*` chỉ dùng diagnostic.

## Phải đọc trước

- [rules/DATA_RULES.md](../rules/DATA_RULES.md)
- [rules/VISUALIZATION_RULES.md](../rules/VISUALIZATION_RULES.md) khi tạo data figure hoặc notebook visualization.
- [docs/WORKFLOW.md](../../docs/WORKFLOW.md)
- [configs/data.yaml](../../configs/data.yaml)
- `src/config.py`, `src/data/validator.py`, `src/data/preprocessing.py`, `src/data/splits.py`, `src/data/dataset.py`, `src/data/dataloader.py`
- `artifacts/preprocessing/feature_schema.json` và `artifacts/preprocessing/tv2_data_cleaning_policy.json`

## Input và contract

- Raw source: `data/raw/jena_climate_2009_2016.csv`; không sửa hoặc commit file dữ liệu này.
- Target: `T (degC)`.
- Batch: `x float32 [B,168,n_features]`, `y float32 [B,72,1]`, timestamp `int64` đúng alignment.
- Feature order, target index, gap policy và split ratio phải lấy từ schema/config.

## Output cần bàn giao

- Processed split tái tạo được từ raw source và config.
- Scaler fit trên train, feature schema, split metadata và data-quality evidence.
- Assertion/test thực thi được cho schema, dtype, shape, timestamp, boundary và window validity.
- Notebook kiểm chứng có narrative; canonical logic vẫn nằm trong `src/data/`.

## Quy trình thực hiện

1. Validate cột, dtype, timestamp và provenance của raw data.
2. Sort timestamp; xử lý duplicate và sentinel trước resampling.
3. Resample theo policy trong `configs/data.yaml`.
4. Tạo time features và diagnostic indicators trước imputation.
5. Chia train/validation/test theo thời gian, không shuffle row.
6. Fit preprocessing statistics và scaler chỉ trên train.
7. Chỉ causal forward-fill input theo giới hạn; giữ target tương lai missing.
8. Tạo dataset riêng cho từng split; loại window chứa NaN/Inf theo policy.
9. Chạy tests và notebook verification liên quan.

## File được phép sửa

- `configs/data.yaml`
- `src/data/`
- `scripts/prepare_data.py`
- `tests/test_dataset.py`, `tests/test_preprocessing.py`, `tests/test_prepare_data.py`, `tests/test_splits.py`, `tests/test_validator.py`
- `notebooks/01_data_check.ipynb`, `notebooks/02_preprocessing_check.ipynb`, `notebooks/03_dataset_check.ipynb`
- `artifacts/preprocessing/` chỉ cho schema/policy nhỏ được dự án chủ ý version-control; không commit binary hoặc output lớn ngoài chính sách Git.

## File không được tự ý sửa

- `src/models/`, `src/training/`, `src/evaluation/`, `src/serving/`, `api/`, `app/`.
- Config của model, training loop, metric hoặc serving code.
- Raw/processed dataset và generated scaler/run/checkpoint không thuộc một thay đổi artifact đã được leader duyệt.

## Quy tắc bắt buộc

- Không backward-fill bằng dữ liệu tương lai; không interpolate nếu policy không cho phép.
- Không fit scaler/statistics bằng validation hoặc test.
- Không impute target tương lai.
- Không dùng test để điều chỉnh preprocessing.
- Không đưa `missing_*` vào model tensor khi schema đã loại chúng.
- Không để notebook là nơi duy nhất chứa preprocessing logic.
- Data Quality Gate phải là assertion/test chạy được, không chỉ là Markdown.

## Quality gate

- Test validation, preprocessing, split, dataset và DataLoader contract.
- Assert timestamp tăng, grid theo giờ, split không overlap và window không vượt boundary.
- Assert dtype/shape đúng và output không có NaN/Inf.
- Assert scaler fit scope là train và ordered features khớp schema.
- Chạy `python -m pytest`, Ruff, compileall và `git diff --check`.

## Khi nào phải dừng và hỏi leader

- `configs/data.yaml`, locked schema và artifact metadata không đồng nhất.
- Thay đổi feature order, target, window, gap hoặc split contract.
- Raw provenance/checksum không xác định.
- Cần sửa consumer thuộc model, training, evaluation hoặc serving.
- Có nguy cơ future leakage hoặc mất dữ liệu.

## Pull Request checklist

Dùng [checklist chung](../templates/PR_CHECKLIST.md) cùng các kiểm tra riêng sau:

- [ ] Raw source và policy được ghi rõ.
- [ ] Split chronological; preprocessing fit train-only.
- [ ] Target tương lai không bị impute.
- [ ] `missing_*` không nằm trong model tensor.
- [ ] Window invalid bị loại theo config.
- [ ] Schema, dtype, shape, timestamp và boundary tests pass.
- [ ] Notebook chỉ kiểm chứng canonical code.
- [ ] Không commit raw/processed data hoặc generated artifact bị cấm.
