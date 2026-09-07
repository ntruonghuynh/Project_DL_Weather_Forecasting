# Serving Developer

## Mục tiêu

Phục vụ inference an toàn qua FastAPI và Streamlit, tái sử dụng đúng config/schema/scaler/checkpoint của một run hợp lệ.

## Bối cảnh riêng của project

Kiến trúc đích là `Streamlit -> FastAPI -> Predictor -> ModelBundle`. Hiện FastAPI chỉ có health scaffold; `Predictor.predict` và Streamlit client đều `CHƯA TRIỂN KHAI`. Serving không được train model hoặc fit lại preprocessing.

## Phải đọc trước

- [rules/MODEL_RULES.md](../rules/MODEL_RULES.md)
- [rules/EVALUATION_RULES.md](../rules/EVALUATION_RULES.md)
- [rules/DATA_RULES.md](../rules/DATA_RULES.md)
- [docs/WORKFLOW.md](../../docs/WORKFLOW.md)
- `src/serving/schemas.py`, `src/serving/predictor.py`, `api/main.py`, `app/streamlit_app.py`
- `src/models/base.py`, `artifacts/preprocessing/feature_schema.json`

## Input và contract

- Request phải cung cấp đủ 168 hourly observations theo feature schema hoặc định dạng có thể validate và biến đổi rõ ràng.
- Sắp xếp feature theo schema; không hard-code order/count/target index.
- Inference gọi model với `y=None`, `teacher_forcing_ratio=0.0`.
- Prediction model `[B,72,1]` được inverse transform bằng train-fitted scaler và trả với timestamp cùng đơn vị °C.

## Output cần bàn giao

- Loader kiểm tra config, schema, scaler, checkpoint và run identity.
- `Predictor` thực hiện validation/preprocessing/inference/inverse scaling.
- FastAPI endpoint dự báo, health check và error response có cấu trúc.
- Streamlit client chỉ gọi FastAPI; không load model trực tiếp.
- Serving tests; `tests/test_serving.py` hiện `CHƯA TRIỂN KHAI`.

## Quy trình thực hiện

1. Xác minh bundle có checkpoint, resolved config, schema và scaler cùng run.
2. Validate request count, timestamp cadence, missingness, dtype và feature names.
3. Sắp xếp feature theo schema và áp dụng đúng preprocessing contract đã lưu.
4. Chạy model ở eval/no-grad với `y=None` và teacher forcing `0.0`.
5. Validate output shape/device/dtype; inverse target về °C.
6. Sinh đủ 72 target timestamps và response metadata.
7. Giữ Streamlit là HTTP client của FastAPI.
8. Test success path, invalid request, missing artifact và health behavior.

## File được phép sửa

- `src/serving/`
- `api/`
- `app/`
- `tests/test_serving.py` — `CHƯA TRIỂN KHAI`; tạo khi có serving behavior thật.

## File không được tự ý sửa

- `src/data/`, `src/models/`, `src/training/`, `src/evaluation/`.
- Config/schema/scaler/checkpoint của run để làm request chạy được.
- Training script hoặc run artifacts.

## Quy tắc bắt buộc

- Không fit lại scaler hoặc train model trong API.
- Không đọc target tương lai và không truyền `y` vào model.
- Không hard-code feature order, count hoặc target index.
- Streamlit không được load model trực tiếp.
- Không dùng checkpoint thiếu config/schema/scaler đi kèm.
- Response phải ghi rõ đơn vị °C, timestamps, model version và run ID.
- Không log secret hoặc toàn bộ payload nhạy cảm.

## Quality gate

- Contract test cho request 168 giờ và response 72 giờ.
- Test feature reordering, cadence, dtype, missing/extra feature và invalid timestamp.
- Mock/spy test xác nhận inference dùng `y=None`, ratio `0.0`.
- Test inverse scaling và metadata đơn vị.
- Test API không train/fit; Streamlit chỉ gọi FastAPI.
- `python -m pytest`, Ruff, compileall và `git diff --check`.

## Khi nào phải dừng và hỏi leader

- Checkpoint, schema, scaler hoặc resolved config thiếu/mâu thuẫn.
- Request contract không ánh xạ được vào data schema.
- Model không inference được khi `y=None`.
- Cần sửa data/model/training để serving hoạt động.
- Có nguy cơ leakage hoặc trả sai đơn vị/timestamp.

## Pull Request checklist

Dùng [checklist chung](../templates/PR_CHECKLIST.md) cùng các kiểm tra riêng sau:

- [ ] Artifact bundle cùng run và truy vết được.
- [ ] Feature order lấy từ schema.
- [ ] Inference dùng `y=None`, ratio `0.0`.
- [ ] Prediction đủ 72 giờ và đơn vị °C.
- [ ] FastAPI không train/fit; Streamlit chỉ gọi API.
- [ ] Validation, error handling và health tests pass.
- [ ] Không có checkpoint, secret hoặc run artifact trong diff.
