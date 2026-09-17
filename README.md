# Dự báo thời tiết bằng mô hình chuỗi sang chuỗi

Dự án học sâu dự báo thời tiết tại Jena trong 72 giờ tiếp theo dựa trên dữ liệu của 168 giờ trước đó.

Ba nhóm mô hình dự kiến gồm Seq2Seq LSTM, Seq2Seq có cơ chế chú ý và Transformer.

Kiến trúc khi vận hành: `Streamlit -> FastAPI -> Predictor -> ModelBundle`. Streamlit chỉ đóng vai trò giao diện phía người dùng; việc tải mô hình và suy luận được thực hiện phía sau API.

## Cấu trúc kho mã nguồn

- `configs/`: cấu hình chung và cấu hình mẫu cho từng mô hình
- `src/`: các giao diện xử lý dữ liệu, mô hình, huấn luyện, đánh giá, phục vụ và tiện ích
- `api/`, `app/`: khung điểm khởi chạy FastAPI và Streamlit
- `scripts/`: các điểm khởi chạy để chuẩn bị dữ liệu, huấn luyện, đánh giá và so sánh mô hình
- `tests/`: khung kiểm thử các quy ước giao tiếp
- `docs/`: quy trình làm việc, quy tắc thực nghiệm và danh sách kiểm tra
- `data/`, `runs/`, `experiments/`, `report/`: dữ liệu đầu vào cục bộ và các sản phẩm thực nghiệm

## Thiết lập môi trường cơ bản

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

Sao chép `.env.example` thành `.env` và điều chỉnh các đường dẫn cục bộ khi phần triển khai được bổ sung.

## Kết quả đã khóa

Ba model được huấn luyện trên Jena thật với input 168 giờ, horizon 72 giờ và
`train data_stride=6`. Validation dùng stride 1 trên cùng 10.026 cửa sổ. Đây
không phải full-overlapping-window training.

Candidate được chọn chỉ bằng validation RMSE là Attention LSTM, run
`seq2seq_attention_20260917_100234_95dc80`:

| Split | MAE | MSE | RMSE |
|---|---:|---:|---:|
| Validation | 2.620241 °C | 11.487950 °C² | 3.389388 °C |
| Final test | 2.721700 °C | 12.300567 °C² | 3.507216 °C |

Persistence baseline trên validation có RMSE 5.243047 °C. Selection được khóa
tại `experiments/selection_manifest.json`; final-test provenance nằm tại
`experiments/final_test_audit.json`.

The current candidate was selected exclusively using validation results. The
test split had been accessed during an earlier invalid development iteration
and therefore is not considered a pristine unseen holdout. Those earlier test
results were not used to select or modify the final candidate.

## Production demo

Production bundle ở `bundle/seq2seq_attention/` là release artifact đi kèm,
không phải checkpoint được commit vào Git. Xác minh và chạy stack mà không
retrain:

```bash
python -c "from pathlib import Path; from src.serving.bundle import verify_bundle; verify_bundle(Path('bundle/seq2seq_attention')); print('PASS')"
python scripts/run_stack.py --bundle bundle/seq2seq_attention
```

Kiến trúc runtime là `Streamlit -> FastAPI -> Predictor -> ModelBundle`.
Streamlit không tải model, checkpoint hoặc scaler và không có local inference
fallback. Hướng dẫn evaluation, bundle, payload validation thật và E2E nằm tại
[docs/EVALUATION_DEMO.md](docs/EVALUATION_DEMO.md).
