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

## Trạng thái

Kho mã nguồn hiện mới ở giai đoạn dựng khung. Các phần tiền xử lý, mô hình, huấn luyện, suy luận và thực nghiệm chưa được triển khai; dự án chưa công bố bất kỳ kết quả thực nghiệm nào.
