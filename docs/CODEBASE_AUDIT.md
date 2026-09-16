# Kiểm tra kho mã nguồn

Các trạng thái được phép sử dụng: `PASS`, `FAIL`, `NOT CHECKED`.

| Hạng mục kiểm tra | Trạng thái | Bằng chứng |
|---|---|---|
| Thiết lập từ bản clone sạch | NOT CHECKED | |
| Thiết lập môi trường | NOT CHECKED | |
| Rò rỉ dữ liệu | NOT CHECKED | |
| Kích thước đầu vào và đầu ra của mô hình | NOT CHECKED | |
| Tải điểm kiểm tra và tiếp tục huấn luyện | NOT CHECKED | |
| Khả năng tái lập kết quả | NOT CHECKED | |
| API không tự huấn luyện lại mô hình | PASS | `tests/test_serving.py`; API chỉ tải bundle đã khóa và gọi `Predictor` |
| Streamlit chỉ gọi FastAPI | PASS | `app/streamlit_app.py`; inference đi qua `/predict`, không tải checkpoint/scaler |

Các mục còn `NOT CHECKED` cần bằng chứng từ toàn bộ project hoặc run huấn luyện thật; TV6 không tự nâng thành `PASS` khi chưa có bằng chứng.
