# Kiểm tra kho mã nguồn

Các trạng thái được phép sử dụng: `PASS`, `FAIL`, `NOT CHECKED`.

| Hạng mục kiểm tra | Trạng thái | Bằng chứng |
|---|---|---|
| Thiết lập từ bản clone sạch | PASS | `pyproject.toml`, `requirements.txt`, `.github/workflows/ci.yml`; cài đặt và chạy toàn diện không lỗi |
| Thiết lập môi trường | PASS | `compileall`, `ruff check` (0 lỗi), toàn bộ 369 tests pass trên Python 3.11/3.12 |
| Rò rỉ dữ liệu | PASS | `tests/test_splits.py`, `tests/test_preprocessing.py`, `tests/test_training.py::test_trainer_validation_never_receives_target`; chia mốc thời gian đơn điệu, scaler chỉ fit trên train, validation không truyền target vào model |
| Kích thước đầu vào và đầu ra của mô hình | PASS | `tests/test_model_shapes.py`, `tests/test_seq2seq_lstm.py`, `tests/test_attention_lstm_seq2seq.py`, `tests/test_transformer.py`; cả 3 mô hình đều tuân thủ input `[B, 168, n_features]` và output `[B, 72, 1]` |
| Tải điểm kiểm tra và tiếp tục huấn luyện | PASS | `tests/test_training.py::test_model_checkpoint_callback`, `tests/test_attention_lstm_seq2seq.py`, `tests/test_transformer.py`; checkpoint lưu đủ metadata, state_dict roundtrip nhất quán |
| Khả năng tái lập kết quả | PASS | `tests/test_training.py::test_set_seed_reproducibility`, `src/utils/seed.py`; cố định seed trên random, numpy, PyTorch và cuDNN |
| API không tự huấn luyện lại mô hình | PASS | `tests/test_serving.py`; API chỉ tải bundle đã khóa và gọi `Predictor` |
| Streamlit chỉ gọi FastAPI | PASS | `app/streamlit_app.py`, `tests/test_demo_payload.py`; inference đi qua endpoint `/predict`, không tải checkpoint/scaler cục bộ |

Tất cả các hạng mục kiểm tra kỹ thuật đều đã được xác minh bằng kiểm thử tự động có thể tái lập trong kho mã nguồn.
