# Quy trình làm việc

## Nhánh Git, commit và yêu cầu hợp nhất

Mỗi thay đổi nên được thực hiện trên một nhánh riêng, sử dụng các commit nhỏ và rõ nghĩa, sau đó gửi yêu cầu hợp nhất để xem xét. Không commit dữ liệu cục bộ, thông tin bí mật, điểm kiểm tra hoặc đầu ra của các lần chạy.

## Các quy ước giao tiếp dùng chung

- Lô dữ liệu: `x [B, 168, n_features]`, `y [B, 72, 1]`, kèm dấu thời gian đầu vào và dấu thời gian mục tiêu được căn chỉnh tương ứng.
- Mô hình: `forward(x, y=None, teacher_forcing_ratio=0.0) -> [B, 72, 1]`; bộ giải mã bắt đầu từ giá trị mục tiêu cuối cùng trong đầu vào; kỹ thuật ép theo dữ liệu thật chỉ được phép sử dụng khi huấn luyện.
- Bản ghi dự báo: gồm dấu thời gian, giá trị thực, giá trị dự báo, định danh mô hình, tập dữ liệu và mã lần chạy.
- Lần chạy: gồm mã bất biến, trạng thái vòng đời, lần chạy cha tùy chọn, cấu hình đã được tổng hợp, hạt giống ngẫu nhiên, chỉ số đánh giá, điểm kiểm tra và kết quả dự báo.

## Contract v1 — Data and Model

`WeatherDataset.__getitem__` trả về một `WeatherSample`; sau đó `DataLoader` ghép nhiều sample thành một `WeatherBatch` bằng cách thêm chiều batch ở đầu. Hai cấp dữ liệu được phân biệt như sau:

- `WeatherSample.x`: `float32 [168, n_features]`.
- `WeatherSample.y`: `float32 [72, 1]`.
- `WeatherSample.input_timestamps`: `int64 [168]`.
- `WeatherSample.target_timestamps`: `int64 [72]`.

- `WeatherBatch.x`: `float32 [B, 168, n_features]`.
- `WeatherBatch.y`: `float32 [B, 72, 1]` và phải cùng thiết bị với `x`.
- `WeatherBatch.input_timestamps`: `int64 [B, 168]`.
- `WeatherBatch.target_timestamps`: `int64 [B, 72]`.
- Đơn vị và múi giờ của dấu thời gian phải được khai báo trong siêu dữ liệu schema.
- Timestamp không bắt buộc phải được chuyển lên GPU.
- `target_feature_index` phải lấy từ schema hoặc cấu hình, không được ghi cứng trong code dùng chung.

Mọi mô hình dùng chữ ký `forward(x, y=None, teacher_forcing_ratio=0.0)` và trả về floating-point tensor `[B, 72, 1]` trên cùng thiết bị với `x`. Khi dùng AMP, output có thể có dtype khác input; việc ép kiểu prediction và đưa prediction về thang đo gốc thuộc evaluation/serving, không thuộc model contract. Khi xác thực, kiểm thử hoặc suy luận, `y` bắt buộc là `None` và `teacher_forcing_ratio` bắt buộc bằng `0.0`; dữ liệu mục tiêu tương lai tuyệt đối không được đưa vào suy luận. Mọi mô hình phải hỗ trợ `batch_size=1`. Trọng số chú ý, nếu có, phải được cung cấp qua phương thức riêng và không được thay đổi đầu ra chung của `forward`.

TV1 chịu trách nhiệm chốt interface và validator. TV2 triển khai dữ liệu theo contract này; TV3–TV5 triển khai các mô hình tương ứng mà không thay đổi interface chung.

## Lệnh huấn luyện

Ba lệnh real training đã được sử dụng:

```bash
.venv/bin/python scripts/train.py --config configs/lstm.yaml --device cpu --data-dir data/processed --runs-dir runs
.venv/bin/python scripts/train.py --config configs/attention.yaml --device cpu --data-dir data/processed --runs-dir runs
.venv/bin/python scripts/train.py --config configs/transformer.yaml --device cpu --data-dir data/processed --runs-dir runs
```

### Training compute policy của ba real runs ngày 2026-09-17

- Input là 168 giờ và horizon là 72 giờ theo contract đã khóa.
- Cả ba model đọc `train_processed.csv` thật với `training.data_stride=6`. Đây là huấn luyện trên mỗi cửa sổ chồng lấp thứ sáu trải dọc tập train; **không phải full-overlapping-window training**.
- Validation đọc `val_processed.csv` thật với stride 1, dùng toàn bộ 10.026 cửa sổ hợp lệ và cùng population cho cả ba model.
- Transformer dùng cấu hình compute-feasible cho CPU: `d_model=64`, 2 encoder layers, 2 decoder layers, 4 heads, feed-forward 256 và teacher forcing 1.0 khi training. Validation/prediction của Transformer vẫn autoregressive với teacher forcing 0.0.
- Split chronological, scaler train-only, feature schema, input length và horizon không được thay đổi để cải thiện metric.
- Provenance cụ thể của từng run nằm trong `resolved_config.yaml`, `environment.json` và `run_record.json`; model selection chỉ dùng validation artifacts trong experiment registry.

## Bàn giao sản phẩm

Mỗi lần bàn giao phải có cấu hình đã được tổng hợp, siêu dữ liệu của lần chạy, chỉ số đánh giá, `best.pt`, `last.pt`, kết quả dự báo và lệnh tái tạo chính xác. Các sản phẩm dung lượng lớn phải được lưu ngoài Git.

## Định nghĩa hoàn thành (Definition of Done)

Thay đổi chỉ được xem là hoàn thành khi tuân thủ quy ước giao tiếp, có kiểm thử phù hợp, lưu đủ siêu dữ liệu phục vụ tái lập, có tài liệu bàn giao, đã được xem xét, không gây rò rỉ dữ liệu và không đưa sản phẩm sinh tự động vào Git.

## Báo cáo trở ngại

Báo cáo trở ngại phải nêu rõ người phụ trách, quy ước giao tiếp bị ảnh hưởng, bằng chứng, các kiểm tra đã thử, mức độ ảnh hưởng và quyết định cần được đưa ra. Không được che giấu các lần chạy thất bại.
