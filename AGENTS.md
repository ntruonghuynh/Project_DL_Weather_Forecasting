# Hướng dẫn làm việc theo vai trò

`AGENTS.md` là điểm điều hướng cho các role playbook của project. Các role là hướng dẫn làm việc, không phải source code, framework multi-agent hoặc vị trí cố định của thành viên.

## Project context

Project xây dựng hệ thống sequence-to-sequence dự báo nhiệt độ Jena Climate từ 168 giờ dữ liệu quá khứ sang 72 giờ tương lai. Ba kiến trúc mục tiêu là Encoder–Decoder LSTM, Attention LSTM và Transformer.

Data Batch Contract v1 đã khóa:

```text
x:                 float32 [B, 168, n_features]
y:                 float32 [B, 72, 1]
input_timestamps:  int64   [B, 168]
target_timestamps: int64   [B, 72]
```

Model Contract v1 đã khóa:

```python
forward(x, y=None, teacher_forcing_ratio=0.0) -> prediction
```

`prediction` phải là floating tensor `[B, 72, 1]` trên cùng device với `x`. Schema hiện có 18 model features và target `T (degC)`, nhưng code dùng chung phải đọc feature order, feature count và target index từ schema/config, không ghi cứng các giá trị này.

Trạng thái được xác minh trong codebase: data pipeline và dataset contract đã có implementation; ba model, training CLI/loop, evaluation CLI/metrics và serving inference/client vẫn `CHƯA TRIỂN KHAI` hoặc mới là scaffold.

## Role routing

| Loại công việc | Role playbook |
| --- | --- |
| Contract, CI, tích hợp và review PR | [Project Integrator](agents/roles/project_integrator.md) |
| Data pipeline | [Data Engineer](agents/roles/data_engineer.md) |
| Model | [Model Developer](agents/roles/model_developer.md) |
| Training | [Training Engineer](agents/roles/training_engineer.md) |
| Evaluation và audit | [Evaluation Auditor](agents/roles/evaluation_auditor.md) |
| API và Streamlit | [Serving Developer](agents/roles/serving_developer.md) |

Xem [bản đồ hệ thống](agents/README.md), [rules bắt buộc](agents/rules/README.md) và [templates tái sử dụng](agents/templates/PR_CHECKLIST.md) trước khi bắt đầu.

## Global workflow

1. Đọc `AGENTS.md`, rồi đọc [agents/README.md](agents/README.md).
2. Chọn role trong `agents/roles/` và đọc rule được yêu cầu trong `agents/rules/`.
3. Kiểm tra contract, file ownership và chỉ sửa đúng phạm vi.
4. Chạy quality gate.
5. Dùng checklist trong `agents/templates/`; ghi [bug](docs/BUG_LOG.md) hoặc [decision](docs/DECISION_LOG.md) khi cần.
6. Tạo Pull Request; không push trực tiếp vào `main`.

## Global prohibitions

- Không tự thay đổi contract đã khóa.
- Không dùng test set để chọn model hoặc hyperparameter.
- Không hard-code feature count hoặc target index trong model và code dùng chung.
- Không commit raw/processed dataset, checkpoint hoặc run artifacts.
- Không sửa file ngoài scope nếu chưa được leader đồng ý.
- Không che giấu test bị skip, validation bị bỏ qua hoặc run thất bại.
- Không tạo số liệu, log, checkpoint, figure hoặc artifact giả.

## Stop conditions

Dừng thay đổi và báo leader khi:

- Contract mâu thuẫn với implementation.
- Schema, config và artifact không đồng nhất.
- Cần sửa file do thành viên hoặc role khác sở hữu.
- Test thất bại nhưng nguyên nhân nằm ngoài phạm vi.
- Có nguy cơ data leakage.
- Không xác định được nguồn của checkpoint hoặc scaler.
- Yêu cầu có thể làm mất dữ liệu hoặc lịch sử Git.
