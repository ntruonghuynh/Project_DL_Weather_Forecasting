# Model Rules

Áp dụng cho ba implementation trong `src/models/` và config model tương ứng.

## Contract

- Model `MUST` dùng signature `forward(x, y=None, teacher_forcing_ratio=0.0)`.
- `x` `MUST` có shape `[B,168,n_features]` theo config/schema.
- Prediction `MUST` là floating tensor `[B,72,1]` trên cùng device với `x`.
- Model `MUST` hỗ trợ `batch_size=1` mà không làm mất chiều batch.
- Model `MUST NOT` hard-code feature count hoặc target index.
- Decoder start, nếu kiến trúc cần, `MUST` lấy target quan sát cuối cùng bằng target index từ schema/config.

## Teacher forcing và inference

- Teacher forcing `MAY` dùng chỉ khi training và ratio trong `[0.0,1.0]`.
- Validation, test và inference `MUST` truyền `y=None` và ratio `0.0`.
- Model ở eval mode `MUST` từ chối target hoặc teacher-forcing ratio khác `0.0`.
- Attention weights `SHOULD` đi qua method/interface riêng; `MUST NOT` thay đổi prediction contract.

## Ranh giới trách nhiệm

- Model `MUST NOT` đọc CSV, validate raw data hoặc tự preprocess.
- Model `MUST NOT` inverse scale prediction.
- Model `MUST NOT` lưu/load checkpoint hoặc quản lý run.
- Model `MUST NOT` tính/report evaluation metric.
- Model `MUST NOT` tự sửa data contract hoặc trainer để phù hợp implementation riêng.

## GATE — Model

- `GATE` shape test với batch thường và batch size 1.
- `GATE` inference-without-target test.
- `GATE` invalid teacher-forcing test trong training/eval modes.
- `GATE` forward/backward smoke test với gradient hữu hạn.
- `GATE` floating dtype và device consistency.
- `GATE` CPU test bắt buộc; MPS/CUDA chỉ chạy khi có sẵn.
- `GATE` Ruff, compileall và pytest pass.
