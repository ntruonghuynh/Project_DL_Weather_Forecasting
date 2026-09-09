# Training Rules

Áp dụng cho `src/training/`, training utilities và `scripts/train.py`.

## Reproducibility và config

- Training `MUST` chạy từ resolved config; hyperparameter `MUST NOT` chỉ tồn tại dưới dạng hằng số ẩn trong loop.
- Mỗi run `MUST` lưu seed, resolved config, Git commit, model identity, environment và run status.
- Seed `MUST` được áp dụng cho các thư viện/backend phù hợp trước khi tạo model/DataLoader.
- Full training `SHOULD` chỉ chạy sau khi smoke training hoàn tất.

## Loop và selection

- Train loop `MAY` truyền `y` vào model cho teacher forcing và `MUST` dùng target để tính loss.
- Validation loop `MUST` gọi model với `y=None`, `teacher_forcing_ratio=0.0`; target chỉ dùng ngoài model để tính loss/metric.
- Test set `MUST` được giữ kín và `MUST NOT` xuất hiện trong training, scheduler, early stopping hoặc model selection.
- Early stopping và best checkpoint `MUST` chỉ dựa trên validation criterion đã khai báo.
- Gradient clipping `MUST` áp dụng đúng ngưỡng và thời điểm nếu config bật; nếu tắt thì `MUST NOT` âm thầm dùng giá trị khác.
- Optimizer và scheduler `MUST` được tạo từ config đã resolve.

## Checkpoint và logs

- Checkpoint `MUST` chứa hoặc tham chiếu model state, optimizer/scheduler state cần thiết, epoch/step, best validation value, config, seed, Git commit, model/run ID và schema identity.
- Structured logs `MUST` phân biệt train/validation, epoch/step, metric name, value và trạng thái run.
- Run lỗi hoặc chưa hoàn tất `MUST NOT` được báo là thành công.
- `MUST NOT` tạo log, metric hoặc checkpoint giả.
- Run artifacts `MUST NOT` được commit.
- Resume, nếu triển khai, `MUST` kiểm tra compatibility trước khi khôi phục state.

## GATE — Training

- `GATE` smoke run thực hiện optimizer step và validation autoregressive.
- `GATE` spy/assertion chứng minh validation không truyền target vào model.
- `GATE` best checkpoint/early stopping chỉ phản ứng với validation criterion.
- `GATE` checkpoint metadata và failure status đầy đủ.
- `GATE` reproducibility test phù hợp backend; mọi giới hạn phải được ghi rõ.
- `GATE` Ruff, compileall và pytest pass.
