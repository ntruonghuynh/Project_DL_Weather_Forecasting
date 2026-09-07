# Model Developer

## Mục tiêu

Triển khai đúng một model được giao mà không thay đổi contract chung, data pipeline hoặc trainer.

## Bối cảnh riêng của project

Ba model đều đang `CHƯA TRIỂN KHAI`: Encoder–Decoder LSTM, Attention LSTM và Transformer hiện là placeholder. Model nhận 168 giờ dữ liệu đã preprocess và trả 72 giờ dự báo target `T (degC)` trên standardized scale.

## Phải đọc trước

- [rules/MODEL_RULES.md](../rules/MODEL_RULES.md)
- [rules/DATA_RULES.md](../rules/DATA_RULES.md)
- [docs/WORKFLOW.md](../../docs/WORKFLOW.md)
- `src/models/base.py`, `src/data/dataset.py`
- `artifacts/preprocessing/feature_schema.json`
- Config và implementation đúng model được giao.

## Input và contract

- Input: `x [B,168,n_features]`.
- Training có thể truyền `y [B,72,1]` và tỷ lệ teacher forcing hợp lệ.
- Validation, test, inference: `y=None`, `teacher_forcing_ratio=0.0`.
- Output: floating tensor `[B,72,1]` cùng device với `x`.
- Nếu decoder cần start value, lấy target quan sát cuối cùng theo target index từ schema/config.

## Output cần bàn giao

- Implementation và config của một model duy nhất.
- Test shape, batch size 1, inference không target, teacher-forcing invalid, forward/backward và CPU.
- Mô tả hyperparameter, interface phụ và giới hạn của model.

## Quy trình thực hiện

1. Xác nhận model ownership và chỉ chọn một cặp implementation/config.
2. Đọc feature schema; truyền `n_features`, target index và horizon qua config/constructor.
3. Implement forward đúng contract và gọi validator chung phù hợp.
4. Tách attention weights qua interface riêng nếu cần phân tích.
5. Viết test độc lập với raw CSV và checkpoint.
6. Chạy smoke forward/backward trên CPU; kiểm tra device khác chỉ khi có sẵn.

## File được phép sửa

- LSTM: `src/models/seq2seq_lstm.py`, `configs/lstm.yaml`.
- Attention: `src/models/seq2seq_attention.py`, `configs/attention.yaml`.
- Transformer: `src/models/transformer.py`, `configs/transformer.yaml`.
- `tests/test_model_shapes.py` hoặc test model mới thuộc đúng model được giao.

Chỉ sửa một nhóm ownership trong mỗi task/PR, trừ khi leader chấp thuận phạm vi khác.

## File không được tự ý sửa

- `src/models/base.py`.
- `src/data/`, `src/training/`, `src/evaluation/`, `src/serving/`, `api/`, `app/`.
- Config và implementation của hai model không được giao.

## Quy tắc bắt buộc

- Không hard-code feature count hoặc target index.
- Không tự đọc CSV, preprocess, inverse scale, tính metric hoặc lưu checkpoint trong model.
- Không tự sửa trainer.
- Teacher forcing chỉ dùng khi `model.training` là true.
- Inference phải chạy với `y=None`; hỗ trợ `batch_size=1`.
- Output phải floating, đúng shape và cùng device với `x`.
- Attention weights không được phá prediction contract.

## Quality gate

- Shape test `[B,72,1]` và batch-size-one test.
- Inference-without-target test.
- Invalid teacher-forcing test, gồm ngoài `[0,1]` và nonzero khi eval.
- Forward/backward smoke test; gradient hữu hạn.
- CPU bắt buộc; MPS/CUDA chỉ khi runtime có sẵn.
- `python -m pytest`, Ruff, compileall và `git diff --check` pass.

## Khi nào phải dừng và hỏi leader

- Contract không đủ để thiết kế decoder start hoặc mask.
- Schema/config thiếu hay mâu thuẫn với dataset.
- Cần sửa `src/models/base.py`, trainer, data pipeline hoặc model khác.
- Test chung thất bại do component ngoài ownership.

## Pull Request checklist

Dùng [checklist chung](../templates/PR_CHECKLIST.md) cùng các kiểm tra riêng sau:

- [ ] Chỉ một model/config được sửa.
- [ ] Không hard-code `18` hoặc target index `1`.
- [ ] Forward signature và prediction contract giữ nguyên.
- [ ] Eval/inference không nhận target tương lai.
- [ ] Batch size 1, dtype và device đúng.
- [ ] Forward/backward và validation tests pass.
- [ ] Không có checkpoint, run artifact hoặc metric giả.
