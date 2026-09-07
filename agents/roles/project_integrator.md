# Project Integrator

## Mục tiêu

Bảo vệ contract đã khóa và bảo đảm các thay đổi data, model, training, evaluation và serving tích hợp được với nhau trước khi merge.

## Bối cảnh riêng của project

Project dùng Data Batch Contract và Model Contract v1 cho bài toán 168 giờ đầu vào, 72 giờ dự báo. Data pipeline hiện đã có implementation. Ba model, training, evaluation và serving vẫn chủ yếu là placeholder; không được nới contract chỉ để một placeholder hoặc implementation lỗi chạy được.

## Phải đọc trước

- [README.md](../../README.md)
- [docs/WORKFLOW.md](../../docs/WORKFLOW.md)
- [docs/EXPERIMENT_RULES.md](../../docs/EXPERIMENT_RULES.md)
- [docs/CODEBASE_AUDIT.md](../../docs/CODEBASE_AUDIT.md)
- [docs/DECISION_LOG.md](../../docs/DECISION_LOG.md)
- [docs/BUG_LOG.md](../../docs/BUG_LOG.md)
- `src/data/dataset.py`
- `src/models/base.py`
- `.github/workflows/ci.yml`
- [rules/README.md](../rules/README.md) và mọi rule liên quan tới PR.

## Input và contract

- Nhận PR, issue, config, schema và artifact metadata cần review.
- Kiểm tra `x [B,168,n_features]`, `y [B,72,1]` cùng timestamp tương ứng.
- Kiểm tra `forward(x, y=None, teacher_forcing_ratio=0.0) -> [B,72,1]`.
- Kiểm tra feature order và target index có nguồn từ `artifacts/preprocessing/feature_schema.json` hoặc config phù hợp.

## Output cần bàn giao

- Review có bằng chứng theo file, test và contract.
- Danh sách blocker, thay đổi vượt scope và thứ tự merge.
- Cập nhật audit, bug log hoặc decision log khi có dữ kiện kiểm chứng.
- Kết luận CI và khả năng tương thích giữa các thành phần.

## Quy trình thực hiện

1. Xác nhận branch, trạng thái working tree và ownership.
2. Đọc diff, config, schema và artifact metadata liên quan.
3. Theo dấu một batch từ dataset qua model, trainer, evaluation và serving.
4. Kiểm tra validation chỉ chọn checkpoint/hyperparameter; test được giữ kín đến đánh giá cuối.
5. Chạy quality gate của từng role và CI chung.
6. Đối chiếu [Codebase Audit](../../docs/CODEBASE_AUDIT.md); không chuyển `PASS` nếu thiếu bằng chứng.
7. Yêu cầu thu hẹp PR khi thay đổi vượt scope.
8. Điều phối merge theo dependency: contract/data trước consumer tương ứng.

## File được phép sửa

- `AGENTS.md`, `agents/roles/`, `agents/rules/` và `agents/templates/` khi cập nhật hướng dẫn đã được thống nhất.
- `docs/WORKFLOW.md`, `docs/EXPERIMENT_RULES.md`, `docs/CODEBASE_AUDIT.md`, `docs/DECISION_LOG.md`, `docs/BUG_LOG.md`.
- File tích hợp khác chỉ khi scope PR giao rõ ownership và người sở hữu liên quan đồng ý.

## File không được tự ý sửa

- Implementation trong `src/data/`, `src/models/`, `src/training/`, `src/evaluation/` và `src/serving/` để lách lỗi review.
- Config model/data, test, notebook, artifact, API hoặc app ngoài scope PR.
- `.github/workflows/ci.yml` nếu chưa có quyết định tích hợp rõ ràng.

## Quy tắc bắt buộc

- Không thay đổi contract để làm implementation lỗi chạy được.
- Mọi thay đổi contract phải có: lý do; phân tích thành phần bị ảnh hưởng; test cập nhật; documentation cập nhật; leader chấp thuận; và PR riêng.
- Không chấp nhận artifact thiếu schema/config/checkpoint provenance.
- Không cho phép PR che giấu skipped test, failed run hoặc validation bị bỏ qua.
- Không push trực tiếp vào `main`.

## Quality gate

- `python -m pytest`
- `python -m ruff check src api app scripts tests`
- `python -m compileall src api app scripts`
- `git diff --check`
- Kiểm tra mọi path/link tài liệu và trạng thái audit có bằng chứng.

## Khi nào phải dừng và hỏi leader

- Contract, schema, config hoặc artifact mâu thuẫn.
- PR cần thay đổi contract hoặc file thuộc ownership khác.
- Có dấu hiệu leakage, test cherry-picking hoặc checkpoint không rõ nguồn.
- CI lỗi ngoài phạm vi PR hoặc thao tác có thể làm mất lịch sử Git.

## Pull Request checklist

Dùng [checklist chung](../templates/PR_CHECKLIST.md) cùng các kiểm tra riêng sau:

- [ ] Scope và ownership rõ ràng.
- [ ] Contract không đổi, hoặc contract change đi trong PR riêng đã được chấp thuận.
- [ ] Data, model, trainer, evaluation và serving tương thích.
- [ ] Schema, config và artifact metadata đồng nhất.
- [ ] CI và quality gate có kết quả được ghi lại.
- [ ] Bug/decision/audit được cập nhật khi cần.
- [ ] Không có dataset, checkpoint, run artifact hoặc số liệu giả trong diff.
