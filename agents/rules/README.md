# Quy tắc kỹ thuật bắt buộc

`agents/rules/` chứa các quy tắc có tính bắt buộc. Chúng khác với [roles/](../roles/project_integrator.md), nơi mô tả trách nhiệm theo vai trò, và [docs/](../../docs/WORKFLOW.md), nơi giải thích quy trình và lưu lịch sử dự án.

Các từ khóa được dùng thống nhất:

- `MUST`: bắt buộc thực hiện.
- `MUST NOT`: bị cấm.
- `SHOULD`: nên thực hiện; mọi ngoại lệ phải có lý do và bằng chứng.
- `GATE`: điều kiện kiểm tra phải đạt trước khi bàn giao.

| Rule | Phạm vi |
| --- | --- |
| [DATA_RULES.md](DATA_RULES.md) | Validation, preprocessing, split, scaling, window và batch |
| [MODEL_RULES.md](MODEL_RULES.md) | Forward contract và ranh giới trách nhiệm model |
| [TRAINING_RULES.md](TRAINING_RULES.md) | Reproducibility, loop, selection, checkpoint và logs |
| [EVALUATION_RULES.md](EVALUATION_RULES.md) | Baseline, metric, fair comparison và traceability |
| [VISUALIZATION_RULES.md](VISUALIZATION_RULES.md) | Figure có bằng chứng và không gây hiểu sai |

Khi rule mâu thuẫn với code hiện tại, contributor `MUST NOT` âm thầm sửa một phía. Dừng, ghi bằng chứng và báo Project Integrator/leader. Thay đổi contract phải theo quy trình trong [project_integrator.md](../roles/project_integrator.md).
