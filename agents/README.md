# Bản đồ hệ thống agent

`agents/` là hệ thống role playbook do nhóm xây dựng riêng cho Jena Weather Forecasting. Đây không phải source code hoặc framework multi-agent. Role không tương ứng cứng với từng thành viên: một thành viên có thể dùng nhiều role và nhiều thành viên có thể dùng chung một role.

[AGENTS.md](../AGENTS.md) là entry point ngắn gọn. Các thành phần trong hệ thống có chức năng khác nhau:

- `roles/` mô tả cách một vai trò thực hiện công việc và phạm vi file ownership.
- [rules/](rules/README.md) chứa quy tắc kỹ thuật bắt buộc.
- `templates/` chứa biểu mẫu tái sử dụng cho PR, decision và bug.
- [docs/](../docs/WORKFLOW.md) lưu quy trình, decision, bug và audit thực tế của project.
- `src/` chứa implementation chạy thật hoặc scaffold được đánh dấu trong code.

## Điều hướng theo công việc

| Loại công việc | Role | Rules chính |
| ------------------------------ | ----------------------------- | ----------------------------------------------------------- |
| Contract, CI, tích hợp, review | [roles/project_integrator.md](roles/project_integrator.md) | [Tất cả rule liên quan](rules/README.md) |
| Data pipeline | [roles/data_engineer.md](roles/data_engineer.md) | [rules/DATA_RULES.md](rules/DATA_RULES.md), [rules/VISUALIZATION_RULES.md](rules/VISUALIZATION_RULES.md) |
| Model | [roles/model_developer.md](roles/model_developer.md) | [rules/MODEL_RULES.md](rules/MODEL_RULES.md) |
| Training | [roles/training_engineer.md](roles/training_engineer.md) | [rules/TRAINING_RULES.md](rules/TRAINING_RULES.md), [rules/MODEL_RULES.md](rules/MODEL_RULES.md) |
| Evaluation và audit | [roles/evaluation_auditor.md](roles/evaluation_auditor.md) | [rules/EVALUATION_RULES.md](rules/EVALUATION_RULES.md), [rules/VISUALIZATION_RULES.md](rules/VISUALIZATION_RULES.md) |
| API và Streamlit | [roles/serving_developer.md](roles/serving_developer.md) | [rules/MODEL_RULES.md](rules/MODEL_RULES.md), [rules/EVALUATION_RULES.md](rules/EVALUATION_RULES.md) |

## Sử dụng template

- Pull Request dùng [templates/PR_CHECKLIST.md](templates/PR_CHECKLIST.md).
- Decision mới dùng [templates/DECISION_TEMPLATE.md](templates/DECISION_TEMPLATE.md), sau đó ghi record vào [docs/DECISION_LOG.md](../docs/DECISION_LOG.md).
- Bug mới dùng [templates/BUG_TEMPLATE.md](templates/BUG_TEMPLATE.md), sau đó ghi record vào [docs/BUG_LOG.md](../docs/BUG_LOG.md).

Role playbook không thay thế review ownership. Công việc chạm nhiều role phải được Project Integrator xác nhận phạm vi trước khi sửa.
