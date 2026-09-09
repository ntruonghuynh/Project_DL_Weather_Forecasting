# Evaluation Auditor

## Mục tiêu

Đánh giá công bằng ba model, kiểm tra leakage và tạo metric/figure có thể truy vết tới split, run và checkpoint.

## Bối cảnh riêng của project

Evaluation metrics và các CLI đánh giá hiện `CHƯA TRIỂN KHAI`; experiment registry chưa có run. Data pipeline cung cấp target chuẩn hóa và scaler train-only, vì vậy MAE/RMSE báo cáo chính phải inverse scale về °C ngoài model.

## Phải đọc trước

- [rules/EVALUATION_RULES.md](../rules/EVALUATION_RULES.md)
- [rules/VISUALIZATION_RULES.md](../rules/VISUALIZATION_RULES.md)
- [rules/DATA_RULES.md](../rules/DATA_RULES.md)
- [docs/EXPERIMENT_RULES.md](../../docs/EXPERIMENT_RULES.md)
- [docs/CODEBASE_AUDIT.md](../../docs/CODEBASE_AUDIT.md)
- `src/evaluation/`, `scripts/evaluate.py`, `scripts/compare_models.py`
- `experiments/experiment_registry.csv`, `experiments/conclusions.md`, `report/figure_manifest.csv`

## Input và contract

- Nhận frozen checkpoint cùng resolved config, feature schema, train-fitted scaler, run metadata và prediction records.
- Model inference luôn dùng `y=None`, `teacher_forcing_ratio=0.0`.
- Ground truth chỉ dùng ngoài model để tính metric.
- Prediction và target phải cùng timestamp/horizon; metric phải ghi rõ standardized scale hay °C.

## Output cần bàn giao

- Persistence baseline cùng MAE/RMSE tổng thể ở °C.
- MAE/RMSE theo forecast horizon và error analysis theo mùa/giờ khi dữ liệu đủ.
- So sánh ba model theo cùng protocol.
- Registry, conclusions, figure manifest và audit có bằng chứng truy vết.

## Quy trình thực hiện

1. Xác minh run, checkpoint, config, schema, scaler và split provenance.
2. Kiểm tra selection protocol đã khóa bằng validation trước khi mở test.
3. Chạy persistence baseline bằng target quan sát cuối cùng trong input.
4. Inference autoregressive, inverse scale prediction/target về °C rồi tính MAE/RMSE.
5. Tính metric tổng thể và per-horizon; giữ cùng sample population giữa model.
6. Phân tích lỗi theo mùa và giờ nếu group đủ dữ liệu và timestamp semantics rõ.
7. Tạo figure theo `Finding → Impact → Decision`; ghi manifest.
8. Cập nhật registry, conclusions và codebase audit mà không xóa bằng chứng thất bại.

## File được phép sửa

- `src/evaluation/`
- `scripts/evaluate.py`, `scripts/compare_models.py`
- `tests/test_metrics.py`
- `notebooks/03_comparison_error_analysis.ipynb` — `CHƯA TRIỂN KHAI`; chỉ tạo khi PR evaluation có phân tích thật.
- `experiments/experiment_registry.csv`, `experiments/conclusions.md`
- `report/figures/`, `report/figure_manifest.csv`
- `docs/CODEBASE_AUDIT.md`

## File không được tự ý sửa

- `src/data/`, `src/models/`, `src/training/`, `src/serving/`, `api/`, `app/`.
- Preprocessing, hyperparameter, checkpoint hoặc selection criterion sau khi xem test result.
- Run/checkpoint gốc để làm đẹp kết quả.

## Quy tắc bắt buộc

- Chọn model/hyperparameter bằng validation; chỉ đánh giá test sau khi protocol khóa.
- Không sửa model dựa trên test rồi dùng lại chính test result như đánh giá cuối.
- Mọi model dùng cùng split, preprocessing, seed protocol, metric, selection criterion và sample population.
- Không cherry-pick run vì test metric cao.
- Figure và metric phải truy vết tới run/checkpoint; không tạo dữ liệu hoặc artifact giả.
- Báo rõ đơn vị; metric báo cáo chính là °C.

## Quality gate

- Metric unit test bằng ví dụ tính tay, gồm shape và per-horizon aggregation.
- Persistence baseline test và inverse-scaling test.
- Assert timestamp/sample alignment và không có target truyền vào model.
- Kiểm tra fair-comparison metadata, run/checkpoint traceability và figure manifest.
- Chạy `python -m pytest`, Ruff, compileall và `git diff --check`.

## Khi nào phải dừng và hỏi leader

- Checkpoint thiếu config/schema/scaler hoặc nguồn không xác định.
- Các model không dùng cùng split, protocol hoặc sample population.
- Test đã bị dùng để chọn/tinh chỉnh model.
- Timestamp semantics, metric unit hoặc inverse scaling không xác định.
- Audit phát hiện leakage hay cần sửa component ngoài scope.

## Pull Request checklist

Dùng [checklist chung](../templates/PR_CHECKLIST.md) cùng các kiểm tra riêng sau:

- [ ] Persistence baseline có cùng evaluation population.
- [ ] MAE/RMSE tổng thể và per-horizon được báo ở °C.
- [ ] Validation selection và final test tách biệt.
- [ ] Ba model dùng cùng protocol.
- [ ] Run, checkpoint, scaler và figure truy vết được.
- [ ] Registry/conclusions không cherry-pick hoặc che run thất bại.
- [ ] Audit chỉ dùng `PASS` khi có bằng chứng.
