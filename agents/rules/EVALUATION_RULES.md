# Evaluation Rules

Áp dụng cho `src/evaluation/`, evaluation scripts, experiment registry, conclusions, audit và figures.

## Protocol

- Persistence baseline `MUST` được tính trên cùng evaluation population với model.
- MAE và RMSE báo cáo chính `MUST` được tính sau inverse scaling về °C.
- Metric `MUST` ghi rõ scale/đơn vị; metric standardized chỉ được dùng bổ sung khi có nhãn rõ.
- Validation `MUST` dùng cho model/hyperparameter/checkpoint selection.
- Test `MUST` chỉ được đánh giá sau khi selection protocol đã khóa.
- Contributor `MUST NOT` điều chỉnh model theo test rồi báo lại cùng test như đánh giá cuối.

## Fair comparison

- Các model `MUST` dùng cùng split, preprocessing/scaler, seed protocol, metric, selection criterion và sample population.
- Comparison `MUST` có metric tổng thể và per-horizon cho 72 forecast steps.
- Error analysis `SHOULD` xem theo mùa và giờ khi timestamp semantics và sample size cho phép.
- `MUST NOT` cherry-pick run chỉ vì test metric cao.
- Failed/incomplete run `MUST` giữ trạng thái thật trong registry.

## Traceability và leakage

- Mỗi metric/prediction/figure `MUST` truy vết tới run ID, model, checkpoint, resolved config, schema, scaler và split.
- Inference `MUST` dùng `y=None`, teacher forcing `0.0`.
- Ground truth `MUST` chỉ đi vào metric bên ngoài model.
- Audit `MUST` dùng `PASS` chỉ khi có bằng chứng; nếu chưa kiểm tra thì dùng `NOT CHECKED`.
- `MUST NOT` tạo prediction, metric, run hoặc figure giả.

## GATE — Evaluation

- `GATE` metric unit test với ví dụ tính tay và per-horizon aggregation.
- `GATE` persistence baseline và inverse-scaling test.
- `GATE` timestamp, horizon và sample alignment.
- `GATE` validation/test protocol cùng fair-comparison metadata.
- `GATE` run/checkpoint/figure traceability đầy đủ.
- `GATE` Ruff, compileall và pytest pass.
