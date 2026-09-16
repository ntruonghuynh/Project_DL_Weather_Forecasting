# Quy tắc thực nghiệm

## Contract đánh giá

- Input lịch sử là 168 giờ; dự báo là 72 giờ; target là `T (degC)` theo feature schema của run.
- Prediction artifact dùng `y_true`, `y_pred` với shape `[N,72]` hoặc `[N,72,1]`, và nên có `target_timestamps [N,72]` cùng `last_observed_target [N]`.
- Prediction metadata bắt buộc có `run_id`, model name/version, split, population ID, checkpoint, resolved config, schema hash và scaler checksum.
- Inference validation/test luôn gọi model với `y=None` và `teacher_forcing_ratio=0.0`. Ground truth chỉ được dùng bên ngoài model.
- MAE/MSE/RMSE chính được tính sau inverse scale; MAE/RMSE có đơn vị °C, MSE có đơn vị °C².

## Vòng đời run

- `run_id` bất biến. Retry do hạ tầng phải tạo run mới và ghi `parent_run_id`; resume chỉ dùng `last.pt` tương thích của cùng run.
- Phải lưu seed, resolved config, Git commit, môi trường, schema/scaler identity, checkpoint và trạng thái thật.
- Run lỗi hoặc chưa hoàn tất vẫn được giữ trong registry; không được che giấu, viết lại hoặc cherry-pick theo test metric.

## Selection và final test

1. Development chỉ dùng train/validation.
2. Tất cả model phải dùng cùng split, preprocessing/scaler, seed protocol, metric và population ID.
3. Persistence baseline được tính trên đúng cùng sample population.
4. Candidate có validation RMSE tốt nhất chỉ được promotion khi cải thiện baseline tối thiểu 3%.
5. `selection_manifest.json` phải được khóa, có provenance và người phê duyệt trước khi test được mở.
6. Final test chỉ được bắt đầu một lần cho đúng `run_id` đã khóa. `final_test_audit.json` được tạo ngay khi bắt đầu.
7. Kết quả test chỉ dùng báo cáo cuối; tuyệt đối không quay lại đổi checkpoint, config, preprocessing hay selection criterion.

## Figure và báo cáo

- Figure phải ghi model, run, split, đơn vị và được đăng ký trong `figure_manifest.csv`.
- Caption phải có Finding → Impact → Decision và truy vết được tới metrics/prediction artifact.
- Không tạo metric, prediction, figure, checkpoint hoặc demo artifact giả.

## Lệnh chuẩn

Xem [TV6 Evaluation and Demo Guide](EVALUATION_DEMO.md) để chạy evaluation, comparison, bundle và demo.
