# Pull Request Checklist

## Phạm vi

- [ ] PR chỉ sửa các file thuộc role được giao.
- [ ] Không có thay đổi ngoài mục tiêu PR.
- [ ] Không tự thay đổi Data Batch Contract hoặc Model Contract.
- [ ] Không commit raw data, processed data, checkpoint hoặc run artifact.

## Contract và leakage

- [ ] Input/output tuân thủ contract hiện hành.
- [ ] Không hard-code feature count hoặc target index.
- [ ] Không dùng validation/test để fit preprocessing.
- [ ] Không dùng test set để lựa chọn model.
- [ ] Validation/test/inference không truyền future target vào model.

## Kiểm tra

- [ ] `python -m pytest` pass.
- [ ] `python -m ruff check src api app scripts tests` pass.
- [ ] `python -m compileall src api app scripts` pass.
- [ ] `git diff --check` pass.
- [ ] Đã ghi rõ test bị skip nếu có.

## Tài liệu và bằng chứng

- [ ] Config sử dụng đã được ghi rõ.
- [ ] Đã cập nhật decision hoặc bug log nếu cần.
- [ ] Không tạo log, metric, checkpoint hoặc artifact giả.
- [ ] Mô tả PR nêu rõ file thay đổi và cách kiểm tra.
