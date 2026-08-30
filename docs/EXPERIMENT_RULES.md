# Quy tắc thực nghiệm

- Không được tinh chỉnh siêu tham số, bộ tiền xử lý, ngưỡng hoặc lựa chọn mô hình trên tập kiểm thử.
- Chỉ sử dụng tập xác thực để lựa chọn mô hình.
- Chỉ chạy đánh giá cuối cùng trên tập kiểm thử sau khi danh mục lựa chọn mô hình đã được khóa.
- Phải lưu hạt giống ngẫu nhiên, cấu hình đã được tổng hợp, mã lần chạy bất biến, thông tin môi trường và siêu dữ liệu của lần chạy.
- Phải giữ lại và đánh dấu rõ các lần chạy thất bại; không được che giấu, viết lại hoặc âm thầm loại bỏ.
- Phải lưu `parent_run_id` khi một lần chạy được phát triển hoặc chạy lại từ lần chạy khác.
