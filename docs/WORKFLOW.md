# Quy trình làm việc

## Nhánh Git, commit và yêu cầu hợp nhất

Mỗi thay đổi nên được thực hiện trên một nhánh riêng, sử dụng các commit nhỏ và rõ nghĩa, sau đó gửi yêu cầu hợp nhất để xem xét. Không commit dữ liệu cục bộ, thông tin bí mật, điểm kiểm tra hoặc đầu ra của các lần chạy.

## Các quy ước giao tiếp dùng chung

- Lô dữ liệu: `x [B, 168, n_features]`, `y [B, 72, 1]`, kèm dấu thời gian đầu vào và dấu thời gian mục tiêu được căn chỉnh tương ứng.
- Mô hình: `forward(x, y=None, teacher_forcing_ratio=0.0) -> [B, 72, 1]`; bộ giải mã bắt đầu từ giá trị mục tiêu cuối cùng trong đầu vào; kỹ thuật ép theo dữ liệu thật chỉ được phép sử dụng khi huấn luyện.
- Bản ghi dự báo: gồm dấu thời gian, giá trị thực, giá trị dự báo, định danh mô hình, tập dữ liệu và mã lần chạy.
- Lần chạy: gồm mã bất biến, trạng thái vòng đời, lần chạy cha tùy chọn, cấu hình đã được tổng hợp, hạt giống ngẫu nhiên, chỉ số đánh giá, điểm kiểm tra và kết quả dự báo.

## Lệnh huấn luyện

Giao diện dự kiến: `python scripts/train.py --config configs/<model>.yaml`. Lệnh này chưa hoạt động cho đến khi thành viên phụ trách huấn luyện triển khai.

## Bàn giao sản phẩm

Mỗi lần bàn giao phải có cấu hình đã được tổng hợp, siêu dữ liệu của lần chạy, chỉ số đánh giá, `best.pt`, `last.pt`, kết quả dự báo và lệnh tái tạo chính xác. Các sản phẩm dung lượng lớn phải được lưu ngoài Git.

## Định nghĩa hoàn thành (Definition of Done)

Thay đổi chỉ được xem là hoàn thành khi tuân thủ quy ước giao tiếp, có kiểm thử phù hợp, lưu đủ siêu dữ liệu phục vụ tái lập, có tài liệu bàn giao, đã được xem xét, không gây rò rỉ dữ liệu và không đưa sản phẩm sinh tự động vào Git.

## Báo cáo trở ngại

Báo cáo trở ngại phải nêu rõ người phụ trách, quy ước giao tiếp bị ảnh hưởng, bằng chứng, các kiểm tra đã thử, mức độ ảnh hưởng và quyết định cần được đưa ra. Không được che giấu các lần chạy thất bại.
