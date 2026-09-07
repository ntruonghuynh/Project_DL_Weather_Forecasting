# Visualization Rules

Mọi visualization phải phục vụ chuỗi lập luận:

```text
Finding → Impact → Decision
```

## Nội dung bắt buộc

- Figure `MUST` có tiêu đề, nhãn trục và đơn vị.
- Figure `MUST` có legend khi có nhiều series/category.
- Figure `MUST` ghi data split hoặc phạm vi thời gian.
- Figure thí nghiệm `MUST` ghi run/model identifier.
- Caption hoặc Markdown đi kèm `MUST` nêu `Finding`, `Impact` và `Decision`.
- Nếu không có decision ngay, `Decision` `MUST` ghi rõ cần thêm bằng chứng nào; không được phát minh kết luận.

## Tính trung thực và khả năng so sánh

- Figure `MUST NOT` chỉ dùng để trang trí.
- Trục `MUST NOT` bị cắt hoặc chọn range gây hiểu sai; nếu cần zoom, `MUST` ghi rõ.
- Before/after `SHOULD` đặt cạnh nhau và dùng cùng encoding.
- Comparison `MUST` dùng cùng scale, đơn vị, split và aggregation.
- Missing data, filtered sample và uncertainty `SHOULD` được thể hiện hoặc chú thích khi ảnh hưởng kết luận.
- `MUST NOT` tạo figure, metric hoặc data giả.

## Xuất bản và truy vết

- Figure đưa vào report `MUST` được xuất từ code có thể chạy lại.
- Figure report `MUST` có entry trong `report/figure_manifest.csv` với path, run ID, split và mô tả.
- File figure `MUST` nằm trong `report/figures/` khi được đưa vào report.
- Source run/checkpoint/config/schema/scaler `MUST` truy vết được từ manifest hoặc registry.

## GATE — Visualization

- `GATE` tiêu đề, trục, đơn vị, legend và phạm vi dữ liệu đầy đủ.
- `GATE` caption theo `Finding → Impact → Decision`.
- `GATE` comparison dùng cùng scale và population.
- `GATE` manifest trỏ tới file tồn tại và run có provenance.
- `GATE` figure được tái tạo từ code, không chỉnh số liệu thủ công.
