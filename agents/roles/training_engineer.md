# Training Engineer

## Mục tiêu

Triển khai training và validation loop tái lập được, chọn checkpoint chỉ bằng validation và lưu run metadata có cấu trúc.

## Bối cảnh riêng của project

`src/training/trainer.py` và `scripts/train.py` hiện `CHƯA TRIỂN KHAI`. Dataset contract đã có, còn ba model vẫn là placeholder. Không được tạo run giả hoặc tuyên bố training thành công trước khi model và loop thực sự chạy hết.

## Phải đọc trước

- [rules/TRAINING_RULES.md](../rules/TRAINING_RULES.md)
- [rules/MODEL_RULES.md](../rules/MODEL_RULES.md)
- [rules/DATA_RULES.md](../rules/DATA_RULES.md)
- [docs/WORKFLOW.md](../../docs/WORKFLOW.md)
- [docs/EXPERIMENT_RULES.md](../../docs/EXPERIMENT_RULES.md)
- `src/data/dataloader.py`, `src/models/base.py`, `src/training/`, `src/utils/`, `scripts/train.py`

## Input và contract

- Nhận resolved config, seed, model đúng contract và train/validation DataLoader.
- Train được dùng `y` cho loss và teacher forcing.
- Validation gọi model với `y=None`, `teacher_forcing_ratio=0.0`; `y` chỉ dùng ngoài model để tính loss.
- Test không được xuất hiện trong training loop.

## Output cần bàn giao

Mỗi run hoàn tất dự kiến có cấu trúc sau; hiện persistence đầy đủ là `CHƯA TRIỂN KHAI`:

```text
runs/<model>/<run_id>/
├── resolved_config.yaml
├── metrics.csv
├── train.log
├── best_checkpoint.pt
├── final_metrics.json
└── environment.json
```

Metadata phải chứa seed, resolved config, Git commit, trạng thái run, model identity và validation criterion. Run artifacts không được commit lên Git.

## Quy trình thực hiện

1. Resolve và validate config; ghi seed và Git commit trước khi train.
2. Chọn CPU/MPS/CUDA theo config và runtime; chuyển batch/model nhất quán.
3. Chạy smoke training ngắn trước full training.
4. Train với loss, optimizer, scheduler, teacher-forcing schedule và gradient clipping theo config.
5. Validation autoregressive với `y=None`; không dùng target làm input.
6. Early stopping và best checkpoint chỉ dựa trên validation criterion.
7. Ghi structured log, environment và trạng thái thất bại/thành công trung thực.
8. Nếu hỗ trợ resume, kiểm tra metadata/config compatibility trước khi tiếp tục.

## File được phép sửa

- `src/training/`
- `src/utils/seed.py`, `src/utils/run_manager.py`, `src/utils/io.py`
- `scripts/train.py`
- `tests/test_training.py` — `CHƯA TRIỂN KHAI`; tạo trong PR training khi có test thật.

## File không được tự ý sửa

- `src/data/`, `src/models/`, `src/evaluation/`, `src/serving/`, `api/`, `app/`.
- Config/data schema hoặc model implementation để né lỗi integration.
- `runs/`, checkpoint, log và metrics dưới dạng nội dung version-control.

## Quy tắc bắt buộc

- Training phải config-driven và reproducible.
- Train có thể dùng target; validation tuyệt đối không truyền target vào model.
- Test set phải sealed và không tham gia loop, scheduler, early stopping hay selection.
- Best checkpoint chỉ theo validation criterion đã khai báo.
- Gradient clipping chỉ áp dụng khi config bật và phải log giá trị.
- Không tạo file giả để mô phỏng run thành công; run lỗi phải có trạng thái failed.

## Quality gate

- Smoke training hoàn tất trên fixture nhỏ.
- Test validation gọi model với `y=None` và teacher forcing `0.0`.
- Test optimizer step, scheduler, clipping, early stopping và checkpoint metadata theo phần triển khai.
- Test seed reproducibility trong giới hạn backend.
- Resume test nếu resume được triển khai.
- `python -m pytest`, Ruff, compileall và `git diff --check` pass.

## Khi nào phải dừng và hỏi leader

- Model chưa đáp ứng forward contract hoặc dataset batch không hợp lệ.
- Validation cần target làm model input để chạy.
- Selection criterion chưa được chốt hoặc test bị đưa vào training.
- Resume/checkpoint thiếu config, schema, Git commit hoặc nguồn gốc.
- Cần sửa ownership của data/model/evaluation.

## Pull Request checklist

Dùng [checklist chung](../templates/PR_CHECKLIST.md) cùng các kiểm tra riêng sau:

- [ ] Seed, resolved config và Git commit được lưu.
- [ ] Validation gọi `y=None`, teacher forcing `0.0`.
- [ ] Test set không xuất hiện trong training/selection.
- [ ] Best checkpoint dựa trên validation.
- [ ] Device, clipping, scheduler và early stopping theo config.
- [ ] Run status/log phản ánh đúng kết quả.
- [ ] Không commit run artifact hoặc checkpoint.
