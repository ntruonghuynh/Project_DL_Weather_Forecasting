# NỘI DUNG ĐẦY ĐỦ CHO SLIDE BÁO CÁO DỰ ÁN DEEP LEARNING

**Đề tài:** Dự báo nhiệt độ Jena Climate đa bước từ 168 giờ lịch sử sang 72 giờ tương lai  
**Các kiến trúc:** Encoder–Decoder LSTM, Attention LSTM và Transformer  
**Môn học:** Deep Learning / Time Series Forecasting  
**Phạm vi tài liệu:** Bản nội dung đầy đủ để nhóm dựng slide và tự rút gọn khi thiết kế  
**Nguyên tắc:** Chỉ đưa kết quả thực nghiệm lên slide khi có prediction artifact, metadata, checkpoint, scaler và manifest truy vết được

---

## CẤU TRÚC BÀI THUYẾT TRÌNH

1. **Slide 0:** Trang bìa
2. **Slide 1:** Bài toán dự báo và mục tiêu dự án
3. **Slide 2:** Dữ liệu Jena Climate và các contract đã khóa
4. **Slide 3:** Tiền xử lý dữ liệu và kiểm soát rò rỉ
5. **Slide 4:** Encoder–Decoder LSTM
6. **Slide 5:** Attention LSTM với Bahdanau Attention
7. **Slide 6:** Transformer cho dự báo chuỗi thời gian
8. **Slide 7:** Training engine, quản lý run và khả năng tái lập
9. **Slide 8:** Giao thức đánh giá và trạng thái thực nghiệm
10. **Slide 9:** ModelBundle, Predictor và FastAPI
11. **Slide 10:** Streamlit demo và phạm vi ứng dụng
12. **Slide 11:** Kết luận, giới hạn hiện tại và hướng phát triển

---

## SLIDE 0: TRANG BÌA

### Tiêu đề đề xuất

**DỰ BÁO NHIỆT ĐỘ JENA CLIMATE ĐA BƯỚC BẰNG MÔ HÌNH SEQUENCE-TO-SEQUENCE**

### Phụ đề

So sánh Encoder–Decoder LSTM, Attention LSTM và Transformer cho bài toán dự báo từ 168 giờ lịch sử sang 72 giờ tương lai

### Thông tin trình bày

- **Giảng viên hướng dẫn:** [Điền tên giảng viên]
- **Lớp / học phần:** [Điền thông tin]
- **Nhóm thực hiện:** [Điền tên nhóm]
- **Thời gian báo cáo:** [Điền ngày báo cáo]

### Thành viên và phạm vi phụ trách

- **TV1:** Tích hợp dự án, training engine, contract và quality gate
- **TV2:** Data engineering, kiểm tra chất lượng và preprocessing
- **TV3:** Persistence baseline và Encoder–Decoder LSTM
- **TV4:** Attention LSTM và giao diện attention weights
- **TV5:** Transformer và causal masking
- **TV6:** Evaluation, error analysis, ModelBundle, FastAPI và Streamlit demo

### Gợi ý hình ảnh

- Một đường nhiệt độ theo thời gian làm nền nhẹ.
- Một sơ đồ đơn giản thể hiện 7 ngày lịch sử và 3 ngày dự báo.
- Không đặt bảng kết quả hoặc quá nhiều thông tin kỹ thuật trên trang bìa.

---

## SLIDE 1: BÀI TOÁN DỰ BÁO VÀ MỤC TIÊU DỰ ÁN

**Người trình bày gợi ý:** TV1  
**Thông điệp chính:** Dự án xây dựng một hệ thống thống nhất để dự báo 72 giờ nhiệt độ từ 168 giờ quan trắc trước đó, đồng thời bảo đảm ba mô hình được huấn luyện và đánh giá theo cùng một contract.

### 1. Bài toán dự báo

Dữ liệu đầu vào là chuỗi quan trắc khí tượng đa biến trong 168 giờ liên tiếp, tương đương 7 ngày. Hệ thống cần dự báo nhiệt độ không khí `T (degC)` cho 72 giờ tiếp theo, tương đương 3 ngày.

```text
1. Input: 168 giờ quan trắc lịch sử
2. Xử lý: mô hình sequence-to-sequence
3. Output: 72 giá trị nhiệt độ tương lai
```

Đây là bài toán multi-horizon forecasting vì mô hình phải sinh đồng thời một chuỗi tương lai dài, thay vì chỉ dự báo một thời điểm kế tiếp.

### 2. Những khó khăn chính

**Sai số tích lũy trong suy luận tự hồi quy**  
Sau bước dự báo đầu tiên, decoder phải dùng chính giá trị đã dự báo làm đầu vào cho bước tiếp theo. Sai số tại một bước có thể ảnh hưởng tới nhiều bước sau, đặc biệt khi horizon kéo dài tới 72 giờ.

**Quan hệ theo thời gian ở nhiều thang khác nhau**  
Nhiệt độ có tính liên tục ngắn hạn, chu kỳ ngày và biến động theo mùa. Mô hình cần sử dụng cả thông tin gần thời điểm dự báo và thông tin nằm xa trong cửa sổ 168 giờ.

**So sánh công bằng giữa các kiến trúc**  
Khác biệt kết quả chỉ có ý nghĩa khi ba mô hình sử dụng cùng dữ liệu, cùng feature order, cùng split, cùng scaler, cùng giao thức validation và cùng tập mẫu đánh giá.

### 3. Mục tiêu kỹ thuật

- Xây dựng data pipeline tạo mẫu đúng contract 168 giờ input và 72 giờ output.
- Hiện thực hóa ba kiến trúc cùng tuân thủ một Model Contract.
- Huấn luyện với teacher forcing nhưng validation và inference không được nhận target tương lai.
- Đánh giá bằng MAE, MSE và RMSE sau khi inverse scale về °C.
- Đóng gói checkpoint, schema, scaler và config thành một bundle có checksum.
- Cung cấp API dự báo và giao diện demo không tự huấn luyện lại mô hình.

### Gợi ý trực quan

- Dùng một timeline 240 giờ, trong đó 168 giờ đầu là input và 72 giờ sau là output.
- Tách rõ input đa biến và output chỉ gồm nhiệt độ.

---

## SLIDE 2: DỮ LIỆU JENA CLIMATE VÀ CÁC CONTRACT ĐÃ KHÓA

**Người trình bày gợi ý:** TV2  
**Thông điệp chính:** Feature schema và tensor contract là điểm kết nối thống nhất giữa data pipeline, model, evaluation và serving.

### 1. Nguồn dữ liệu

Dự án sử dụng bộ dữ liệu Jena Climate, gồm các quan trắc khí tượng tại Jena, Đức. Dữ liệu gốc có tần suất khoảng 10 phút. Pipeline làm sạch timestamp, xử lý sentinel và tổng hợp dữ liệu về tần suất 1 giờ.

Sau bước resampling, tập dữ liệu có **70.129 mốc giờ**, từ `2009-01-01 00:00:00` đến `2017-01-01 00:00:00`.

### 2. Feature schema

Schema hiện tại định nghĩa **18 model features**:

**14 biến khí tượng gốc**

- Áp suất không khí `p (mbar)`
- Nhiệt độ `T (degC)`
- Nhiệt độ thế `Tpot (K)`
- Điểm sương `Tdew (degC)`
- Độ ẩm tương đối `rh (%)`
- Các biến áp suất hơi `VPmax`, `VPact`, `VPdef`
- Độ ẩm riêng `sh (g/kg)`
- Hàm lượng hơi nước `H2OC (mmol/mol)`
- Mật độ không khí `rho (g/m**3)`
- Tốc độ gió `wv (m/s)`
- Tốc độ gió cực đại `max. wv (m/s)`
- Hướng gió `wd (deg)`

**4 biến thời gian tuần hoàn**

- `hour_sin`, `hour_cos`
- `dayofyear_sin`, `dayofyear_cos`

Các thành phần dùng chung phải đọc feature order, số lượng feature và target index từ schema. Code không được ghi cứng các giá trị này.

### 3. Data Batch Contract v1

```text
x:                 float32 [B, 168, n_features]
y:                 float32 [B, 72, 1]
input_timestamps:  int64   [B, 168]
target_timestamps: int64   [B, 72]
```

Trong schema hiện tại, `n_features = 18` và target là `T (degC)` tại index 1. Hai giá trị này là dữ liệu của schema hiện tại, không phải hằng số mà model được phép giả định.

### 4. Model Contract v1

```python
forward(x, y=None, teacher_forcing_ratio=0.0) -> prediction
```

`prediction` phải là tensor số thực có shape `[B, 72, 1]` và nằm trên cùng device với `x`.

Trong inference và validation:

- `y` bắt buộc bằng `None`.
- `teacher_forcing_ratio` bắt buộc bằng `0.0`.
- Ground truth chỉ được dùng bên ngoài model để tính metric.

### Gợi ý trực quan

- Bảng contract với bốn dòng tensor.
- Một vòng tròn phân nhóm 14 biến khí tượng và 4 biến chu kỳ.
- Đánh dấu riêng target `T (degC)`.

---

## SLIDE 3: TIỀN XỬ LÝ DỮ LIỆU VÀ KIỂM SOÁT RÒ RỈ

**Người trình bày gợi ý:** TV2  
**Thông điệp chính:** Pipeline giữ thứ tự thời gian, chỉ fit scaler trên train và loại các cửa sổ không có ground truth hợp lệ.

### 1. Quy trình xử lý dữ liệu

```text
1. Raw CSV 10 phút
2. Kiểm tra schema và timestamp
3. Loại bản ghi trùng, thay sentinel bằng NaN
4. Resample về 1 giờ
5. Tạo feature chu kỳ
6. Chia train, validation và test theo thời gian
7. Xử lý thiếu theo từng split
8. Fit StandardScaler trên train
9. Tạo sliding windows 168h + 72h
```

### 2. Kết quả làm sạch đã ghi nhận

- Số dòng raw đầu vào: **420.551**.
- Số dòng sau xử lý duplicate timestamp: **420.224**.
- Số nhóm timestamp trùng: **327**.
- Các bản ghi trùng trong dữ liệu kiểm tra là bản ghi giống hệt nhau, không có nhóm xung đột phải lấy trung bình.
- Sentinel `-9999` được phát hiện trong các biến gió và chuyển thành giá trị thiếu trước khi resample.

### 3. Temporal split chính thức

Pipeline chia dữ liệu theo thứ tự thời gian, không random shuffle ở cấp bản ghi:

| Split | Tỷ lệ | Khoảng thời gian | Số mốc giờ |
|---|---:|---|---:|
| Train | 70% | 01/01/2009 00:00 đến 08/08/2014 09:00 | 49.090 |
| Validation | 15% | 08/08/2014 10:00 đến 20/10/2015 16:00 | 10.519 |
| Test | 15% | 20/10/2015 17:00 đến 01/01/2017 00:00 | 10.520 |

Train chứa phần dữ liệu cũ nhất. Validation nằm sau train. Test là phần mới nhất và chỉ được mở sau khi model selection đã khóa.

### 4. Chính sách dữ liệu thiếu

- Input feature được forward-fill theo hướng nhân quả tối đa 3 giờ và độc lập trong từng split.
- Pipeline không backward-fill và không nội suy từ tương lai.
- Target `T (degC)` không được impute.
- Giá trị thiếu còn lại sau bước forward-fill được giữ dưới dạng NaN.
- Dataset loại mọi sliding window có NaN hoặc Inf trong input hoặc target.

Chính sách này tránh tạo ground truth giả và tránh truyền thông tin tương lai vào quá khứ.

### 5. Chuẩn hóa

`StandardScaler` chỉ fit trên 49.090 mốc giờ của train. Validation và test chỉ sử dụng `transform` với mean và scale đã học từ train. Scaler được lưu thành artifact để training, evaluation và serving dùng cùng một phép biến đổi.

### Gợi ý trực quan

- Timeline thể hiện đúng tỷ lệ 70%, 15%, 15%.
- Một cửa sổ 240 giờ với 168 giờ input và 72 giờ target.
- Một ví dụ cửa sổ bị loại do có NaN, không dùng số liệu dự báo giả.

---

## SLIDE 4: ENCODER–DECODER LSTM

**Người trình bày gợi ý:** TV3  
**Thông điệp chính:** LSTM cung cấp baseline học sâu có cấu trúc gọn, nhưng decoder chỉ nhận trạng thái cuối của encoder làm ngữ cảnh.

### 1. Encoder

Encoder nhận tensor `[B, 168, n_features]` và cập nhật hidden state cùng cell state qua từng giờ:

$$
(h_t, c_t) = \operatorname{LSTM}(x_t, h_{t-1}, c_{t-1})
$$

Sau 168 bước, encoder chuyển trạng thái cuối `(h_{168}, c_{168})` cho decoder. Mô hình không trả toàn bộ encoder outputs cho decoder.

### 2. Decoder tự hồi quy

Decoder sinh một giá trị nhiệt độ tại mỗi bước. Giá trị khởi đầu là nhiệt độ quan sát cuối cùng trong input:

$$
d_0 = x_{168,\text{target}}
$$

Tại mỗi bước dự báo:

$$
s_t = \operatorname{LSTMDecoder}(d_{t-1}, s_{t-1})
$$

$$
\hat{y}_t = W_o s_t + b_o
$$

Trong inference, dự báo của bước trước trở thành đầu vào của bước tiếp theo.

### 3. Teacher forcing trong training

Khi model ở training mode và có target, hệ thống có thể thay dự báo bước trước bằng ground truth theo xác suất `teacher_forcing_ratio`. Cấu hình LSTM hiện đặt ratio ban đầu là `0.5` và hỗ trợ decay `0.05` mỗi epoch.

Validation không dùng teacher forcing. Trainer gọi:

```python
model(x, y=None, teacher_forcing_ratio=0.0)
```

### 4. Cấu hình kiến trúc hiện tại

- Hidden size: 128
- Số lớp LSTM: 1
- Dropout: 0.0
- Horizon: 72
- Feature count và target index: đọc từ schema

### 5. Điểm mạnh và giới hạn cần kiểm chứng

LSTM có ít thành phần hơn Transformer và phù hợp làm baseline học sâu. Tuy nhiên, decoder chỉ nhận trạng thái cuối của encoder. Đây là một giới hạn kiến trúc có thể ảnh hưởng tới việc sử dụng thông tin nằm xa trong cửa sổ 168 giờ. Mức ảnh hưởng thực tế phải được xác định bằng kết quả validation, không kết luận trước khi có prediction artifact thật.

### Gợi ý trực quan

- Sơ đồ encoder 168 bước, trạng thái cuối và decoder 72 bước.
- Bảng nhỏ về hyperparameter.
- Không vẽ đường RMSE hoặc tuyên bố sai số tăng sau 48 giờ khi chưa có kết quả thật.

---

## SLIDE 5: ATTENTION LSTM VỚI BAHDANAU ATTENTION

**Người trình bày gợi ý:** TV4  
**Thông điệp chính:** Attention LSTM cho phép mỗi bước decoder tạo một context riêng từ toàn bộ encoder outputs.

### 1. Encoder outputs

Khác với LSTM baseline, attention encoder giữ lại hidden representation tại từng giờ:

$$
H = [h_1, h_2, \ldots, h_{168}]
$$

Decoder có thể tham chiếu toàn bộ chuỗi này thay vì chỉ sử dụng trạng thái cuối.

### 2. Alignment score

Tại bước decoder `t`, Bahdanau attention tính mức liên quan giữa hidden state trước đó của decoder và từng encoder state:

$$
e_{t,i} = v_a^\top \tanh(W_s s_{t-1} + W_h h_i + b_a)
$$

### 3. Attention weights và context

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{k=1}^{168}\exp(e_{t,k})}
$$

$$
c_t = \sum_{i=1}^{168}\alpha_{t,i}h_i
$$

Mỗi hàng attention weights tương ứng với một bước dự báo và có tổng xấp xỉ 1. Context `c_t` thay đổi theo bước decoder.

### 4. Decoder

Decoder ghép giá trị target của bước trước với context:

$$
s_t = \operatorname{LSTMDecoder}([d_{t-1};c_t], s_{t-1})
$$

Output projection sử dụng decoder output cùng context để sinh `ŷ_t`.

### 5. Giao diện attention weights

Prediction contract vẫn chỉ trả `[B, 72, 1]`. Attention weights được lấy qua phương thức riêng:

```python
model.get_last_attention_weights()
```

Shape kỳ vọng là `[B, 72, 168]`.

### 6. Cách trình bày heatmap đúng

Heatmap chỉ được đưa vào báo cáo khi được trích xuất từ:

- Một checkpoint có nguồn gốc rõ ràng.
- Một sample thuộc split đã khai báo.
- Một lần inference thật với `y=None`.
- Metadata gồm run ID, model ID và timestamp.

Notebook hiện có hình heatmap mô phỏng không được dùng làm bằng chứng rằng model đã học chu kỳ 24 giờ. Khi có attention artifact thật, nhóm mới được mô tả vị trí model tập trung và phải tránh diễn giải attention như quan hệ nhân quả.

### Gợi ý trực quan

- Công thức alignment và context.
- Sơ đồ decoder truy vấn 168 encoder states.
- Chừa vị trí cho heatmap thật. Nếu chưa có artifact, ghi rõ “Đang chờ inference từ checkpoint đã khóa”.

---

## SLIDE 6: TRANSFORMER CHO DỰ BÁO CHUỖI THỜI GIAN

**Người trình bày gợi ý:** TV5  
**Thông điệp chính:** Transformer dùng attention để mô hình hóa quan hệ giữa các vị trí và dùng causal mask để ngăn decoder truy cập target tương lai.

### 1. Input projection và positional encoding

Input `[B, 168, n_features]` được ánh xạ sang không gian `d_model = 128`:

$$
z_t = W_x x_t + b_x
$$

Sinusoidal positional encoding bổ sung thông tin vị trí:

$$
PE_{pos,2i} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{pos,2i+1} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

### 2. Encoder

Encoder gồm 3 lớp Transformer Encoder. Mỗi lớp có multi-head self-attention, feed-forward network, residual connection và layer normalization.

### 3. Decoder và causal mask

Decoder nhận một chuỗi target prefix. Token đầu tiên là nhiệt độ cuối cùng trong input. Khi inference, prefix được mở rộng bằng các dự báo đã sinh.

Causal mask có dạng tam giác trên:

$$
M_{i,j} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

Mask ngăn vị trí `i` truy cập các target token ở vị trí lớn hơn `i`.

### 4. Cấu hình hiện tại

- `d_model = 128`
- `nhead = 8`
- 3 encoder layers
- 3 decoder layers
- Feed-forward dimension: 512
- Dropout: 0.1
- Activation: GELU
- Horizon và input length: đọc từ schema/config

### 5. Training và inference

Code hỗ trợ một masked decoder pass khi training với teacher forcing hoàn toàn. Với teacher forcing từng phần hoặc inference, decoder sinh tự hồi quy. Phiên bản hiện tại chưa triển khai KV cache, vì vậy không được tuyên bố đã tối ưu cache hoặc đạt một mức latency cụ thể.

### 6. Điểm cần đánh giá thực nghiệm

Transformer có khả năng biểu diễn quan hệ giữa nhiều vị trí nhưng sử dụng nhiều tham số và bộ nhớ hơn các mô hình LSTM hiện tại. Hiệu quả dự báo, thời gian training và latency inference phải được đo trên cùng phần cứng và cùng population trước khi kết luận.

### Gợi ý trực quan

- Sơ đồ encoder–decoder Transformer.
- Ma trận causal mask có chú thích vị trí được phép và bị chặn.
- Không dùng từ “tốt nhất” hoặc “nhanh nhất” trước khi có benchmark thật.

---

## SLIDE 7: TRAINING ENGINE, QUẢN LÝ RUN VÀ KHẢ NĂNG TÁI LẬP

**Người trình bày gợi ý:** TV1  
**Thông điệp chính:** Training engine tách training và validation, ghi lại provenance của run và áp dụng cùng một vòng lặp cho ba mô hình.

### 1. Training loop

Mỗi epoch gồm hai pha:

**Training**

- Model ở `train()` mode.
- Cho phép target và teacher forcing theo config.
- Tính loss, backpropagation và optimizer step.
- Áp dụng gradient clipping với max norm mặc định bằng 1.0.

**Validation**

- Model ở `eval()` mode.
- Dùng `torch.no_grad()`.
- Gọi model với `y=None` và `teacher_forcing_ratio=0.0`.
- Dùng validation loss cho checkpoint và early stopping.

### 2. Config và optimizer

Cấu hình cơ sở hiện sử dụng:

- Optimizer Adam
- Learning rate `0.001`
- Weight decay `0.0001`
- Loss MSE
- Batch size 64
- Patience 5 epoch
- Seed 42

Trainer có hỗ trợ `ReduceLROnPlateau` và cosine scheduler khi config yêu cầu. Config mặc định hiện chưa bật scheduler, vì vậy báo cáo không được nói scheduler đang hoạt động trong mọi run.

### 3. Callback

- `ModelCheckpointCallback` lưu best và last checkpoint.
- `EarlyStoppingCallback` theo dõi validation loss.
- `MetricLoggerCallback` ghi metric theo epoch vào CSV.

### 4. Provenance của run

Mỗi run lưu:

- Resolved config.
- Seed.
- Python, PyTorch, platform và thông tin thiết bị.
- Git commit tại thời điểm chạy.
- Best checkpoint và last checkpoint.
- Metric theo epoch cùng trạng thái completed hoặc failed.

Các thành phần này hỗ trợ audit và tái lập. Không nên dùng cụm “tái lập 100%” vì khả năng tái lập còn phụ thuộc phần cứng, phiên bản thư viện và tính xác định của operation.

### 5. Trạng thái kiểm thử kỹ thuật

Tại lần kiểm tra ngày 17/09/2026:

- **369 tests passed**.
- **24 tests skipped** vì các cấu hình phần cứng CUDA/MPS không khả dụng trong môi trường kiểm tra.
- **14 warnings**, gồm cảnh báo phiên bản scikit-learn của scaler artifact và cảnh báo nested tensor của Transformer.

Đây là bằng chứng về contract và behavior của code, không phải bằng chứng về độ chính xác dự báo trên dữ liệu thật.

### 6. Trạng thái smoke run và blocker tích hợp

Ba run cục bộ hiện có chỉ chạy 1 epoch trên synthetic smoke batches. Các run này xác nhận training loop, checkpoint và metadata hoạt động nhưng không được dùng làm benchmark mô hình.

Trước khi chạy full experiment, cần thống nhất tên file giữa hai component:

- Preprocessing tạo `train_processed.csv` và `val_processed.csv`.
- Training CLI hiện tìm `train.csv` và `val.csv`.

Nhóm cần sửa mapping này hoặc truyền đúng đường dẫn/tên file trước khi tuyên bố full training hoàn tất.

### Gợi ý trực quan

- Sơ đồ hai pha training và validation.
- Cây thư mục của một run.
- Bảng test snapshot với passed, skipped và warnings.

---

## SLIDE 8: GIAO THỨC ĐÁNH GIÁ VÀ TRẠNG THÁI THỰC NGHIỆM

**Người trình bày gợi ý:** TV6  
**Thông điệp chính:** Evaluation pipeline đã được triển khai, nhưng repository chưa có prediction artifact thật để công bố bảng xếp hạng hoặc chọn model thắng cuộc.

### 1. Input của evaluation

Mỗi model phải bàn giao:

- `y_true` và `y_pred` có shape `[N, 72]` hoặc `[N, 72, 1]`.
- `target_timestamps` thẳng hàng với từng prediction.
- `last_observed_target` để tính persistence baseline.
- Metadata gồm run ID, model name/version, split và population ID.
- Đường dẫn hoặc định danh checkpoint, resolved config, schema hash và scaler checksum.

### 2. Metric chính

Metric phải được tính sau khi inverse scale về độ C:

$$
MAE = \frac{1}{NH}\sum_{n=1}^{N}\sum_{h=1}^{H}|y_{n,h}-\hat{y}_{n,h}|
$$

$$
MSE = \frac{1}{NH}\sum_{n=1}^{N}\sum_{h=1}^{H}(y_{n,h}-\hat{y}_{n,h})^2
$$

$$
RMSE = \sqrt{MSE}
$$

MAE và RMSE có đơn vị °C. MSE có đơn vị °C².

### 3. Persistence baseline

Persistence dự báo mọi bước tương lai bằng target quan sát cuối cùng của input:

$$
\hat{y}^{base}_{n,h} = x_{n,168,target}
$$

Baseline phải dùng đúng các sample mà model được đánh giá. Không được tính baseline trên population khác.

### 4. Fair comparison

Ba model chỉ được so sánh khi cùng:

- Validation split và population ID.
- Số sample và 72 horizon.
- Feature schema và train-fitted scaler.
- Metric definition và đơn vị.
- Selection criterion.

Evaluation tạo metric tổng thể, metric theo horizon, worst cases và phân tích theo giờ/mùa khi timestamp semantics hợp lệ.

### 5. Model selection

Model selection chỉ sử dụng validation RMSE. Candidate tốt nhất phải cải thiện tối thiểu 3% so với persistence baseline:

$$
\frac{RMSE_{baseline}-RMSE_{candidate}}{RMSE_{baseline}} \ge 0.03
$$

Sau khi chọn, hệ thống tạo `selection_manifest.json` bất biến với checksum và người phê duyệt.

### 6. Final test

Test chỉ được mở cho đúng candidate trong selection manifest. `final_test_audit.json` được tạo trước khi đọc prediction test và chặn lần chạy thứ hai. Kết quả test chỉ dùng báo cáo cuối, không quay lại thay model hoặc hyperparameter.

### 7. Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Metric, inverse scaling và persistence baseline | Đã triển khai và có unit test |
| Per-horizon và error analysis | Đã triển khai |
| Fair-comparison gate | Đã triển khai |
| Selection manifest và final-test audit | Đã triển khai |
| Prediction artifact từ full validation run | Chưa có |
| Bảng so sánh ba model ở °C | Chưa có |
| Model được promotion | Chưa xác định |
| Final-test result | Chưa chạy |
| Figure thực nghiệm có manifest | Chưa có |

### 8. Quy tắc cho slide kết quả

Không sử dụng benchmark hard-code, đường metric sinh ngẫu nhiên, cold-front mô phỏng hoặc attention heatmap giả lập làm kết quả dự án. Khi full runs hoàn tất, Slide 8 cần được cập nhật bằng output trực tiếp từ evaluation pipeline và ghi rõ run ID, split, sample count cùng đơn vị.

### Gợi ý trực quan

- Ở phiên bản hiện tại, dùng sơ đồ evaluation pipeline và bảng trạng thái.
- Khi có artifact thật, thay bảng trạng thái bằng bảng validation comparison và tối đa hai figure có provenance.

---

## SLIDE 9: MODELBUNDLE, PREDICTOR VÀ FASTAPI

**Người trình bày gợi ý:** TV6  
**Thông điệp chính:** Serving sử dụng đúng checkpoint, scaler, schema và config của một run đã khóa, không fit lại preprocessing và không huấn luyện trong API.

### 1. ModelBundle v1.0

Bundle gồm:

- `model.pt`: state dictionary của model.
- `scaler.joblib`: train-fitted StandardScaler.
- `feature_schema.json`: feature order, target index và window contract.
- `resolved_config.json`: cấu hình đã dùng để tạo model.
- `metadata.json`: run metadata.
- `bundle_manifest.json`: model identity, constructor kwargs, file mapping và SHA-256 của từng artifact.

Bundle loader kiểm tra file bắt buộc, checksum và run identity. Loader dùng strict state-dict khi phục hồi model và đặt model ở eval mode.

### 2. Predictor

`Predictor` thực hiện các bước:

1. Đọc feature order, target index, input length và horizon từ schema.
2. Kiểm tra đúng 168 observation, feature names và giá trị hữu hạn.
3. Kiểm tra timestamp tăng đều theo cadence 1 giờ.
4. Sắp xếp feature theo schema.
5. Áp dụng scaler đã fit trên train.
6. Gọi model với `y=None` và teacher forcing bằng 0.
7. Kiểm tra output `[1, 72, 1]`.
8. Inverse scale target về °C.
9. Sinh 72 timestamp tương lai và response metadata.

### 3. FastAPI endpoints

**`GET /health`**  
Trả trạng thái `ready` khi Predictor đã tải thành công. Nếu bundle chưa được cấu hình hoặc tải lỗi, endpoint trả `not_ready` cùng chi tiết chẩn đoán.

**`GET /model-info`**  
Trả model name, version, run ID, input length, horizon, số feature, target index, target feature, đơn vị và trạng thái attention availability.

**`POST /predict`**  
Nhận JSON gồm timestamps và observations. Response chứa 72 prediction records, persistence baseline, model metadata, latency của request và attention weights nếu model hỗ trợ.

### 4. Phạm vi kiểm chứng

API, Predictor và bundle được kiểm tra bằng test bundle/fixture. Hiện chưa có production bundle từ model được promotion vì evaluation chính thức chưa hoàn tất. Vì vậy báo cáo không công bố latency trung bình hoặc throughput hệ thống.

### 5. Lệnh chạy stack

```bash
python scripts/run_stack.py --bundle <bundle-dir>
```

Script xác minh bundle trước khi khởi động FastAPI và Streamlit.

### Gợi ý trực quan

- Sơ đồ `Client, FastAPI, Predictor, ModelBundle`.
- Danh sách file trong bundle.
- Một request/response rút gọn, không dùng dữ liệu nhạy cảm hoặc số dự báo giả.

---

## SLIDE 10: STREAMLIT DEMO VÀ PHẠM VI ỨNG DỤNG

**Người trình bày gợi ý:** TV6 hoặc đại diện nhóm  
**Thông điệp chính:** Streamlit là HTTP client của FastAPI và hiển thị lịch sử, forecast, baseline cùng metadata của model.

### 1. Tách biệt giao diện và inference

Streamlit không import checkpoint, không load scaler và không chạy model trực tiếp. Giao diện gửi request tới `POST /predict` và hiển thị response.

Kiến trúc:

```text
1. Người dùng thao tác trên Streamlit
2. Streamlit gửi HTTP JSON tới FastAPI
3. FastAPI gọi Predictor
4. Predictor sử dụng ModelBundle
5. FastAPI trả response để Streamlit hiển thị
```

### 2. Input demo

Giao diện hỗ trợ hai nguồn:

- Upload payload JSON có `timestamps` và `observations`.
- Chọn một JSON thật trong thư mục được cấu hình bằng `JENA_DEMO_PAYLOAD_DIR`.

Configured sample phải có `provenance.synthetic=false`. Script tạo payload chỉ cho phép lấy sample từ train hoặc validation, không lấy sample từ test.

Giao diện hiện không nhận CSV tùy ý và không có sẵn các scenario được gán nhãn như “mùa hè”, “đợt rét” hoặc “cold front”. Nếu nhóm muốn bổ sung scenario, mỗi sample phải được trích từ dữ liệu thật và có provenance.

### 3. Nội dung hiển thị

- Biểu đồ target trong 168 giờ input.
- Bảng 24 observation cuối.
- Forecast 72 giờ ở đơn vị °C.
- Persistence baseline trên cùng input.
- Ground truth tương lai nếu payload có `future_actual`.
- Model name, version, run ID và latency của request.
- Attention heatmap nếu model cung cấp attention weights hợp lệ.

Giao diện hiện hiển thị toàn bộ attention matrix. Chưa có tính năng chọn riêng một forecast hour để khám phá attention.

### 4. Kịch bản live demo đúng quy trình

1. Chuẩn bị ModelBundle từ candidate đã được promotion.
2. Xuất một validation payload thật với source/schema checksum.
3. Khởi động stack bằng bundle đã xác minh.
4. Kiểm tra `/health` và `/model-info`.
5. Chọn hoặc upload payload.
6. Chạy forecast và kiểm tra đủ 72 điểm.
7. Giải thích baseline, model metadata và latency.
8. Chỉ trình bày heatmap khi model đang chạy là Attention LSTM.

### 5. Phạm vi ứng dụng tiềm năng

Kết quả dự báo nhiệt độ 72 giờ có thể là một đầu vào cho:

- Lập kế hoạch tưới tiêu hoặc cảnh báo điều kiện nhiệt độ bất lợi trong nông nghiệp.
- Ước lượng nhu cầu làm mát và sưởi trong vận hành năng lượng.
- Điều khiển HVAC có xét dự báo nhiệt độ ngoài trời.
- Hỗ trợ phân tích chuỗi thời gian khí tượng trong nghiên cứu và giáo dục.

Đây là các hướng ứng dụng tiềm năng. Dự án chưa thực hiện nghiên cứu tác động, chưa đo mức tiết kiệm năng lượng và chưa đủ điều kiện đưa ra cam kết vận hành ngoài thực tế.

### Gợi ý trực quan

- Screenshot thật của Streamlit sau khi đã chạy bằng bundle hợp lệ.
- Một biểu đồ input và forecast.
- Không dùng mockup hoặc screenshot chứa prediction giả mà không ghi rõ là minh họa.

---

## SLIDE 11: KẾT LUẬN, GIỚI HẠN HIỆN TẠI VÀ HƯỚNG PHÁT TRIỂN

**Người trình bày gợi ý:** TV1 hoặc đại diện nhóm  
**Thông điệp chính:** Dự án đã hoàn thiện phần contract và hạ tầng kỹ thuật chính, trong khi kết luận về model tốt nhất vẫn cần full training cùng evaluation artifact thật.

### 1. Những phần đã hoàn thành

- Data pipeline tạo dữ liệu hourly, temporal split, scaler train-only và sliding windows theo contract.
- Encoder–Decoder LSTM, Attention LSTM và Transformer đã tuân thủ prediction shape `[B, 72, 1]`.
- Training engine tách training khỏi autoregressive validation và lưu checkpoint/metadata.
- Evaluation pipeline hỗ trợ metric °C, persistence baseline, per-horizon analysis, model-selection gate và one-shot final-test audit.
- Serving stack gồm ModelBundle, Predictor, FastAPI và Streamlit HTTP client.
- Bộ test hiện tại ghi nhận 369 passed, 24 skipped và 14 warnings trong môi trường kiểm tra ngày 17/09/2026.

### 2. Những kết luận chưa được phép đưa ra

- Chưa có bảng validation comparison chính thức giữa ba model.
- Chưa xác định model tốt nhất.
- Chưa có evidence rằng các model vượt persistence baseline.
- Chưa có final-test MAE/RMSE.
- Chưa có attention heatmap từ checkpoint và sample thật để diễn giải.
- Chưa có latency benchmark trên production bundle.

### 3. Công việc cần hoàn tất trước báo cáo kết quả cuối

1. Sửa mapping tên file giữa preprocessing và training CLI.
2. Chạy full training cho ba model với cùng protocol.
3. Xuất validation predictions trên cùng population.
4. Chạy evaluation và so sánh với persistence baseline.
5. Khóa selection manifest trước khi mở test.
6. Đánh giá final test đúng một lần.
7. Cập nhật Slide 8 bằng số liệu và figure có provenance.
8. Build bundle của candidate đã promotion và chạy live demo.

### 4. Hướng phát triển

- Probabilistic forecasting bằng quantile loss hoặc prediction intervals.
- Bổ sung calibration và uncertainty visualization.
- Đánh giá robustness khi có missing observations hoặc distribution shift.
- Tối ưu inference Transformer và đo latency trên phần cứng mục tiêu.
- Mở rộng từ một trạm sang nhiều trạm bằng mô hình không gian–thời gian khi có dữ liệu phù hợp.

### 5. Kết thúc

**Cảm ơn giảng viên và các bạn đã theo dõi. Nhóm sẵn sàng trả lời câu hỏi về dữ liệu, kiến trúc model, evaluation protocol và serving demo.**

---

## NGUỒN VÀ BẰNG CHỨNG CẦN GẮN KHI DỰNG SLIDE

### Nguồn nội bộ

- `artifacts/preprocessing/feature_schema.json`
- `artifacts/preprocessing/split_metadata.json`
- `configs/data.yaml`
- `configs/base.yaml`
- `configs/lstm.yaml`
- `configs/attention.yaml`
- `configs/transformer.yaml`
- `src/models/seq2seq_lstm.py`
- `src/models/attention_lstm_seq2seq.py`
- `src/models/transformer.py`
- `src/training/trainer.py`
- `src/evaluation/`
- `src/serving/`
- `api/main.py`
- `app/streamlit_app.py`
- `docs/EVALUATION_DEMO.md`
- `experiments/experiment_registry.csv`
- `experiments/conclusions.md`
- `report/figure_manifest.csv`

### Tài liệu học thuật nên trích dẫn

- Bahdanau, Cho và Bengio: *Neural Machine Translation by Jointly Learning to Align and Translate*.
- Vaswani và cộng sự: *Attention Is All You Need*.
- Hochreiter và Schmidhuber: *Long Short-Term Memory*.
- Nguồn phát hành bộ dữ liệu Jena Climate từ Max Planck Institute for Biogeochemistry.

### Quy tắc sử dụng bằng chứng

- Mỗi metric phải truy vết được tới run, checkpoint, config, schema, scaler và split.
- Mỗi figure thực nghiệm phải nằm trong `report/figures/` và có entry trong `report/figure_manifest.csv`.
- Không sử dụng số liệu, prediction, benchmark hoặc hình ảnh mô phỏng như kết quả thật.
- Các số test cần được cập nhật nếu code thay đổi sau ngày 17/09/2026.
- Citation học thuật và nguồn ảnh nên đặt ở speaker notes hoặc footer của slide liên quan.

---

## CHECKLIST TRƯỚC KHI CHUYỂN NỘI DUNG THÀNH SLIDE

- [ ] Tên giảng viên, thành viên và lớp đã được điền.
- [ ] Tỷ lệ split là 70% train, 15% validation và 15% test.
- [ ] Không còn benchmark hard-code hoặc số liệu từ notebook mô phỏng.
- [ ] Không gọi model nào là tốt nhất khi chưa có selection manifest.
- [ ] Nếu có bảng kết quả, toàn bộ metric dùng °C và cùng population.
- [ ] Nếu có test result, ghi cả passed, skipped và warnings.
- [ ] Nếu có heatmap, ảnh được tạo từ checkpoint/sample thật và có run ID.
- [ ] Lệnh demo có `--bundle <bundle-dir>`.
- [ ] Screenshot demo được chụp từ stack chạy thật.
- [ ] Các giới hạn hiện tại vẫn xuất hiện trong phần kết luận.
- [ ] Mọi claim định lượng đều có nguồn hoặc artifact hỗ trợ.
- [ ] Người dựng slide đã rút gọn nội dung phù hợp thời lượng nhưng không làm mất caveat quan trọng.
