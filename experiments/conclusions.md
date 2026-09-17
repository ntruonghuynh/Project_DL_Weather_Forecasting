# Kết luận thực nghiệm

Ba real runs dùng cùng validation population
`jena_validation_fac7f994f7f54901_in168_out72` gồm 10.026 cửa sổ, cùng input
168 giờ, horizon 72 giờ, schema và train-fitted scaler. Train dùng
`data_stride=6`; validation dùng stride 1. Đây không phải
full-overlapping-window training.

| Model | MAE °C | MSE °C² | RMSE °C | Improvement vs persistence |
|---|---:|---:|---:|---:|
| Attention LSTM | 2.620241 | 11.487950 | 3.389388 | 35.3546% |
| Seq2Seq LSTM | 2.617508 | 11.530652 | 3.395681 | 35.2346% |
| Transformer | 3.593824 | 22.563722 | 4.750129 | 9.4014% |

Persistence validation RMSE là 5.243047 °C. Validation gate yêu cầu cải thiện
tối thiểu 3%; cả ba model vượt gate. Attention LSTM run
`seq2seq_attention_20260917_100234_95dc80` được chọn vì có validation RMSE thấp
nhất, không phải vì test metric. Transformer không phải model thắng.

Final test của locked candidate: MAE 2.721700 °C, MSE 12.300567 °C² và RMSE
3.507216 °C.

The current candidate was selected exclusively using validation results. The
test split had been accessed during an earlier invalid development iteration
and therefore is not considered a pristine unseen holdout. Those earlier test
results were not used to select or modify the final candidate.

Nguồn authoritative: `experiments/experiment_registry.csv`,
`experiments/validation_comparison.csv`, `experiments/selection_manifest.json`
và `experiments/final_test_audit.json`.
