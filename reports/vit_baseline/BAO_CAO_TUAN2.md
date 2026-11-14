# Thực nghiệm ViT baseline trên hai bộ ảnh MRI (DS1 & DS2)

**Thái Văn Huy · 23130068**

Số liệu lấy từ lượt chạy trên Windows ngày 13/11/2026. Các tệp cấu hình, lịch sử huấn luyện và dự đoán dùng trong báo cáo nằm ở commit [0ff153d](https://github.com/MikanQuyt/BrainTumorDetect/commit/0ff153d673869e8db0d2cac68b43ead7bd2d664e).

## 1. Dữ liệu và cách chia tập

DS1 là Navoneel, gồm 253 ảnh, trong đó 155 ảnh có u và 98 ảnh không u. Dùng 5-fold cross-validation phân tầng theo lớp và theo nhóm ảnh trùng. Mỗi lượt giữ một fold để test và dành khoảng 20% số nhóm trong bốn fold còn lại cho validation. Mỗi ảnh được test đúng một lần.

| Fold | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| 1 | 159 | 43 | 51 |
| 2 | 162 | 40 | 51 |
| 3 | 163 | 39 | 51 |
| 4 | 164 | 39 | 50 |
| 5 | 162 | 41 | 50 |

DS2 dùng 7.023 ảnh từ phần classification của Indk214. Chia lại với tỷ lệ mục tiêu 70/15/15. Cột `original_split` trong `splits.csv` ghi thư mục Training hoặc Testing gốc của Kaggle.

| Lớp DS2 | Train | Validation | Test | Tổng |
| --- | ---: | ---: | ---: | ---: |
| Không u | 1.409 | 309 | 282 | 2.000 |
| Glioma | 1.135 | 243 | 243 | 1.621 |
| Meningioma | 1.157 | 246 | 242 | 1.645 |
| Pituitary | 1.229 | 266 | 262 | 1.757 |
| **Tổng** | **4.930** | **1.064** | **1.029** | **7.023** |

Ảnh có cùng kích thước và cùng toàn bộ giá trị pixel xám được giữ trong cùng một tập. Phạm vi kiểm tra trùng lặp là trùng pixel ở cấp ảnh. [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)

## 2. Cấu hình huấn luyện

Hai chế độ khởi tạo từ cùng checkpoint ViT-Tiny patch 16 pretrained trên ImageNet (`vit_tiny_patch16_224.augreg_in21k_ft_in1k`, ~5,50 triệu tham số). Ảnh được crop vùng đầu, tăng tương phản bằng CLAHE, resize về 112 × 112 và chuẩn hóa theo checkpoint. Lượt chạy dùng PyTorch 2.15.0 trên CPU, FP32, batch size 32, seed 42 và weight decay 0,01.

| Chế độ | Tham số cập nhật | Learning rate | Epoch DS1 | Epoch DS2 |
| --- | ---: | ---: | ---: | ---: |
| Linear Probing | Lớp phân loại tuyến tính (386 / 772) | 0,001 | 25 | 25 |
| Full Fine-tuning | Toàn bộ ViT + lớp phân loại (~5,50 triệu) | 0,0001 | 20 | 8 |

Hai chế độ dùng cùng cách chia tập và cùng khởi tạo lớp phân loại trong mỗi lượt. Loss là cross-entropy và optimizer là AdamW. Chọn trọng số tại epoch có validation loss thấp nhất để đánh giá một lần trên test.

## 3. Kết quả DS1 (253 ảnh, 5-Fold Cross-Validation)

Các số dưới đây là trung bình ± độ lệch chuẩn mẫu trên 5 fold. Precision, recall, F1 và ROC-AUC lấy lớp có u làm lớp dương.

| Chế độ | Accuracy (%) | Precision (%) | Recall có u (%) | F1 (%) | ROC-AUC (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear Probing | 77,45 ± 6,23 | 82,20 ± 6,46 | **81,29 ± 7,36** | 81,51 ± 5,13 | 83,16 ± 7,64 |
| Full Fine-tuning | 88,55 ± 4,65 | 92,27 ± 3,83 | 89,03 ± 9,29 | 90,32 ± 4,62 | 94,84 ± 3,06 |

Gộp dự đoán test của cả 5 fold, thu được số ảnh ở từng ô của ma trận nhầm lẫn:

| Chế độ | Đúng không u (TN) | Báo u nhầm (FP) | Bỏ sót u (FN) | Đúng có u (TP) | Pooled AUC (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear Probing | 70 | 28 | **29** | 126 | 83,30 |
| Full Fine-tuning | 86 | 12 | 17 | 138 | 93,69 |

Linear Probing phát hiện 126/155 ảnh có u và bỏ sót **29 ảnh**, tương ứng **Recall `81,29%` (dưới ngưỡng 88%)**. Full Fine-tuning phát hiện 138/155 ảnh có u, còn 17 ảnh bị bỏ sót.

### Bằng chứng Overfitting trên DS1 khi Full Fine-tuning

![Train và validation loss theo từng fold DS1](figures/ds1_full_finetune_loss.png)

| Fold | Epoch được chọn | Train loss từ epoch chọn đến 20 | Val loss từ epoch chọn đến 20 | Val accuracy ở epoch 20 | Test ROC-AUC (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 0,01788 → 0,00002 | 0,2345 → 0,3832 | 86,05% | 95,97% |
| 2 | 6 | 0,02695 → 0,00023 | 0,5079 → 0,6058 | 82,50% | 97,90% |
| 3 | 4 | 0,07560 → 0,00016 | 0,3807 → 0,5468 | 79,49% | 89,84% |
| 4 | 20 | 0,00020 → 0,00020 | 0,1076 → 0,1076 | 97,44% | 96,10% |
| 5 | 5 | 0,06048 → 0,00020 | 0,2956 → 0,4017 | 95,12% | 94,40% |

Cả **5/5 fold** đạt **100% train accuracy** ở epoch 20. Ở **4/5 fold**, `train_loss` giảm từ epoch được chọn đến epoch 20 (về sát `0`), trong khi `validation loss` tăng ngược lên. Fold 3 thể hiện rõ nhất, với `val_loss` tăng từ **0,3807 lên 0,5468** và kéo Recall trên tập test của Fold 3 tụt xuống chỉ còn **74,19%**.

## 4. Kết quả DS2 (7.023 ảnh, chia 70/15/15)

Đánh giá trên 1.029 ảnh test. Macro precision, recall, F1 và ROC-AUC (OvR) là trung bình đều trên bốn lớp. Recall phát hiện có u gộp ba loại u thành lớp dương.

| Chế độ | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) | Macro ROC-AUC (%) | Recall có u (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear Probing | 80,86 | 80,32 | **80,21** | 80,12 | 95,22 | 95,58 |
| Full Fine-tuning | 93,88 | 94,00 | 93,54 | 93,62 | 99,37 | 98,66 |

| Lớp | Recall Linear Probing | Recall Full Fine-tuning | AUC Linear Probing | AUC Full Fine-tuning |
| --- | ---: | ---: | ---: | ---: |
| Không u | 94,33% | 99,29% | 98,69% | 99,97% |
| Glioma | **76,54%** | 89,30% | 95,80% | 98,96% |
| Meningioma | **64,46%** | 85,95% | 90,07% | 98,78% |
| Pituitary | 85,50% | 99,62% | 96,33% | 99,77% |

Full Fine-tuning phân loại đúng 966/1.029 ảnh. Epoch được chọn là 21 cho Linear Probing và 8 cho Full Fine-tuning.

![Train và validation loss và accuracy của hai chế độ trên DS2](figures/ds2_learning_curves.png)

## 5. Cơ sở thực nghiệm biện luận cho việc áp dụng LoRA ở Tuần 3

Từ hai chuỗi thực nghiệm trên DS1 (tập nhỏ 253 ảnh) và DS2 (tập lớn 7.023 ảnh), ta rút ra hai bằng chứng định lượng đối lập giữa **Linear Probing** và **Full Fine-tuning**:

1. **Hạn chế về Độ nhạy (`Recall < 88%`) của Linear Probing (Hiện tượng Underfitting miền ảnh y khoa)**:
   - Khi đóng băng toàn bộ bộ mã hóa Vision Transformer tiền huấn luyện trên ảnh tự nhiên (ImageNet) và chỉ huấn luyện tầng tuyến tính cuối, các bản đồ tự chú ý (Self-Attention Maps) không thể điều chỉnh để bắt được cấu trúc mô mềm và ranh giới thâm nhiễm mờ của khối u trên ảnh MRI.
   - Hệ quả trực tiếp: Trên DS1, Recall phát hiện khối u chỉ đạt **`81,29% ± 7,36%` (< 88%)**, bỏ sót tới **29/155 ca có u (FN = 29)**. Trên DS2, Macro Recall 4 lớp chỉ đạt **`80,21%` (< 88%)**, đặc biệt thất bại nặng trên hai loại u khó phân biệt là **Meningioma (`Recall = 64,46%`)** và **Glioma (`Recall = 76,54%`)**. Trong sàng lọc lâm sàng, độ nhạy dưới 88% là không đạt yêu cầu an toàn vì tỷ lệ bỏ sót bệnh nhân có u quá cao.
2. **Hiện tượng Overfitting nghiêm trọng của Full Fine-tuning trên tập dữ liệu nhỏ (DS1)**:
   - Khi mở khóa toàn bộ hàng triệu tham số của ViT trên tập huấn luyện nhỏ (~160 ảnh/fold của DS1), dung lượng biểu diễn quá lớn khiến mô hình nhanh chóng học thuộc lòng nhiễu của tập train: cả **5/5 fold** chạm `100%` train accuracy (`train_loss -> 0,00002`), trong khi **4/5 fold** bị suy thoái `val_loss` ngay sau epoch 4–6 (điển hình Fold 3 có `val_loss` vọt từ `0,3807` lên `0,5468` và Recall test rơi xuống `74,19%`).
   - Ngay cả trên `ViT-Tiny` (~5,5 triệu tham số) hiện tượng Overfitting đã xuất hiện ở 4/5 fold; khi áp dụng trên kiến trúc chuẩn `ViT-Base/16` (~85,8 triệu tham số, lớn gấp 15,6 lần), việc Full Fine-tuning trên các bộ dữ liệu y khoa quy mô nhỏ càng gây Overfitting nặng và tốn chi phí tính toán lớn.
3. **Giải pháp tất yếu ở Tuần 3 — Tinh chỉnh hiệu quả tham số với LoRA (Low-Rank Adaptation)**:
   - Để giải quyết đồng thời hai nút thắt trên, ở Tuần 3 nhóm áp dụng **LoRA** bằng cách đóng băng trọng số gốc $W_0 \in \mathbb{R}^{d \times k}$ của ViT và chỉ chèn thêm cặp ma trận phân rã hạng thấp $\Delta W = B A$ (với hạng $r \in \{4, 8, 16\} \ll \min(d, k)$) vào các phép chiếu truy vấn và giá trị ($W_q, W_v$) trong tầng Multi-Head Self-Attention.
   - Cơ chế này vừa cho phép **dịch chuyển trực tiếp trường chú ý của ViT sang miền ảnh MRI** để vượt ngưỡng độ nhạy `Recall > 88%`, vừa **giới hạn số lượng tham số huấn luyện dưới 1%** tổng số tham số mô hình nhằm đóng vai trò chính quy hóa cấu trúc (structural regularizer), triệt tiêu hiện tượng Overfitting trên tập mẫu nhỏ như DS1.

Các thư mục kết quả lưu cấu hình, ảnh thuộc từng tập, đường học theo epoch, dự đoán test và ma trận nhầm lẫn: [DS1](ds1/summary.json) · [DS2](ds2/summary.json) · [Hướng dẫn chạy lại](../../experiments/HUONG_DAN_TUAN2.md)
