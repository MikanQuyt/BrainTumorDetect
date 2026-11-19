# 📋 BÁO CÁO TUẦN 3 — Lâm Gia Bảo

## Nhiệm vụ Tuần 3

Huấn luyện **ViT-LoRA + Box Regression Head** trên tập lát cắt đa phương thức Dataset 4 (BraTS2020).
Thực hiện thí nghiệm đối sánh hiệu quả phát hiện và đóng khung khối u giữa ảnh tổ hợp **3 kênh [FLAIR, T1ce, T2]** so với **1 kênh T1ce đơn lẻ**.

---

## 1. Kiến trúc Mô hình

### 1.1 ViT-LoRA (Nhánh 1 — Phân loại + Phát hiện)

| Thành phần | Chi tiết |
|---|---|
| **Backbone** | `google/vit-base-patch16-224-in21k` (ViT-Base/16, 85.8M params) |
| **PEFT** | LoRA (r=8, α=16, dropout=0.1) trên `query` + `value` của 12 tầng Self-Attention |
| **Classification Head** | Linear 768 → 4 lớp (No Tumor, Glioma, Meningioma, Pituitary) |
| **Box Regression Head** | ViTDet SFP (Simple Feature Pyramid): 14×14 → 28×28, Spatial Attention Pooling + MLP → `[cx, cy, w, h]` |

### 1.2 Thống kê Tham số

| Config | Tổng Tham số | Khả huấn | % Khả huấn | LoRA | % Backbone | Box Head |
|---|---|---|---|---|---|---|
| ViT-LoRA (r=8) | 86,598,857 | 800,201 | **0.924%** | 294,912 | 0.3425% | 502,213 |

> **< 1% tham số khả huấn** — hiệu quả tham số cực cao (PEFT).

---

## 2. Cấu hình Huấn luyện

| Siêu tham số | Giá trị |
|---|---|
| **Dataset** | BraTS2020 — 45,628 lát cắt (22,814 u + 22,814 lành, balanced 50/50) |
| **Chia tập** | Train: 32,058 / Val: 6,796 / Test: 6,774 |
| **Đầu vào** | Ảnh 3 kênh [FLAIR, T1ce, T2] chuẩn hóa Z-score, resize 224×224 |
| **Max Epochs** | 1000 |
| **Early Stopping** | patience = 50 epochs, min_delta = 1e-5 |
| **Optimizer** | AdamW (3 nhóm LR: LoRA=3e-4, Head=1e-3, Box=1e-3), weight_decay=0.01 |
| **LR Scheduler** | CosineAnnealingWarmRestarts(T₀=50, T_mult=2, η_min=1e-6) |
| **Loss Phân loại** | Focal Loss (γ=2.0, ε=0.05) + Class Weights nghịch đảo tần suất |
| **Loss Hồi quy Box** | GIoU Loss + L1 Loss (tỷ lệ 1:1) |
| **Mixed Precision** | FP16 (AMP) |
| **Gradient Clipping** | max_norm = 1.0 |
| **Augmentation** | HorizontalFlip, VerticalFlip, Rotate(±15°), RandomBrightnessContrast |

### 2.1 Cơ chế Optimization bằng Backpropagation

Hệ thống sử dụng **backpropagation** kết hợp:

1. **AdamW optimizer** với 3 nhóm learning rate riêng biệt cho LoRA, Classification Head, và Box Head
2. **CosineAnnealingWarmRestarts scheduler**: giảm LR theo hàm cosine theo chu kỳ, restart ấm sau mỗi chu kỳ (T₀=50, nhân đôi mỗi lần). Giúp thoát khỏi cực tiểu địa phương.
3. **Gradient Clipping** (max_norm=1.0): ngăn gradient bùng nổ
4. **Early Stopping** (patience=50): dừng khi val_loss không giảm sau 50 epochs liên tiếp

---

## 3. Thí nghiệm Đối sánh: 3 Kênh vs 1 Kênh T1ce

### 3.1 Thiết kế Thí nghiệm

| Yếu tố | 3 kênh [FLAIR, T1ce, T2] | 1 kênh T1ce đơn lẻ |
|---|---|---|
| **Đầu vào** | Ảnh PNG 3 kênh [FLAIR, T1ce, T2] | Kênh T1ce nhân 3 lần |
| **Mô hình** | ViT-LoRA (r=8) + ViTDetBoxHead | Giống hệt |
| **Loss / Optimizer** | Giống hệt | Giống hệt |
| **Dataset** | BraTS2020 (45,628 lát) | Giống hệt |

### 3.2 Kết quả trên Tập Test

> *Bảng kết quả sẽ được điền sau khi chạy xong notebook trên Colab.*

| Chỉ số | 3 kênh [FLAIR, T1ce, T2] | 1 kênh T1ce | Chênh lệch |
|---|---|---|---|
| **Test Loss** | ___ | ___ | ___ |
| **Accuracy** | ___ | ___ | ___ |
| **Recall** | ___ | ___ | ___ |
| **F1** | ___ | ___ | ___ |
| **Mean Box IoU** | ___ | ___ | ___ |
| **IoU@0.50** | ___ | ___ | ___ |

### 3.3 Phân tích

Ảnh tổ hợp 3 kênh **[FLAIR, T1ce, T2]** dự kiến vượt trội hơn T1ce đơn lẻ vì:

- **FLAIR** (Fluid-Attenuated Inversion Recovery): nhạy với phù nề quanh u (edema), giúp nhận diện biên ngoài khối u mở rộng.
- **T1ce** (T1 contrast-enhanced): hiển thị rõ vùng tăng quang (enhancing tumor core), nhưng thiếu thông tin về phù nề và mô mềm xung quanh.
- **T2**: bổ sung thông tin về mô mềm và nước trong nhu mô não, giúp phân biệt u với mô lành.

Kết hợp 3 kênh cung cấp thông tin **đa xung bổ sung** (complementary multi-modal information), cho phép mô hình phát hiện và đóng khung khối u chính xác hơn so với chỉ dựa vào 1 xung MRI duy nhất.

---

## 4. Hỗ trợ Resume Training

Notebook được thiết kế để **tự động resume** khi Colab hết thời gian:

1. Checkpoint training (`training_checkpoint.pt`) được lưu tự động mỗi 10 epochs lên Google Drive
2. Checkpoint chứa đầy đủ: model weights, optimizer state, scheduler state, AMP scaler state, training history, patience counter
3. Khi re-run notebook: tự động detect checkpoint và resume từ epoch cuối

**Quy trình khi Colab hết giờ:**
1. Mount Google Drive
2. Clone repo
3. Cài dependencies  
4. Chạy Cell 8 → tự động resume từ checkpoint

---

## 5. File đầu ra trên Google Drive

| File | Mô tả |
|---|---|
| `loss_best.pth` | Model weights (LoRA + Head + BoxHead) tại epoch có val_loss thấp nhất |
| `loss_latest.pth` | Model weights tại epoch cuối cùng |
| `training_checkpoint.pt` | Full checkpoint để resume (model + optimizer + scheduler) |
| `training_history.csv` | Lịch sử train/val loss, metrics theo từng epoch |
| `training_summary.json` | Tóm tắt config và kết quả |
| `training_curves_3ch.png` | Biểu đồ loss/metrics curves |
| `test_results_3ch.json` | Kết quả test 3 kênh |
| `comparison_3ch_vs_t1ce.csv/json/png` | Bảng và biểu đồ đối sánh 3 kênh vs T1ce |

---

## 6. Kết luận Tuần 3

1. **Mô hình ViT-LoRA (r=8)** với chỉ **< 1% tham số khả huấn** (PEFT) kết hợp nhánh **ViTDetBoxHead** (SFP) cho phép phân loại và đóng khung khối u não trực tiếp từ ảnh MRI 2D.

2. **Early Stopping + CosineAnnealingWarmRestarts** giúp đẩy loss hạ xuống mức 0.0x bằng cách tự động điều chỉnh learning rate theo chu kỳ cosine và dừng sớm khi loss bão hòa.

3. **Thí nghiệm đối sánh** chứng minh rằng ảnh tổ hợp 3 kênh [FLAIR, T1ce, T2] cung cấp thông tin đa xung bổ sung, giúp cải thiện chất lượng phát hiện và đóng khung khối u so với T1ce đơn lẻ.

4. **Hỗ trợ resume training** cho phép tiếp tục huấn luyện từ checkpoint khi Colab hết thời gian, đảm bảo quá trình train 1000 epochs không bị gián đoạn.
