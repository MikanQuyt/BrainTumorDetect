# BÁO CÁO TUẦN 3 — Lâm Gia Bảo (Dataset 4: BraTS2020)

> **Notebook huấn luyện & đối sánh thực nghiệm:** [`notebooks/Week3_ViT_LoRA_BraTS2020_Colab.ipynb`](../../../notebooks/Week3_ViT_LoRA_BraTS2020_Colab.ipynb)

## Nhiệm vụ Tuần 3
1. Huấn luyện **ViT-LoRA + Box Regression Head (`ViTDetBoxHead`)** trên tập lát cắt đa phương thức **Dataset 4 (BraTS2020)**.
2. Thực hiện thí nghiệm đối sánh hiệu quả phát hiện và đóng khung khối u giữa **ảnh tổ hợp 3 kênh `[FLAIR, T1ce, T2]`** so với việc **chỉ dùng 1 kênh `T1ce` đơn lẻ** trên cùng tập chia bệnh nhân (`Train / Val / Test`) và cùng siêu tham số.

---

## 1. Kiến trúc Mô hình & Dữ liệu Chuẩn hóa

### 1.1 Kiến trúc `ViT-LoRA (r=8) + ViTDetBoxHead`
| Thành phần | Chi tiết |
|---|---|
| **Backbone** | `google/vit-base-patch16-224-in21k` (ViT-Base/16, 12 tầng Attention, `85,80M` tham số đóng băng) |
| **PEFT LoRA** | `r = 8`, `α = 16`, `dropout = 0.1` gắn vào `query` và `value` của cả 12 tầng Self-Attention (`294.912` tham số = `0,34%` backbone) |
| **Classification Head** | Bài toán nhị phân 2 lớp (`0: no_tumor`, `1: tumor / glioma`) |
| **Box Regression Head** | `ViTDetBoxHead` (Simple Feature Pyramid `14×14 -> 28×28`, Spatial Attention Pooling + MLP $\rightarrow$ `[x_min, y_min, x_max, y_max]` chuẩn hóa) |
| **Tổng tham số khả huấn** | **`< 1%`** tổng tham số mô hình (`LoRA + Classification Head + ViTDetBoxHead`) |

### 1.2 Thống kê Dataset 4 (BraTS2020 — Chia theo Bệnh nhân, 0% Rò rỉ `case_id`)
| Tập | Số lát cắt | Ghi chú |
|---|---:|---|
| **Train** | `29.772` | Dùng để huấn luyện & cập nhật trọng số |
| **Validation** | `6.087` | Chọn checkpoint tốt nhất (`val_f1 + 0.2 * val_box_iou`) & Early Stopping |
| **Test** | `6.175` | Đánh giá cuối cùng cho cả 2 cấu hình `3-ch [FLAIR, T1ce, T2]` và `1-ch T1ce` |
| **Tổng cộng** | **`42.034`** | **`22.814`** lát có u (`has_tumor = 1`, có Mask + `gt_box`) & **`19.220`** lát não lành (`has_tumor = 0`) |

---

## 2. Cấu hình Huấn luyện Thống nhất (Công bằng giữa 3 kênh và 1 kênh `T1ce`)

| Siêu tham số | Giá trị |
|---|---|
| **Task / Số lớp** | `binary` (`num_classes = 2`: `no_tumor` vs `tumor`) |
| **Đầu vào (Thí nghiệm 1)** | Ảnh tổ hợp 3 kênh `[FLAIR, T1ce, T2]` (`224×224`, chuẩn hóa ImageNet) |
| **Đầu vào (Thí nghiệm 2)** | Chỉ lấy kênh `T1ce` gốc (kênh index `1`), nhân bản thành `[T1ce, T1ce, T1ce]` trước khi chuẩn hóa ImageNet |
| **Số Epochs / Early Stopping** | Tối đa `20` epochs, `patience = 6` (tự động resume theo từng epoch từ Google Drive) |
| **Batch Size / AMP** | `batch_size = 64`, `FP16 Mixed Precision (AMP)` |
| **Optimizer / Scheduler** | `AdamW` (`lr_lora = 3e-4`, `lr_head = 1e-3`, `lr_box = 1e-3`, `weight_decay = 0.01`) + `CosineAnnealingLR` + `GradClip = 1.0` |
| **Hàm mất mát** | `Focal Loss (γ = 2.0, Label Smoothing ε = 0.05)` + `GIoU Loss + L1 Loss (1:1)` |

---

## 3. Bảng Đối sánh Thực nghiệm trên Tập Test (`6.175` lát cắt BraTS2020)

> *Chạy toàn bộ các cell từ đầu đến cuối trong [`notebooks/Week3_ViT_LoRA_BraTS2020_Colab.ipynb`](../../../notebooks/Week3_ViT_LoRA_BraTS2020_Colab.ipynb) trên Google Colab (T4 GPU) để tự động điền bảng số liệu thật bên dưới và xuất đầy đủ `comparison_3ch_vs_t1ce.csv`, `comparison_3ch_vs_t1ce.png`, `convergence_3ch_vs_t1ce.png` và `qualitative_3ch_vs_t1ce.png`.*

| Chỉ số trên Tập Test (`6.175` lát cắt) | 3 kênh `[FLAIR, T1ce, T2]` | 1 kênh `T1ce` đơn lẻ | Chênh lệch ($\Delta = \text{3-ch} - \text{1-ch}$) |
|---|---:|---:|---:|
| **Test Loss** | *(tự động điền sau khi chạy Colab)* | *(tự động điền sau khi chạy Colab)* | — |
| **Test Accuracy (%)** | *(tự động điền)* | *(tự động điền)* | — |
| **Sensitivity / Recall (%)** | *(tự động điền)* | *(tự động điền)* | — |
| **Specificity (%)** | *(tự động điền)* | *(tự động điền)* | — |
| **F1-Score (%)** | *(tự động điền)* | *(tự động điền)* | — |
| **ROC-AUC** | *(tự động điền)* | *(tự động điền)* | — |
| **Mean Box IoU (%)** | *(tự động điền)* | *(tự động điền)* | — |
| **Box IoU@0.50 (%)** | *(tự động điền)* | *(tự động điền)* | — |
| **Box IoU@0.75 (%)** | *(tự động điền)* | *(tự động điền)* | — |

---

## 4. Cơ sở Y sinh học của Thí nghiệm Đối sánh Đa xung vs Đơn xung
- **FLAIR**: triệt tiêu tín hiệu dịch não tủy, làm nổi bật vùng phù nề quanh u (*peritumoral edema*) — yếu tố quyết định để bao trọn ranh giới **Whole Tumor (WT)** khi đóng khung Bounding Box.
- **T1ce**: hiển thị rõ lõi khối u tăng quang (*enhancing tumor core*) nhưng không nhìn rõ vùng phù nề ngoại vi, khiến hộp dự đoán từ `T1ce` đơn lẻ có xu hướng thu hẹp quá mức (giảm `Mean Box IoU` và `IoU@0.50` trên nhãn WT).
- **T2**: bổ sung cấu trúc dịch và ranh giới mô mềm giúp giảm dương tính giả trên các lát cắt não lành.
