# Báo cáo Huấn luyện & Đánh giá Tuần 3 — ViT-LoRA + ViTDetBoxHead + MedSAM (Multiclass)

## 1. Cấu hình Huấn luyện
- **Tác vụ & Dữ liệu:** Phân loại đa lớp khối u não (`multiclass`, 4 lớp) trên bộ dữ liệu BRISC 2025 (`train=4200`, `val=900`, `test=900`).
- **Kiến trúc:** `google/vit-base-patch16-224-in21k` + LoRA (`r=8`, `alpha=16`, 800,201 / 86,598,857 tham số khả huấn = `0.92%`) + `ViTDetBoxHead`.
- **Siêu tham số:** `loss=focal`, `box_loss_weight=1.0`, `batch_size=32`, `lr_lora=3e-4`, `scheduler=plateau`, `early_stop_metric=loss` (`patience=30`).
- **Hội tụ:** Dừng sớm (Early Stopping) tại **Epoch 226**, trọng số tốt nhất lưu tại **Epoch 196** (`test_loss=0.3557`: `cls_loss=0.0271`, `box_loss=0.3285`).

## 2. Kết quả Phân loại & Sàng lọc trên tập Test (900 ảnh)
| Nhiệm vụ | Chỉ số | Kết quả |
| :--- | :--- | :--- |
| **Phân loại đa lớp (Multiclass)** | Test Accuracy | **98.11%** (`0.9811`) |
| **Phân loại đa lớp (Multiclass)** | Macro F1-Score | **98.10%** (`0.9810`) |
| **Phân loại đa lớp (Multiclass)** | Macro Recall | **98.08%** (`0.9808`) |
| **Phân loại đa lớp (Multiclass)** | ROC-AUC | **0.9994** |
| **Sàng lọc nhị phân ($\tau = 0.35$)** | Recall (Độ nhạy) | **99.58%** |
| **Sàng lọc nhị phân ($\tau = 0.35$)** | Specificity (Độ đặc hiệu) | **98.34%** |
| **Sàng lọc nhị phân ($\tau = 0.35$)** | F1-Score | **99.58%** |

## 3. Kết quả Định vị Auto-Prompt & Phân đoạn Zero-shot MedSAM (100 mẫu Test có u)
| Nhóm chỉ số | Phương pháp sinh Prompt | Mean ± SD | Median | Ghi chú |
| :--- | :--- | :--- | :--- | :--- |
| **Box IoU** | Weakly-supervised Heatmap | 30.73% ± 27.03% | 23.79% | `IoU@50 = 29.0%`, `Pointing Acc = 84.0%` |
| **Box IoU** | **ViTDetBoxHead (Supervised)** | **68.01% ± 21.24%** | **75.19%** | **`IoU@50 = 84.0%`**, **`IoU@75 = 51.0%`** |
| **MedSAM Dice** | Heatmap Prompt | 46.22% ± 37.66% | 49.94% | `n_pred_empty = 0` |
| **MedSAM Dice** | **ViTDetBoxHead Prompt** | **82.13% ± 21.54%** | **89.75%** | **Tăng +35.91% Dice** |
| **MedSAM IoU** | Heatmap Prompt | 38.32% ± 34.51% | 33.29% | — |
| **MedSAM IoU** | **ViTDetBoxHead Prompt** | **73.79% ± 23.11%** | **81.41%** | **Tăng +35.47% IoU** |
| **MedSAM HD95 (px)** | Heatmap Prompt | 166.52 ± 145.30 | 122.15 | — |
| **MedSAM HD95 (px)** | **ViTDetBoxHead Prompt** | **38.53 ± 49.24** | **21.59** | **Giảm 4.3 lần sai số biên** |

## 4. Vị trí lưu trữ Checkpoint (Google Drive)
- Thư mục: `ProjectAI/Hai/multiclass/`
- `loss_best.pth` (Epoch 196) và `loss_latest.pth` (Epoch 226).
