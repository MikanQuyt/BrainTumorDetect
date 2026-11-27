# BÁO CÁO THỰC NGHIỆM TUẦN 4 — Nguyễn Thanh Hải (`Nguyen_Thanh_Hai_tuan4`)

**Đề tài:** Huấn luyện `MedSAM-LoRA (r = 8, α = 16)` trên BRISC 2025, Đánh giá theo Nhóm Bệnh (`Glioma`, `Meningioma`, `Pituitary`) & Ablation Chiến lược Prompt (`Box` vs `Hybrid: Box + Peak Point`).

---

## 1. Cấu hình Huấn luyện & Kiểm định Dữ liệu BRISC 2025
- **Dữ liệu phân vùng (`has_mask=True, has_tumor=1`):** `3,355` Train / `719` Validation / `719` Test (`210` Glioma, `245` Meningioma, `264` Pituitary).
- **Kiến trúc:** Đóng băng Image Encoder (`ViT-B`) và Prompt Encoder, chèn LoRA (`r = 8`, `α = 16.0`, `dropout = 0.05`) vào nhánh `Query` và `Value` của 12 khối Attention, mở khóa toàn bộ `Mask Decoder`.
- **Hàm mất mát:** $L_{\text{Seg}} = 0.5\,L_{\text{Dice}} + 0.5\,L_{\text{Focal}}$ + `Box Jittering ±10px` (hệ `1024×1024`).
- **Peak Positive Point (chiến lược Hybrid):** Trích xuất hoàn toàn tự động từ tương phản cường độ cục bộ và độ bất đối xứng bán cầu bên trong vùng lõi hộp (`extract_peak_points`), hoàn toàn không rò rỉ nhãn `gt_mask`.

---

## 2. Kết quả Ablation Chiến lược Prompt trên BRISC 2025 Test (`n = 719`)

### 2.1. Đối sánh trên Oracle Box (`0px` Jitter)
| Mô hình | Chiến lược Prompt | Box Jitter Eval | Số mẫu | Dice Mean (%) | Dice SD (%) | Dice Median (%) | IoU Mean (%) | HD95 Mean (px) | HD95 Median (px) |
|---|---|---|---|---|---|---|---|---|---|
| Zero-shot MedSAM (Baseline) | Box-only | 0px (Oracle Box) | 719 | 88.54 | 9.91 | 91.33 | 80.6 | 21.51 | 15.56 |
| MedSAM-LoRA (r=8) | Box-only | 0px (Oracle Box) | 719 | 93.94 | 4.76 | 95.35 | 88.92 | 12.5 | 8.53 |
| MedSAM-LoRA (r=8) | Hybrid (Box + Peak Point) | 0px (Oracle Box) | 719 | 93.91 | 4.79 | 95.2 | 88.86 | 12.6 | 8.54 |

- **Cải thiện `MedSAM-LoRA (Box)` so với Zero-shot Tuần 2:** `ΔDice = +5.41%`, `ΔHD95 = -9.01 px`.
- **So sánh `Hybrid (Box + Peak Point)` vs `Box-only` (Oracle Box):** `ΔDice = -0.03%` (Wilcoxon $p = 3.9781e-05$), `ΔHD95 = +0.10 px` ($p = 4.6182e-03$).

### 2.2. Kiểm định Độ bền vững khi Bounding Box bị Nhiễu (`Perturbed Box ±15px`)
| Mô hình | Chiến lược Prompt | Box Jitter Eval | Số mẫu | Dice Mean (%) | Dice SD (%) | Dice Median (%) | IoU Mean (%) | HD95 Mean (px) | HD95 Median (px) |
|---|---|---|---|---|---|---|---|---|---|
| MedSAM-LoRA (r=8) | Box-only | ±15px | 719 | 93.03 | 5.35 | 94.38 | 87.39 | 14.19 | 10.0 |
| MedSAM-LoRA (r=8) | Hybrid (Box + Peak Point) | ±15px | 719 | 93.01 | 5.44 | 94.41 | 87.35 | 14.15 | 10.0 |

- **Chênh lệch khi Box nhiễu `±15px`:** `ΔDice = -0.03%` (Wilcoxon $p = 4.1647e-01$), `ΔHD95 = -0.04 px`.

---

## 3. Đánh giá Chi tiết theo Từng Nhóm Bệnh (`Glioma`, `Meningioma`, `Pituitary`)
| Nhóm bệnh | Số mẫu | Zero-shot Dice (%) | LoRA Box Dice (%) | LoRA Hybrid Dice (%) | ΔDice (Hybrid - Box) | LoRA Box IoU (%) | LoRA Hybrid IoU (%) | Zero-shot HD95 (px) | LoRA Box HD95 (px) | LoRA Hybrid HD95 (px) | ΔHD95 (Hybrid - Box) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Glioma | 210 | 84.16 | 91.33 | 91.31 | -0.02 | 84.63 | 84.61 | 28.07 | 18.92 | 19.51 | 0.59 |
| Meningioma | 245 | 92.65 | 95.93 | 95.95 | 0.02 | 92.3 | 92.33 | 18.55 | 10.56 | 9.72 | -0.84 |
| Pituitary | 264 | 88.2 | 94.18 | 94.08 | -0.1 | 89.19 | 89.01 | 19.04 | 9.21 | 9.8 | 0.59 |

---

## 4. Biểu đồ & Minh họa Trực quan
- `image/tuan4/hai/fig1_medsam_lora_training_curves.png`
- `image/tuan4/hai/fig2_per_class_tumor_comparison.png`
- `image/tuan4/hai/fig3_prompt_ablation_box_vs_hybrid.png`
- `image/tuan4/hai/fig4_qualitative_box_vs_hybrid_grid.png`
