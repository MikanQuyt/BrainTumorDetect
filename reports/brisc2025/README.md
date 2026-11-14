# THỰC NGHIỆM TUẦN 2: BRISC 2025 & FOUNDATION MODELS

Báo cáo và kết quả thực nghiệm của Tuần 2 trên bộ dữ liệu **BRISC 2025 (Dataset 3)** và **Dataset 1 (Navoneel)**.

---

## 1. Cấu Trúc Tổ Chức

```
experiments/brisc2025/                # Mã nguồn thực nghiệm DS3 & DS1
├── config.py                         # Cấu hình đường dẫn tương đối theo repo, siêu tham số
├── utils.py                          # Hàm tiện ích tính Dice, IoU, HD95, Accuracy, F1
├── exp1_vit_classification.py        # Thực nghiệm 1: ViT Linear Probing & Full Fine-Tuning (tách biệt train/val)
├── exp2_medsam_oracle_box.py         # Thực nghiệm 2: Zero-shot MedSAM (medsam_vit_b.pth) với Oracle Box
├── exp3_sam_everything.py            # Thực nghiệm 3: SAM Everything Mode (sam_vit_b_01ec64.pth) trên 25 BRISC + 25 DS1
└── run_all.py                        # Master script chạy lại toàn bộ thực nghiệm

reports/brisc2025/                    # Kết quả đầu ra & báo cáo
├── BRISC2025_Experiments.md          # Báo cáo Markdown chi tiết
├── CONSOLIDATED_EXPERIMENT_REPORT.md # Tóm tắt 3 thực nghiệm
├── consolidated_report.json          # Báo cáo JSON gộp cả 3 thực nghiệm
├── classification/                   # Metrics JSON, CSV từng lớp, đồ thị loss/acc, ma trận nhầm lẫn
├── segmentation/                     # Metrics MedSAM JSON, CSV 50 lát cắt, lưới ảnh định tính 4 cột
└── sam_everything/                   # Stats JSON, CSV 50 ảnh, biểu đồ thống kê 4 ô, 50 ảnh PNG overlay
```

---

## 2. Tóm Tắt Kết Quả 3 Thực Nghiệm

| Thực nghiệm | Mô hình & Lời nhắc | Chỉ số chính | Kết luận khoa học |
| :--- | :--- | :---: | :--- |
| **1. Phân loại ViT** | **Linear Probing**<br>**Full Fine-Tuning** | **82.75% Acc** (F1: 82.63%)<br>**72.00% Acc** (F1: 68.39%) | Linear Probing ổn định hơn trên tập mẫu giới hạn nhưng Recall Meningioma mới đạt 66% (< 88%), làm cơ sở áp dụng LoRA ở Tuần 3. |
| **2. Phân vùng MedSAM** | **`medsam_vit_b.pth` + Oracle Box** | **89.64% Dice**<br>**81.99% IoU \| 8.29px HD95** | Đạt chuẩn lâm sàng trên cả 3 loại u: Meningioma 92.15%, Pituitary 90.53%, Glioma 85.64% (+8.53% so với MobileSAM). |
| **3. SAM Everything** | **`sam_vit_b_01ec64.pth` không Prompt** | **89.77% mask ngoài u**<br>Best-Dice: **43.61%** | Bắt nhầm xương vòm sọ và nhu mô lành (phủ trung bình 50.29% lát cắt) trên cả BRISC 2025 và Dataset 1, khẳng định bắt buộc phải có nhánh định vị ViT đi trước. |

---

## 3. Cách Tái Lập Kết Quả (Reproduction)

Từ thư mục gốc của repo:

```powershell
.\venv\Scripts\python.exe experiments/brisc2025/exp1_vit_classification.py
.\venv\Scripts\python.exe experiments/brisc2025/exp2_medsam_oracle_box.py
.\venv\Scripts\python.exe experiments/brisc2025/exp3_sam_everything.py
```
