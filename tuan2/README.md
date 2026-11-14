# THỰC NGHIỆM TUẦN 2: BRISC 2025 & FOUNDATION MODELS

Thư mục lưu trữ toàn bộ mã nguồn, dữ liệu thực nghiệm, trọng số mô hình và báo cáo khoa học của Tuần 2 trên bộ dữ liệu **BRISC 2025 (Dataset 3)** và **Dataset 1**.

---

## 1. Cấu Trúc Thư Mục

```
├── BRISC2025_Experiments.docx        # BÁO CÁO WORD TOÀN DIỆN (Bảng biểu, hình ảnh, phân tích)
├── BRISC2025_Experiments.md          # Báo cáo Markdown chi tiết
├── run_all.py                        # Master script chạy lại toàn bộ thực nghiệm
│
├── src/                              # Mã nguồn Python
│   ├── config.py                     # Cấu hình đường dẫn, siêu tham số
│   ├── utils.py                      # Hàm tiện ích tính Dice, IoU, Accuracy, F1
│   ├── exp1_vit_classification.py    # Thực nghiệm 1: ViT Linear Probing & Full Fine-Tuning
│   ├── exp2_medsam_oracle_box.py     # Thực nghiệm 2: Zero-shot MedSAM với Oracle Box
│   ├── exp3_sam_everything.py        # Thực nghiệm 3: SAM Everything Mode (50 ảnh đối chứng)
│   ├── create_docx_report.py         # Script tự động xuất file Word
│   └── mobile_sam/                   # Engine TinyViT tương thích MedSAM/SAM
│
├── results/                          # Kết quả đầu ra
│   ├── classification/               # Metrics JSON, CSV từng lớp, đồ thị loss/acc, ma trận nhầm lẫn
│   ├── segmentation/                 # Metrics MedSAM JSON, CSV 50 lát cắt, lưới ảnh định tính 4 cột
│   ├── sam_everything/               # Stats JSON, CSV 50 ảnh, biểu đồ thống kê 4 ô, 50 ảnh PNG overlay
│   ├── consolidated_report.json      # Báo cáo JSON gộp cả 3 thực nghiệm
│   ├── CONSOLIDATED_EXPERIMENT_REPORT.md
│   └── BRISC2025_Experiments.docx
│
└── checkpoints/                      # Trọng số mô hình
    ├── mobile_sam.pt                 # Trọng số MobileSAM/MedSAM (40.7 MB)
    ├── vit_linear_probe_head.pth     # Trọng số Head phân loại Linear Probing
    └── vit_full_finetune_best.pth    # Trọng số Full Fine-Tuning
```

---

## 2. Tóm Tắt Kết Quả 3 Thực Nghiệm

| Thực nghiệm | Mô hình & Lời nhắc | Chỉ số chính | Kết luận khoa học |
| :--- | :--- | :---: | :--- |
| **1. Phân loại ViT** | **Linear Probing**<br>**Full Fine-Tuning** | **82.75% Acc** (F1: 82.63%)<br>**72.00% Acc** (F1: 68.39%) | Linear Probing giữ đặc trưng ImageNet trừu tượng tốt hơn, không bị overfitting trên tập MRI. |
| **2. Phân vùng MedSAM** | **Zero-shot + Oracle Box** | **87.87% Dice**<br>**81.57% IoU** | Đạt chuẩn lâm sàng xuất sắc khi có Bounding Box định hướng; U tuyến yên 93.54%, U màng não 91.49%. |
| **3. SAM Everything** | **Tự động không có Prompt** | **76.75% mask bắt nhầm**<br>Phủ cực đại: **87.7%** | SAM bắt nhầm xương vòm sọ và nhu mô lành trên cả não lành (No Tumor), chứng minh Medical Prompting là bắt buộc. |

---

## 3. Cách Tái Lập Kết Quả (Reproduction)

Mở PowerShell tại thư mục này và chạy:

```powershell
# Chạy toàn bộ 3 thực nghiệm và xuất báo cáo:
$env:PYTHONIOENCODING="utf-8"
py -3.12 run_all.py --exp all

# Hoặc chạy từng bài riêng biệt:
py -3.12 src/exp1_vit_classification.py
py -3.12 src/exp2_medsam_oracle_box.py
py -3.12 src/exp3_sam_everything.py
```
