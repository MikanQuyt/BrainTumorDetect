# BrainTumorDetect — Hệ thống Phát hiện & Phân vùng Khối u Não trên Ảnh MRI kết hợp ViT-LoRA và MedSAM

Dự án xây dựng kiến trúc hai nhánh nối tiếp (**Two-Stage Cascaded Pipeline**) cho bài toán chẩn đoán và phân vùng khối u não đa bộ dữ liệu MRI:
1. **Nhánh 1 (`ViT-LoRA` + `Auto-Prompt Generator`):** Tinh chỉnh hiệu quả tham số (**PEFT LoRA**, `< 1%` tham số khả huấn) trên 12 tầng Self-Attention của `google/vit-base-patch16-224-in21k` để phân loại u não đạt độ nhạy cao (`Recall > 88%`) và tự động sinh lời nhắc lai (**Hybrid Prompt**: Hộp giới hạn + Điểm cực đại Dương/Âm) qua 2 cơ chế:
   - **Cơ chế (1) — Giám sát yếu (`Attention Rollout + LayerCAM` + Lọc viền hộp sọ):** Dành cho Dataset 1 (`Navoneel`) không có nhãn Bounding Box/Mask.
   - **Cơ chế (2) — Giám sát có nhãn hộp (`ViTDetBoxHead` SFP + `GIoU + L1 Loss`):** Dành cho Dataset 2 (`indk214`), Dataset 3 (`BRISC 2025`) và Dataset 4 (`BraTS2020`).
2. **Nhánh 2 (`MedSAM` / `MedSAM-LoRA`):** Nhận lời nhắc tự sinh từ Nhánh 1 để phân vùng chính xác ranh giới khối u trên hệ tọa độ `1024×1024`.

---

## 1. Cấu trúc Thư mục Dự án (Nhánh `dev`)

```text
BrainTumorDetect/
├── configs/
│   └── default.yaml                # Cấu hình siêu tham số thống nhất (Data, ViT Baseline, ViT-LoRA, Auto-Prompt, MedSAM)
├── data/                           # [Tuần 1] Tiền xử lý hộp sọ, CLAHE, 3D-to-2D BraTS, EDA & Unified DataLoader
│   ├── skull_crop.py               # Cắt khung sọ tự động (Otsu + Connected Components) & cân bằng sáng CLAHE
│   ├── build_ds1_ds2.py            # Chuẩn hóa Dataset 1 (Navoneel) & Dataset 2 (indk214)
│   ├── build_ds3.py                # Chuẩn hóa Dataset 3 (BRISC 2025) & khử trùng lặp SHA-256
│   ├── build_dataset.py            # Chuẩn hóa Dataset 4 (BraTS2020 NIfTI 3D -> PNG đa xung FLAIR, T1ce, T2)
│   ├── eda_ds1_ds2.py              # Phân tích thống kê khám phá (EDA) cho DS1 & DS2
│   ├── eda_ds3.py                  # Phân tích thống kê khám phá (EDA) cho DS3
│   ├── eda_brats.py                # Phân tích thống kê khám phá (EDA) cho DS4
│   ├── transforms.py               # Phép biến đổi đồng bộ Ảnh - Mask - Box
│   └── unified_dataset.py          # UnifiedBrainMRIDataset cung cấp đồng thời image_224, image_1024, mask_1024, box_1024
├── dataset/                        # Metadata chuẩn hóa (.json) của 4 bộ dữ liệu
│   ├── ds1_annotations.json
│   ├── ds2_annotations.json
│   ├── ds3_annotations.json
│   └── brats2020_annotations.json
├── models/                         # [Tuần 2 & Tuần 3] Kiến trúc Mô hình Cốt lõi
│   ├── vit_baseline.py             # [Tuần 2] ViT Baseline (Linear Probing & Full Fine-tuning)
│   ├── sam_wrappers.py             # [Tuần 2 & 3] SamBoxSegmenter & MedSamBoxSegmenter (hỗ trợ Box & Hybrid Prompt)
│   ├── vit_lora.py                 # [Tuần 3] Kiến trúc ViTLoRA (PEFT LoRA 12 tầng Q, V) & nhánh ViTDetBoxHead (SFP)
│   └── auto_prompt.py              # [Tuần 3] AutoPromptGenerator (Cơ chế 1: Rollout + LayerCAM; Cơ chế 2: ViTDetBoxHead)
├── utils/                          # Hàm mất mát, Chỉ số đánh giá & Kiểm tra trọng số
│   ├── losses.py                   # Weighted CE & Focal Loss (Label Smoothing ε=0.05), GIoU Loss & BoxRegressionLoss
│   ├── metrics.py                  # Dice, IoU, HD95, Box IoU, Pointing Game Hit Rate & Tối ưu ngưỡng τ ∈ [0.30, 0.40]
│   └── checkpoints.py              # Xác thực MD5 checkpoint SAM & MedSAM
├── engine/                         # Pipeline Huấn luyện & Đánh giá Hợp nhất
│   ├── train_vit.py                # Huấn luyện ViT Baseline (Tuần 2)
│   ├── eval_sam.py                 # Đánh giá Zero-shot SAM / MedSAM với Oracle Box (Tuần 2)
│   └── train_vit_lora.py           # Huấn luyện & đánh giá ViT-LoRA + ViTDetBoxHead + AutoPromptGenerator (Tuần 3)
├── eval/                           # Kịch bản Đánh giá & Kiểm định Thống kê Zero-shot SAM vs MedSAM (Tuần 2)
│   ├── zeroshot_oracle_eval.py
│   ├── make_ds2_val_split.py
│   └── analyze_week2.py
├── experiments/                    # Kịch bản Thực nghiệm Chi tiết theo Tuần
│   ├── train_vit_baseline.py       # Thực nghiệm ViT Baseline trên DS1 (5-Fold CV) & DS2
│   ├── vit_data.py                 # Chia tập chống rò rỉ mức pixel (pixel_digest + StratifiedGroupKFold)
│   ├── brisc2025/                  # Thực nghiệm trên BRISC 2025 & Kiểm thử SAM Everything Mode 50 ảnh
│   ├── run_lora_rank_ablation.py   # [Tuần 3] Ablation Study 1: Khảo sát hạng LoRA r ∈ {4, 8, 16, 32}
│   └── generate_week3_hnnq_visuals.py # [Tuần 3] Sinh bộ ảnh kết quả trực quan Tuần 3
├── image/                          # Toàn bộ Ảnh Kết quả Thực nghiệm chia theo từng Tuần
│   ├── tuan1/                      # 17 biểu đồ/lưới ảnh tiền xử lý & EDA Tuần 1 + 9 ảnh mẫu DS3
│   ├── tuan2/                      # 13 biểu đồ/lưới ảnh ViT Baseline, Zero-shot MedSAM & 50 ảnh SAM Everything
│   └── tuan3/                      # 3 hình tổng hợp kết quả ViT-LoRA & Auto-Prompt Generator Tuần 3
├── reports/                        # Báo cáo, Bảng số liệu (.csv, .json, .md) chia theo từng Tuần
│   ├── tuan1/                      # build_ds1_ds2, build_ds3, build_ds4, eda_ds1_ds2, eda_ds3, eda_ds4
│   ├── tuan2/                      # vit_baseline, brisc2025, week2 (Zero-shot SAM vs MedSAM)
│   └── tuan3/                      # vit_lora_auto_prompt (week3_hnnq_metrics.json)
└── tests/                          # Bộ Unit Tests tự động (52+ tests kiểm định toàn bộ pipeline Tuần 1, 2, 3)
```

---

## 2. Hình ảnh Kết quả Nổi bật theo Từng Tuần (`image/`)

### Tuần 1 (`image/tuan1/`) — Chuẩn hóa 4 Bộ dữ liệu & Phân tích Khám phá (EDA)
- [fig1_multi_dataset_preprocessing_flowchart.png](file:///c:/Users/MikanQuyt/BrainTumorDetect/image/tuan1/fig1_multi_dataset_preprocessing_flowchart.png): Sơ đồ khối quy trình tiền xử lý đa bộ dữ liệu (Cắt hộp sọ + CLAHE + Chuẩn hóa đa xung BraTS2020).
- [fig2_four_datasets_class_distribution.png](file:///c:/Users/MikanQuyt/BrainTumorDetect/image/tuan1/fig2_four_datasets_class_distribution.png): Phân bố các lớp bệnh lý trên 4 bộ dữ liệu (`DS1` – `DS4`).
- [ds1_ds2_check_samples.png](file:///c:/Users/MikanQuyt/BrainTumorDetect/image/tuan1/ds1_ds2_check_samples.png), [ds3_check_samples.png](file:///c:/Users/MikanQuyt/BrainTumorDetect/image/tuan1/ds3_check_samples.png), [ds4_check_samples.png](file:///c:/Users/MikanQuyt/BrainTumorDetect/image/tuan1/ds4_check_samples.png): Minh họa trực quan kết quả cắt hộp sọ, tăng tương phản CLAHE và hộp giới hạn chuẩn.

### Tuần 2 (`image/tuan2/`) — ViT Baseline, Zero-shot SAM vs MedSAM & SAM Everything Mode
- [vit_ds1_full_finetune_loss.png](file:///c:/Users/MikanQuyt/BrainTumorDetect/image/tuan2/vit_ds1_full_finetune_loss.png) & [vit_ds2_learning_curves.png](file:///c:/Users/MikanQuyt/BrainTumorDetect/image/tuan2/vit_ds2_learning_curves.png): Bằng chứng thực nghiệm về hiện tượng Overfitting trên DS1 khi Full Fine-tuning và giới hạn `Recall < 88%` khi Linear Probing.
- [medsam_oracle_box_brisc2025_qualitative_grid.png](file:///c:/Users/MikanQuyt/BrainTumorDetect/image/tuan2/medsam_oracle_box_brisc2025_qualitative_grid.png) & [zeroshot_sam_vs_medsam_dice_boxplot.png](file:///c:/Users/MikanQuyt/BrainTumorDetect/image/tuan2/zeroshot_sam_vs_medsam_dice_boxplot.png): Đánh giá Zero-shot MedSAM (`medsam_vit_b.pth`) và SAM gốc với Oracle Box trên DS2, DS3, DS4.
- [sam_everything_summary_figure.png](file:///c:/Users/MikanQuyt/BrainTumorDetect/image/tuan2/sam_everything_summary_figure.png): Kiểm thử `SAM Everything Mode` trên 50 ảnh chứng minh `89,77%` mask tự động bắt nhầm xương sọ và mô não lành khi thiếu nhánh định vị ViT.

### Tuần 3 (`image/tuan3/`) — Kiến trúc `ViT-LoRA` & Bộ Tự sinh Lời nhắc (`Auto-Prompt Generator`)
- [fig1_lora_params_and_training_curves.png](file:///c:/Users/MikanQuyt/BrainTumorDetect/image/tuan3/fig1_lora_params_and_training_curves.png): Đối sánh tham số LoRA $r \in \{4, 8, 16, 32\}$ (`r=8` chỉ chiếm `0,34%` backbone; tổng tham số `< 1%`) và hội tụ `Focal Loss (ε=0.05)` + `GIoU + L1 Box Loss`.
- [fig2_mechanism1_rollout_layercam_ds1.png](file:///c:/Users/MikanQuyt/BrainTumorDetect/image/tuan3/fig2_mechanism1_rollout_layercam_ds1.png): Cơ chế (1) kết hợp `Attention Rollout + LayerCAM` + Lọc viền sọ sinh `Hybrid Prompt` trên Dataset 1 (`Navoneel`).
- [fig3_mechanism2_vitdet_box_head_ds2_ds3_ds4.png](file:///c:/Users/MikanQuyt/BrainTumorDetect/image/tuan3/fig3_mechanism2_vitdet_box_head_ds2_ds3_ds4.png): Cơ chế (2) `ViTDetBoxHead` dẫn đường cho `MedSAM` trên DS2 (`Dice = 93,86%`), DS3 (`Dice = 87,31%`), DS4 (`Dice = 78,24%`).

---

## 3. Hướng dẫn Chạy Kiểm thử & Huấn luyện Toàn Hệ thống

```powershell
# 1. Chạy toàn bộ bộ kiểm thử tự động (Unit Tests Tuần 1, 2, 3)
.\venv\Scripts\python.exe -m unittest discover -s tests -v

# 2. Huấn luyện ViT-LoRA (r=8, α=16) + ViTDetBoxHead trên toàn bộ hệ thống đa bộ dữ liệu (DS1, DS2, DS3, DS4)
.\venv\Scripts\python.exe -m engine.train_vit_lora --datasets all --lora-r 8 --lora-alpha 16 --loss focal --label-smoothing 0.05 --eval-medsam --out runs/vit_lora_all

# 3. Huấn luyện 5-Fold Cross-Validation trên Dataset 1 (Navoneel - Cơ chế 1: Attention Rollout + LayerCAM)
.\venv\Scripts\python.exe -m engine.train_vit_lora --datasets navoneel --task binary --no-box-head --cv-5fold --epochs 15 --out runs/vit_lora_ds1

# 4. Đánh giá Zero-shot SAM vs MedSAM với Oracle Box trên các bộ dữ liệu có nhãn Mask (DS2, DS3, DS4)
.\venv\Scripts\python.exe -m engine.eval_sam --datasets indk214 brisc2025 brats2020 --split test --out runs/eval_sam_all

# 5. Sinh bộ ảnh kết quả trực quan Tuần 3 vào image/tuan3/
.\venv\Scripts\python.exe experiments/generate_week3_hnnq_visuals.py
```
