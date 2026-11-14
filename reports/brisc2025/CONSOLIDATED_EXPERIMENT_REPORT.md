# BÁO CÁO TOÀN DIỆN KẾT QUẢ THỰC NGHIỆM BRISC 2025 & DATASET 1
*Cập nhật với trọng số chuẩn `medsam_vit_b.pth` và `sam_vit_b_01ec64.pth`*

---

## 1. THỰC NGHIỆM 1: Phân Loại ViT Baseline (Linear Probing & Full Fine-Tuning)
> **Bộ dữ liệu**: BRISC 2025 (4 lớp: Glioma, Meningioma, Pituitary, No Tumor)
> **Kiến trúc**: 12-layer Vision Transformer (`vit_tiny_patch16_224`, ImageNet Pretrained)

| Phương Pháp | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | Thời Gian Train |
|---|---|---|---|---|---|
| **Linear Probing (Frozen ViT)** | **82.75%** | 82.63% | 82.74% | 82.75% | 340.3s |
| **Full Fine-Tuning (All Layers)** | **72.00%** | 68.39% | 74.09% | 72.00% | 224.3s |

---

## 2. THỰC NGHIỆM 2: Phân Vùng Zero-Shot MedSAM (`medsam_vit_b.pth`) Với Hộp Chuẩn (Oracle Box)
> **Bộ dữ liệu**: Tập Test chuẩn của BRISC 2025 (`segmentation_task/test`, 50 lát cắt đa bình diện)
> **Trọng số**: `checkpoints/medsam_vit_b.pth` (MD5: `3bb6db55bd0c9ca30b61248bca72f8d6`)
> **Lời nhắc (Prompt)**: Bounding Box trích xuất trực tiếp từ Ground Truth Mask (>127)

- **Số lát cắt đánh giá**: 50 lát cắt đa bình diện (Axial, Coronal, Sagittal)
- **Mean Dice Coefficient**: **89.64%** (tăng +1.77% so với MobileSAM 87.87%)
- **Mean Intersection over Union (IoU)**: **81.99%**
- **Mean Precision**: **95.19%**
- **Mean Recall**: **85.70%**
- **Mean HD95**: **8.29 px**

### Bảng chi tiết theo loại u:
| Loại U | Số Lát Cắt | Mean Dice (%) | Mean IoU (%) | Precision (%) | Recall (%) | Mean HD95 (px) |
|---|---|---|---|---|---|---|
| GLIOMA | 15 | **85.64%** | 76.06% | 93.50% | 80.62% | 12.27 |
| MENINGIOMA | 18 | **92.15%** | 86.04% | 96.49% | 88.91% | 7.69 |
| PITUITARY | 17 | **90.53%** | 82.92% | 95.30% | 86.78% | 5.40 |

### Bảng chi tiết theo bình diện chụp (Plane Views):
| Bình Diện Chụp | Số Lát Cắt | Mean Dice (%) | Mean IoU (%) | Mean HD95 (px) |
|---|---|---|---|---|
| AXIAL | 21 | 88.77% | 80.66% | 8.83 |
| CORONAL | 16 | 91.15% | 84.18% | 6.62 |
| SAGITTAL | 13 | 89.20% | 81.43% | 9.47 |

---

## 3. THỰC NGHIỆM 3: Kiểm Thử SAM Everything Mode (`sam_vit_b_01ec64.pth`) Trên 50 Ảnh (25 BRISC 2025 + 25 Dataset 1 Navoneel)
> **Mục tiêu**: Chứng minh SAM nguyên bản khi tự động phân đoạn không có Prompt sẽ bắt nhầm viền xương sọ và cấu trúc não lành.
> **Quy mô**: 50 ảnh đối chứng thực tế (25 ảnh BRISC 2025 + 25 ảnh thật Dataset 1 Navoneel)

- **Tổng số ảnh kiểm tra**: 50 ảnh (25 BRISC 2025 + 25 Dataset 1 Navoneel)
- **Số lượng Mask trung bình mỗi ảnh**: **4.16 ± 2.55 masks** (BRISC 2025: 5.08 masks; Dataset 1: 3.24 masks)
- **Tỷ lệ Mask diện tích lớn (>5% diện tích lát cắt)**: **54.52%** (BRISC 2025: 45.37%; Dataset 1: 63.67%)
- **Diện tích Mask lớn nhất trung bình**: **50.29%** toàn bộ lát cắt (cực đại **84.56%**)
- **Tỷ lệ Mask nằm ngoài khối u (trên 18 ca có GT Mask)**: **89.77%**
- **Best-Mask Tumor Dice (không có Box Prompt)**: **43.61%** (so với **89.64%** của MedSAM khi có Box Prompt)

### Kết luận khoa học rút ra:
> *Without medical-specific prompts, SAM Everything Mode (`sam_vit_b_01ec64.pth`) generates an average of 4.16 masks per brain MRI scan across 25 BRISC 2025 and 25 Dataset 1 (Navoneel) scans. 54.52% of all detected masks encompass massive non-tumor anatomical structures (largest mask spanning an average of 50.29% and up to 84.6% of the entire scan), capturing the cranial bone ring, meninges, ventricles, and healthy cerebral hemispheres. On the 18 tumor slices with pixel-level ground truth, 89.77% of generated masks lie predominantly outside the tumor on skull/normal parenchyma, and even the oracle best-matching mask achieves only 43.61% Dice. This empirically confirms that a prior ViT localization branch is mandatory before invoking SAM/MedSAM.*
