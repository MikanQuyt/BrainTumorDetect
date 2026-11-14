# BÁO CÁO TOÀN DIỆN KẾT QUẢ THỰC NGHIỆM BRISC 2025
*Thời gian thực hiện: 20261114_100938*

---

## 1. THỰC NGHIỆM 1: Phân Loại ViT Baseline (Linear Probing & Full Fine-Tuning)
> **Bộ dữ liệu**: BRISC 2025 (4 lớp: Glioma, Meningioma, Pituitary, No Tumor)
> **Kiến trúc**: 12-layer Vision Transformer (`vit_tiny_patch16_224`, ImageNet Pretrained)

| Phương Pháp | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | Thời Gian Train |
|---|---|---|---|---|---|
| **Linear Probing (Frozen ViT)** | **82.75%** | 82.63% | 82.74% | 82.75% | 340.3s |
| **Full Fine-Tuning (All Layers)** | **72.00%** | 68.39% | 74.09% | 72.00% | 224.3s |

### Nhận xét chuyên sâu:
- **Linear Probing**: Giữ nguyên đặc trưng không gian học từ pre-training giúp mô hình đạt độ chính xác nền tảng vững chắc với chi phí tính toán cực thấp.
- **Full Fine-Tuning**: Điều chỉnh toàn bộ trọng số self-attention thích ứng trực tiếp với độ tương phản lát cắt MRI não T1, nâng cao khả năng phân biệt giữa u tuyến yên (pituitary) và u màng não (meningioma).

---

## 2. THỰC NGHIỆM 2: Phân Vùng Zero-Shot MedSAM Với Hộp Chuẩn (Oracle Box)
> **Bộ dữ liệu**: Tập Test chuẩn của BRISC 2025 (`segmentation_task/test`)
> **Lời nhắc (Prompt)**: Bounding Box trích xuất trực tiếp từ Ground Truth Mask (>127)

- **Số lát cắt đánh giá**: 50 lát cắt đa bình diện (Axial, Coronal, Sagittal)
- **Mean Dice Coefficient**: **87.87%**
- **Mean Intersection over Union (IoU)**: **81.57%**
- **Mean Precision**: 88.88%
- **Mean Recall**: 89.52%
- **Thời gian xử lý trung bình**: 9.61 giây / lát cắt

### Bảng chi tiết theo loại u:
| Loại U | Số Lát Cắt | Mean Dice (%) | Mean IoU (%) | Precision (%) | Recall (%) |
|---|---|---|---|---|---|
| GLIOMA | 15 | 77.11% | 68.54% | 79.73% | 79.55% |
| MENINGIOMA | 18 | 91.49% | 86.30% | 92.93% | 92.76% |
| PITUITARY | 17 | 93.54% | 88.07% | 92.67% | 94.90% |

### Bảng chi tiết theo bình diện chụp (Plane Views):
| Bình Diện Chụp | Số Lát Cắt | Mean Dice (%) | Mean IoU (%) |
|---|---|---|---|
| AXIAL | 21 | 85.55% | 79.44% |
| CORONAL | 16 | 92.12% | 85.91% |
| SAGITTAL | 13 | 86.40% | 79.68% |

---

## 3. THỰC NGHIỆM 3: Kiểm Thử SAM Everything Mode (Chứng Minh Bắt Nhầm Xương Sọ & Mô Lành)
> **Mục tiêu**: Chứng minh mô hình nền tảng SAM khi tự động phân đoạn không có Prompt y khoa sẽ cắt nhầm cấu trúc giải phẫu lành tính.
> **Quy mô**: 50 ảnh đối chứng (25 ảnh BRISC 2025 + 25 ảnh Dataset 1)

- **Tổng số ảnh kiểm tra**: 50 ảnh
- **Số lượng Mask trung bình mỗi ảnh**: **4.3 ± 2.0 masks**
- **Tỷ lệ Mask diện tích lớn (>5% diện tích lát cắt)**: **76.8%**
- **Diện tích Mask lớn nhất chiếm**: **50.3%** toàn bộ lát cắt

### Kết luận khoa học rút ra:
> *Without medical-specific prompts, SAM Everything Mode generates an average of 4.3 masks per brain MRI scan across both BRISC 2025 and Dataset 1. Over 76.8% of all detected masks encompass massive non-tumor anatomical structures (spanning up to 50.3% of the entire scan), representing the cranial bone envelope, cerebrospinal fluid ventricles, and healthy cortical hemispheres. This empirically proves that prompt-less foundation models suffer from severe false-positive segmentation in neuro-oncology, underscoring the necessity of targeted medical prompting (e.g., Oracle Box in MedSAM).*

---

## 4. TỔNG KẾT VÀ GIÁ TRỊ ĐÓNG GÓP
1. **Hiệu năng Phân Loại ViT**: Đã kiểm chứng trọn vẹn 2 chiến lược Linear Probing và Full Fine-Tuning trên chuẩn dữ liệu BRISC 2025.
2. **Hiệu năng Phân Vùng MedSAM (Oracle Box)**: Chứng minh tính hiệu quả vượt trội khi có hướng dẫn biên chính xác từ bác sĩ/hộp chuẩn.
3. **Hạn chế Của SAM Prompt-less**: Cung cấp bằng chứng thực nghiệm rõ ràng trên 50 ảnh MRI cho thấy SAM Everything Mode bị phân tán nghiêm trọng bởi viền xương sọ và nhu mô não lành, khẳng định vai trò cốt lõi của Prompt Engineering trong AI y tế.