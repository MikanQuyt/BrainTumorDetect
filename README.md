# BrainTumorDetect — Hệ Thống Phát Hiện và Phân Vùng U Não Trên Ảnh MRI Với Kiến Trúc ViTSAM + LoRA Trên Đa Bộ Dữ Liệu

Dự án xây dựng hệ thống **End-to-End ViTSAM + LoRA** tự động hoàn toàn cho bài toán **Phát hiện sớm (Classification & High-Recall Screening)**, **Tự động định vị & Sinh lời nhắc (Automated Hybrid Prompt Generation)** và **Phân vùng ranh giới khối u (PEFT Segmentation)** trên 4 bộ dữ liệu MRI sọ não chuẩn quốc tế.

---

## 1. Phân Công Nhân Sự (Kịch Bản 5 Thành Viên)

| STT | Họ và Tên | Vai Trò & Trách Nhiệm Chuyên Môn |
| :---: | :--- | :--- |
| **1** | **Huỳnh Nguyễn Nhân Quyết** *(Trưởng nhóm)* | **Kiến trúc sư Trưởng & Kỹ sư Lõi (Team Lead & Core AI Architect)**: Khởi tạo kiến trúc mô-đun, xây dựng `UnifiedBrainMRIDataset`, mã nguồn lõi (`ViT-LoRA`, `Auto-Prompt`, `MedSAM-LoRA`), ghép nối pipeline End-to-End và điều phối GitHub. |
| **2** | **Thái Văn Huy** | **Chuyên viên Dữ liệu 2D (DS1, DS2) & Nhánh ViT-LoRA**: EDA, Extreme-Point Skull Cropping + CLAHE cho Dataset 1 (`navoneel`) & Dataset 2 (`indk214`), trích xuất `gt_box`, huấn luyện/kiểm thử nhánh ViT-LoRA (Recall > 95%) và đo Box IoU. |
| **3** | **Lâm Gia Bảo** | **Chuyên viên Dữ liệu Đa xung (DS4 BraTS2020) & Nhánh MedSAM-LoRA**: EDA 3D, chuẩn hóa Z-score, cắt lát 2D `[FLAIR, T1ce, T2]` + mặt nạ Whole Tumor (`>= 50 px`) cho BraTS2020; thực nghiệm SAM gốc vs. MedSAM vs. MedSAM-LoRA (Dice, IoU, HD95). |
| **4** | **Nguyễn Thành Hải** | **Chuyên viên Dữ liệu BRISC 2025 (DS3), Ablation Studies & Failure Cases**: Kiểm định chất lượng 1-1 Ảnh–Mask, lọc CLAHE, trích xuất `gt_box`, chia tập Stratified 70/15/15 chống rò rỉ bệnh nhân trên BRISC 2025; chạy Ablation hạng LoRA $r \in \{4,8,16,32\}$ & chiến lược Prompt. |
| **5** | **Nguyễn Phước Hoàng Quân** | **Trưởng ban Tài liệu Học thuật, Trực quan hóa & UI/UX Lead**: Biên soạn báo cáo học thuật, thiết kế sơ đồ kiến trúc & biểu đồ thực nghiệm, xây dựng ứng dụng Web lâm sàng CDSS (Gradio/Streamlit 5 khung hiển thị) và Slide bảo vệ. |

---

## 2. Cấu Trúc Thư Mục Chuẩn Hóa Toàn Dự Án (Tuần 1 → Tuần 8)

```text
BrainTumorDetect/
├── configs/
│   └── default.yaml                 # Cấu hình siêu tham số, kích thước ảnh (224/1024), Albumentations & đường dẫn 4 JSON
├── data/                            # Mô-đun Dữ liệu & Tiền xử lý (Tuần 1)
│   ├── __init__.py
│   ├── unified_dataset.py           # Lớp UnifiedBrainMRIDataset(torch.utils.data.Dataset) cho cả 4 bộ dữ liệu
│   ├── transforms.py                # Pipeline Albumentations đồng bộ ảnh, mask và bounding box (pascal_voc)
│   ├── skull_crop.py                # Hàm cắt sát hộp sọ (Extreme-Point Contour), CLAHE, nhị phân hóa mask & mask_to_box
│   ├── ds12_index.py                # Lập chỉ mục Dataset 1 (navoneel) & Dataset 2 (indk214)
│   ├── eda_ds1_ds2.py               # Phân tích thống kê (EDA) Dataset 1 & 2
│   ├── build_ds1_ds2.py             # Skull Cropping + CLAHE toàn bộ DS1 & DS2 -> ds1/ds2_annotations.json
│   ├── check_ds1_ds2.py             # Kiểm định tự động đầu ra DS1 & DS2
│   ├── ds3_index.py                 # Phân tích tên file & đối sánh 1-1 hai chiều cho Dataset 3 (BRISC 2025)
│   ├── eda_ds3.py                   # Kiểm định chất lượng 1-1, quét trùng lặp SHA-256 & vẽ biểu đồ EDA DS3
│   ├── build_ds3.py                 # Lọc CLAHE, trích xuất gt_box, gom cụm bệnh nhân Union-Find & chia tập 70/15/15
│   ├── check_ds3.py                 # Kiểm chứng tự động 0% rò rỉ bệnh nhân / SHA-256 trên DS3
│   ├── eda_brats.py                 # Khảo sát thống kê 369 ca 3D MRI Dataset 4 (BraTS 2020)
│   ├── build_dataset.py             # Chuẩn hóa Z-score, cắt lát 2D [FLAIR, T1ce, T2], tạo WT mask (>=50px) & bbox
│   ├── check_dataset.py             # Kiểm định chất lượng lát cắt 2D BraTS 2020
│   └── README.md                    # Hướng dẫn chi tiết mô-đun data/
├── dataset/                         # Thư mục lưu dữ liệu ảnh & 4 tệp JSON chú thích chuẩn
│   ├── ds1_annotations.json         # 253 bản ghi Dataset 1 (Navoneel, kèm chia sẵn 5-Fold CV)
│   ├── ds2_annotations.json         # 9.215 bản ghi Dataset 2 (2.192 cặp Segmentation có gt_box + 7.023 ảnh Classification)
│   ├── ds3_annotations.json         # 6.000 bản ghi Dataset 3 (BRISC 2025, chia Stratified 70/15/15 chống rò rỉ bệnh nhân)
│   └── brats2020_annotations.json   # Bản ghi lát cắt 2D đa xung Dataset 4 (BraTS 2020)
├── models/                          # Kiến trúc Mô hình (Tuần 2 → Tuần 5)
│   └── __init__.py                  # ViT Baseline, SAM/MedSAM Wrapper, ViT-LoRA, Auto-Prompt, MedSAM-LoRA, End-to-End
├── engine/                          # Luồng Huấn luyện, Đánh giá, Ablation & Kiểm thử chéo (Tuần 2 → Tuần 6)
│   └── __init__.py
├── utils/                           # Hàm Mất mát, Đo lường (Dice, IoU, HD95, Recall), Hậu xử lý & Chỉ số Lâm sàng (Tuần 2 → Tuần 7)
│   └── __init__.py
├── app/                             # Ứng dụng Giao diện Lâm sàng CDSS Web App trên Gradio/Streamlit (Tuần 7)
│   └── __init__.py
├── checkpoints/                     # Thư mục lưu trọng số SAM/MedSAM gốc & LoRA Adapters (Tuần 2 → Tuần 7)
│   └── .gitkeep
├── reports/                         # Thư mục tập trung Báo cáo thống kê, Bảng CSV & Biểu đồ (Tuần 1 → Tuần 8)
│   ├── eda_ds1_ds2/                 # Báo cáo BAO_CAO_TUAN1.md, dataset_statistics.json, CSV & biểu đồ EDA DS1-DS2
│   ├── build_ds1_ds2/               # Nhật ký crop_log.csv và lưới ảnh check_samples.png của DS1-DS2
│   ├── eda_ds3/                     # Báo cáo audit_report.json, eda_summary.txt, duplicate_sha256.csv & biểu đồ DS3
│   ├── build_ds3/                   # Bảng split_summary.csv, brisc2025_processed_manifest.csv, samples/ & check_samples.png
│   ├── eda_ds4/                     # Báo cáo eda_notes.md, eda_overview.txt, eda_summary.csv & biểu đồ 3D BraTS2020
│   └── build_ds4/                   # Kết quả mẫu case_summary.csv, metadata.csv, splits.csv & check_samples.png của DS4
├── tests/                           # Bộ kiểm thử tự động (Unit & Integration Tests)
│   ├── __init__.py
│   └── test_unified_dataset.py      # 7 bài kiểm thử UnifiedBrainMRIDataset & Albumentations
├── LICENSE
├── requirement.txt
└── README.md
```

---

## 3. Tóm Tắt Kết Quả Chuẩn Hóa 4 Bộ Dữ Liệu (Hoàn Thành Tuần 1)

| Bộ Dữ Liệu | Quy Mô & Đặc Trưng | Quy Trình Tiền Xử Lý Đã Thực Hiện | Tệp Chú Thích Đầu Ra |
| :--- | :--- | :--- | :--- |
| **Dataset 1: Navoneel** (`navoneel`) | **253 ảnh 2D MRI** (`155` có u, `98` không u), 184 kích thước khác nhau, chỉ có nhãn phân loại nhị phân. | Loại bỏ 253 tệp trùng trong `brain_tumor_dataset/`; cắt sát hộp sọ tự động (**Extreme-Point Contour + CLAHE**); chia sẵn **5-Fold Stratified CV**. | `dataset/ds1_annotations.json` |
| **Dataset 2: indk214** (`indk214`) | **2.192 cặp Ảnh–Mask** (`512×512`) + **7.023 ảnh Classification** thuộc 4 lớp (`no_tumor`, `glioma`, `meningioma`, `pituitary`). | Cắt sát hộp sọ + CLAHE đồng bộ Ảnh & Mask; nhị phân hóa mask (`> 127`); trích xuất hộp chuẩn `gt_box = [x_min, y_min, x_max, y_max]`; chia tập **70% / 15% / 15%**. | `dataset/ds2_annotations.json` |
| **Dataset 3: BRISC 2025** (`brisc2025`) | **6.000 lát cắt T1 MRI** (`4.793` cặp có Mask + `1.207` ảnh `no_tumor`), 183 độ phân giải, 0 ảnh hỏng, 0 mask rỗng. | Kiểm định khớp nối 1-1; phát hiện & gom **46 nhóm ảnh trùng SHA-256** vào 3.646 cụm `patient_id` bằng **Union-Find**; lọc CLAHE, trích xuất `gt_box` & chia **Stratified 70/15/15 (0% rò rỉ bệnh nhân)**. | `dataset/ds3_annotations.json` |
| **Dataset 4: BraTS 2020** (`brats2020`) | **369 ca 3D MRI** (`240×240×155`), đủ 4 chuỗi xung (`FLAIR`, `T1`, `T1ce`, `T2`) và mặt nạ `seg` (`{0, 1, 2, 4}`). | Chuẩn hóa **Z-score** trên vùng mô não ($\text{voxel} > 0$); cắt lát 2D Axial, lọc lát rỗng (`< 500 px`), giữ lát u $\ge 50\text{ px}$; ghép 3 kênh `[FLAIR, T1ce, T2]`; gộp mặt nạ **Whole Tumor ($\text{WT}$)** kèm `gt_box`; chia **70/15/15 theo ca bệnh**. | `dataset/brats2020_annotations.json` |

---

## 4. Hướng Dẫn Cài Đặt & Chạy Kiểm Thử Nhanh

```bash
# 1. Cài đặt thư viện phụ thuộc
pip install -r requirement.txt

# 2. Chạy bộ kiểm thử tự động cho UnifiedBrainMRIDataset & Albumentations
python -m unittest discover tests
```
