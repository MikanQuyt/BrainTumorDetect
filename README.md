# BrainTumorDetect — Tuần 1: EDA, Cắt sát hộp sọ tự động (Skull Cropping + CLAHE) & Trích xuất Bounding Box cho Dataset 1 & 2

**Sinh viên thực hiện:** Thái Văn Huy · MSSV: 23130068 (`thai_van_huy_tuan1`)

Nhánh **`thai_van_huy_tuan1`** thực hiện trọn vẹn 2 nhiệm vụ Tuần 1:
1. **Tải và phân tích thống kê (EDA)** cho **Dataset 1 (`navoneel` – 253 ảnh)** và **Dataset 2 (`indk214` – Segmentation & Classification)**.
2. **Lập trình và chạy script Cắt sát hộp sọ tự động (Automated Extreme-Point Skull Cropping bằng OpenCV + CLAHE)** trên toàn bộ Dataset 1 & 2; tự động trích xuất tọa độ hộp chuẩn **`[x_min, y_min, x_max, y_max]`** bao quanh Ground-Truth Mask của Dataset 2 xuất ra tệp **`ds2_annotations.json`**.

> Xem chi tiết nội dung rà soát và chỉnh sửa tại: **[chinhsuaHuy.md](chinhsuaHuy.md)** và báo cáo thống kê chi tiết tại **[output_lan2/BAO_CAO_TUAN1.md](output_lan2/BAO_CAO_TUAN1.md)**.

---

## 1. Cấu trúc thư mục trên nhánh

```text
BrainTumorDetect/
├── data/
│   ├── skull_crop.py                # Thuật toán Extreme-Point Skull Cropping (OpenCV), CLAHE & mask_to_box
│   ├── ds12_index.py                # Bộ lập chỉ mục chuẩn hóa cho Dataset 1 (navoneel) và Dataset 2 (indk214)
│   ├── eda_ds1_ds2.py               # Phân tích thống kê (EDA) & xuất biểu đồ cho Dataset 1 & 2
│   ├── build_ds1_ds2.py             # Chạy Skull Cropping + CLAHE toàn bộ DS1 & DS2, xuất ds1/ds2_annotations.json
│   ├── check_ds1_ds2.py             # Kiểm chứng tự động ảnh/mask đã cắt, tính nhị phân {0, 255} và tọa độ gt_box
│   └── README.md                    # Hướng dẫn chi tiết mô-đun data/
├── preprocess.py                    # Script tích hợp thống kê, cắt hộp sọ cực trị + CLAHE & xuất output_lan2/ds2_annotations.json
├── output_lan2/
│   ├── BAO_CAO_TUAN1.md             # Báo cáo phân tích thống kê (EDA) và kết quả tiền xử lý Dataset 1 & 2
│   ├── dataset_statistics.json      # Toàn bộ số liệu thống kê kích thước, phân bố nhãn, ảnh trùng & diện tích u
│   ├── ds2_annotations.json         # Tọa độ hộp chuẩn [x_min, y_min, x_max, y_max] của 2.192 cặp ảnh–mask Dataset 2
│   └── ds_annotations.json          # Bản sao tương thích ngược
├── dataset/
│   ├── ds1_annotations.json         # 253 bản ghi chuẩn hóa của Dataset 1 (sau khi cắt hộp sọ + CLAHE)
│   └── ds2_annotations.json         # 9.215 bản ghi chuẩn hóa của Dataset 2 (2.192 mẫu có mask + gt_box, 7.023 mẫu phân loại)
├── reports/
│   ├── eda_ds1_ds2/                 # Báo cáo & biểu đồ EDA: eda_summary.txt, class_counts.csv, fig_*.png
│   └── build_ds1_ds2/               # Nhật ký cắt hộp sọ (crop_log.csv) và lưới ảnh kiểm chứng (check_samples.png)
├── chinhsuaHuy.md                   # Báo cáo kiểm tra và các hạng mục đã hoàn thiện trên nhánh thai_van_huy_tuan1
└── README.md
```

---

## 2. Tóm tắt kết quả EDA – Dataset 1 & Dataset 2

- **Dataset 1 (`navoneel`)**: **253 ảnh** (`yes` – có u: **155**, `no` – không u: **98**), đã loại bỏ 253 tệp bản sao trong `brain_tumor_dataset/`; gồm **184 kích thước ảnh khác nhau** (`150×168` đến `1920×1427`), **0 ảnh hỏng**.
- **Dataset 2 (`indk214` – Classification)**: **7.023 ảnh** (`Training`: **5.712**, `Testing`: **1.311**) thuộc 4 lớp (`no_tumor`: **2.000**, `pituitary`: **1.757**, `meningioma`: **1.645**, `glioma`: **1.621**).
- **Dataset 2 (`indk214` – Segmentation)**: **2.192 cặp ảnh – mask** (`pituitary`: **930**, `meningioma`: **708**, `glioma`: **554**), đồng nhất kích thước `512 × 512`, **0 ảnh thiếu mask**, **0 mask rỗng**. Nền mask gốc có mức xám `2–3` do nén ảnh nên được nhị phân hóa bằng ngưỡng `>= 128` (`> 127`).

---

## 3. Hướng dẫn chạy lệnh

```bash
# Chạy phân tích thống kê (EDA) và xuất biểu đồ vào reports/eda_ds1_ds2/
python -m data.eda_ds1_ds2

# Chạy Automated Extreme-Point Skull Cropping + CLAHE và xuất dataset/ds2_annotations.json
python -m data.build_ds1_ds2

# Kiểm chứng tự động toàn bộ ảnh/mask đã cắt và tọa độ gt_box [x_min, y_min, x_max, y_max]
python -m data.check_ds1_ds2

# Hoặc chạy script tích hợp preprocess.py để xuất ra output_lan2/ds2_annotations.json
python preprocess.py --navoneel dataset/navoneel --indk214 dataset/indk214 --output output_lan2
```


===


# BRISC 2025 (Dataset 3) — Kiểm Định Chất Lượng, Lọc CLAHE, Trích Xuất Bounding Box & Chia Tập 70% / 15% / 15% Chống Rò Rỉ Bệnh Nhân

**Nhánh thực hiện:** `Nguyen_Thanh_Hai_tuan1`

Dự án thực hiện đầy đủ quy trình kiểm định chất lượng dữ liệu **BRISC 2025 (Dataset 3)**, kiểm tra sự khớp nối kích thước 1-1 giữa từng ảnh MRI và tệp Mask tương ứng, loại bỏ ảnh lỗi/hỏng hoặc nhãn rỗng sai quy cách, tiền xử lý tăng cường tương phản bằng **CLAHE**, trích xuất tọa độ **Bounding Box chuẩn `[x_min, y_min, x_max, y_max]`** từ Mask, và phân chia tập dữ liệu **Stratified Train / Validation / Test (70% / 15% / 15%)** đảm bảo **không rò rỉ dữ liệu cùng một bệnh nhân (Zero Patient & SHA-256 Data Leakage)**.

---

## 1. Kết Quả Kiểm Định Chất Lượng Dataset 3 (BRISC 2025)

Chi tiết được xuất tự động tại [audit_report.json](audit_report.json) và [reports/eda_ds3/eda_summary.txt](reports/eda_ds3/eda_summary.txt):

- **Tính toàn vẹn Checksum SHA-256 của Manifest**:
  - `manifest.csv`: **MATCH (Hợp lệ 100%)**
  - `manifest.json`: **MATCH (Hợp lệ 100%)**
- **Tổng số lát cắt T1 MRI duy nhất**: **6.000 mẫu** (`segmentation_task` có mask: **4.793 cặp**; `classification_task/no_tumor`: **1.207 ảnh**).
  - 4.793 ảnh có u (`glioma`: 1.401, `meningioma`: 1.635, `pituitary`: 1.757) trong `classification_task` trùng khớp 100% tên file và mã băm SHA-256 với `segmentation_task/images`.
- **Kiểm tra khớp nối 1-1 & Loại bỏ mẫu lỗi / nhãn rỗng**:
  - Tệp bị thiếu (`missing_files`): **0 / 15.586**
  - Ảnh hoặc Mask lỗi/hỏng không đọc được (`corrupted_files`): **0**
  - Lệch kích thước `(W, H)` 1-1 giữa Ảnh MRI và Mask (`dimension_mismatches`): **0** (kiểm định trên toàn bộ **183 độ phân giải** khác nhau: phổ biến nhất là `512×512` với 4.881 mẫu, `369×369` với 363 mẫu, `216×369` với 354 mẫu...).
  - Nhãn rỗng sai quy cách (`empty_masks`, mẫu có u nhưng diện tích vùng u sau nhị phân hóa `= 0 px`): **0**.
- **Phát hiện trùng lặp nội dung (`SHA-256`) trong bộ dữ liệu gốc**:
  - Phát hiện **46 nhóm ảnh trùng lặp mã băm SHA-256** (50 ảnh trùng dư), trong đó có **7 nhóm bị rò rỉ trực tiếp giữa tập `train` và `test` gốc** của BRISC 2025 (xem [reports/eda_ds3/duplicate_sha256.csv](reports/eda_ds3/duplicate_sha256.csv)).
- **Đặc điểm mức xám của Mask**:
  - Giá trị pixel: `[0..10, 246..255]` do làm mịn biên và nén ảnh trước đó; sử dụng ngưỡng nhị phân hóa `mask > 127` (`binarize_mask`) để thu được mask nhị phân chuẩn `{0, 255}`.

---

## 2. Kỹ Thuật Tiền Xử Lý & Chia Tập 70% / 15% / 15% Chống Rò Rỉ Bệnh Nhân

### A. Tăng Cường Tương Phản Cục Bộ CLAHE
- **Tham số**: `clipLimit = 2.0`, `tileGridSize = (8, 8)` (`cv2.createCLAHE`).
- **Đầu ra**: Toàn bộ 6.000 ảnh MRI đã lọc CLAHE (`dataset/ds3_processed/images/`) và 4.793 mask nhị phân chuẩn `{0, 255}` (`dataset/ds3_processed/masks/`).

### B. Trích Xuất Tọa Độ Bounding Box Chuẩn Từ Ground-Truth Mask
- Trích xuất tọa độ hộp chuẩn toàn cục **`gt_box = [x_min, y_min, x_max, y_max]`** bao sát toàn bộ vùng `mask > 127` trên từng lát cắt có u (tương thích trực tiếp với `UnifiedBrainMRIDataset` và nhánh prompt của MedSAM), đồng thời xuất kèm danh sách hộp thành phần (`bboxes_pixel_xyxy`) và định dạng YOLO chuẩn hóa `[class_id, x_center, y_center, width, height]` theo kích thước thực tế `(W, H)` của từng ảnh.

### C. Phân Cụm Bệnh Nhân (Union-Find) & Chia Tập Stratified 70% / 15% / 15%
- **Khôi phục cụm bệnh nhân (`patient_id`)**: Do tên tệp BRISC 2025 (`brisc2025_<split>_<index>_<tumor>_<plane>_t1`) không chứa mã bệnh nhân tường minh, thuật toán đồ thị liên thông (**Union-Find**) liên kết các lát cắt của cùng một bệnh nhân dựa trên:
  1. **Trùng khớp tuyệt đối mã băm `SHA-256`** (gom toàn bộ 46 cụm ảnh trùng lặp về cùng một `patient_id`, chặn đứng rò rỉ chéo giữa `train` và `test` gốc).
  2. **Chuỗi lát cắt liên tiếp** (`|index_i - index_j| == 1`) trong cùng tập gốc, loại u, mặt phẳng cắt và độ phân giải có tương quan cấu trúc `32 × 32` $\ge 0.90$.
  3. **Tương đồng cấu trúc toàn cục** $\ge 0.96$ trong cùng loại u, mặt phẳng cắt và độ phân giải.
- **Kết quả chia tập Stratified Group 70% / 15% / 15%** (chi tiết tại [reports/build_ds3/split_summary.csv](reports/build_ds3/split_summary.csv) và [dataset/ds3_annotations.json](dataset/ds3_annotations.json)):

| Tập chia (`split`) | Tổng số mẫu | Tỷ lệ (`%`) | Số cụm bệnh nhân | Cặp Segmentation (có Mask + `gt_box`) | `no_tumor` (0) | `glioma` (1) | `meningioma` (2) | `pituitary` (3) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Train** | **4.200** | **70.00%** | 2.547 | **3.355** | 845 | 981 | 1.145 | 1.229 |
| **Validation** | **900** | **15.00%** | 550 | **719** | 181 | 210 | 245 | 264 |
| **Test** | **900** | **15.00%** | 549 | **719** | 181 | 210 | 245 | 264 |
| **Tổng cộng** | **6.000** | **100.00%** | **3.646** | **4.793** | **1.207** | **1.401** | **1.635** | **1.757** |

- **Kiểm chứng chống rò rỉ dữ liệu (Zero Data Leakage)**:
  $$\text{Patient}_{\text{Train}} \cap \text{Patient}_{\text{Val}} = \emptyset, \quad \text{Patient}_{\text{Train}} \cap \text{Patient}_{\text{Test}} = \emptyset, \quad \text{Patient}_{\text{Val}} \cap \text{Patient}_{\text{Test}} = \emptyset$$
  $$\text{SHA256}_{\text{Train}} \cap \text{SHA256}_{\text{Val}} = \emptyset, \quad \text{SHA256}_{\text{Train}} \cap \text{SHA256}_{\text{Test}} = \emptyset, \quad \text{SHA256}_{\text{Val}} \cap \text{SHA256}_{\text{Test}} = \emptyset$$

---

## 3. Cấu Trúc Thư Mục Trên Nhánh

```text
BrainTumorDetect/
├── data/
│   ├── ds3_index.py                         # Phân tích tên file & đối sánh 1-1 hai chiều cho BRISC 2025
│   ├── eda_ds3.py                           # Kiểm định chất lượng 1-1, phát hiện trùng SHA-256 & vẽ biểu đồ EDA
│   ├── build_ds3.py                         # Lọc CLAHE toàn bộ 6.000 ảnh, trích gt_box & chia tập 70/15/15
│   ├── check_ds3.py                         # Kiểm chứng tự động khớp nối 1-1, gt_box & 0% rò rỉ bệnh nhân
│   ├── skull_crop.py                        # Các hàm xử lý ảnh/mask dùng chung (apply_clahe, binarize_mask, mask_to_box)
│   └── README.md                            # Hướng dẫn nhanh mô-đun data/
├── audit_dataset.py                         # Script kiểm định checksum, kích thước 1-1, nhãn rỗng & trùng SHA-256
├── audit_report.json                        # Báo cáo JSON kết quả kiểm định chất lượng BRISC 2025
├── preprocess_pipeline.py                   # Pipeline CLAHE, trích xuất gt_box/YOLO bbox & chia tập 70/15/15
├── dataset/
│   └── ds3_annotations.json                 # 6.000 bản ghi chuẩn hóa của BRISC 2025 cho UnifiedBrainMRIDataset
├── output/
│   ├── brisc2025_processed_manifest.csv     # Bảng tổng hợp tọa độ gt_box, YOLO bbox, patient_id và split_70_15_15
│   └── samples/                             # Ảnh đối chứng 4 khung: Original - CLAHE - Mask - Tumor BBox
├── reports/
│   ├── eda_ds3/                             # Báo cáo & biểu đồ EDA: eda_summary.txt, duplicate_sha256.csv, fig_*.png
│   └── build_ds3/                           # Bảng split_summary.csv, process_log.csv và lưới ảnh check_samples.png
└── README.md
```

---

## 4. Hướng Dẫn Chạy Lệnh

```bash
# 1. Kiểm định chất lượng & xuất báo cáo EDA + biểu đồ vào reports/eda_ds3/
python -m data.eda_ds3
python audit_dataset.py

# 2. Chạy lọc CLAHE, trích xuất gt_box & chia tập Stratified 70% / 15% / 15%
python -m data.build_ds3
python preprocess_pipeline.py

# 3. Kiểm chứng tự động đầu ra và xác nhận 0% rò rỉ bệnh nhân / SHA-256
python -m data.check_ds3
```


===


# BrainTumorDetect — Tuần 1: Tiền xử lý & Cắt lát 2D Dataset 4 (BraTS 2020)

Nhánh **`Lam_Gia_Bao_tuan1`** thực hiện toàn bộ quy trình khảo sát thống kê (EDA), chuẩn hóa cường độ tín hiệu Z-score 3D, cắt lát ngang 2D (Axial Slicing), gộp mặt nạ Whole Tumor (WT), trích xuất tọa độ Bounding Box chuẩn và lấy mẫu cân bằng lát lành cho bộ dữ liệu **BraTS 2020 (369 ca bệnh 3D MRI)**.

---

## 1. Cấu trúc thư mục trên nhánh

```text
BrainTumorDetect/
├── src/
│   └── data/
│       ├── eda_brats.py         # Khảo sát thống kê 3D BraTS2020 (hỗ trợ cả .nii/.nii.gz và .h5)
│       ├── build_dataset.py     # Pipeline chuẩn hóa Z-score, cắt lát 2D, tạo WT mask, bbox & chia tập theo ca
│       ├── check_dataset.py     # Kiểm định tự động không rò rỉ ca bệnh, ngưỡng diện tích u >= 50px, bbox & Z-score
│       └── README.md            # Hướng dẫn nhanh mô-đun src/data/
├── reports/
│   ├── eda_overview.txt         # Tổng kết số liệu khảo sát 369 ca BraTS2020
│   ├── eda_notes.md             # Phân tích chi tiết và quyết định thiết kế pipeline
│   ├── eda_summary.csv          # Thống kê từng ca bệnh (shape, spacing, số lát não, số lát u, thể tích WT)
│   ├── fig_*.png                # Các biểu đồ phân bố thể tích u, tỷ lệ lát u, cường độ tín hiệu theo modality
│   └── processed/               # Kết quả xuất mẫu: case_summary.csv, metadata.csv, splits.csv, check_samples.png
├── .gitignore
└── README.md
```

---

## 2. Tóm tắt kết quả khảo sát (EDA) trên 369 ca BraTS 2020

- **Số lượng & Định dạng**: **369/369** ca bệnh đầy đủ 4 chuỗi xung (`FLAIR`, `T1`, `T1ce`, `T2`) và mặt nạ phân vùng `seg`, không có ca nào thiếu modality hoặc lệch kích thước.
- **Kích thước & Hệ trục**: Đồng nhất `240 × 240 × 155` voxel, khoảng cách điểm ảnh `1.0 × 1.0 × 1.0 mm`, hướng trục `LPS` (155 lát cắt ngang theo trục $z$ mỗi ca).
- **Phân bố lát cắt**:
  - Trung bình **137.9 lát có mô não / ca** (dao động `98 – 149` lát) và **17.1 lát rỗng / ca** ở đỉnh sọ và nền sọ.
  - Trung bình **66.2 lát có u / ca** (trung vị `68`, dao động `23 – 102` lát). Tổng toàn tập gồm **24.422 lát có u** và **26.476 lát mô não lành**.
- **Nhãn tổn thương**: Gồm các nhãn `{0, 1, 2, 4}` (nhãn `3` không tồn tại trong chuẩn BraTS 2020). Thể tích khối u toàn phần (Whole Tumor – WT) có trung vị **90.7 $\text{cm}^3$** (`7.3 – 361.8` $\text{cm}^3$).

---

## 3. Quy trình xử lý trong `src/data/build_dataset.py`

1. **Đọc đa chuỗi xung (`.nii/.nii.gz` hoặc `.h5`)**: Trích xuất 3 chuỗi xung giàu thông tin cấu trúc và ranh giới u nhất gồm **`[FLAIR, T1ce, T2]`** cùng nhãn phân vùng `seg`.
2. **Chuẩn hóa Z-score theo từng Volume & Modality**:
   - Chỉ tính trung bình $\mu$ và độ lệch chuẩn $\sigma$ trên vùng mô não ($\text{voxel} > 0$):
     $$z = \frac{x - \mu_{\text{brain}}}{\sigma_{\text{brain}}}$$
   - Các điểm ảnh nền ngoài hộp sọ được giữ cố định bằng `0`.
3. **Cắt lát 2D theo mặt phẳng ngang (Axial) & Lọc nhiễu**:
   - Loại bỏ các lát rỗng ở đỉnh sọ và nền sọ có diện tích mô não `< 500 pixels` (`--min_brain_area 500`).
   - Chỉ giữ lại các lát có u đạt diện tích tối thiểu **`>= 50 pixels`** (`--min_tumor_area 50`) nhằm loại bỏ nhiễu điểm ảnh rời rạc ở rìa khối u.
   - Lấy mẫu cân bằng một lượng lát cắt mô não lành (`has_tumor = 0`) theo tỷ lệ `--neg_ratio` trên từng ca bệnh.
4. **Gộp nhãn Whole Tumor (WT) & Trích xuất Bounding Box**:
   - Hợp nhất các vùng hoại tử (`NCR/NET = 1`), phù nề quanh u (`ED = 2`) và u bắt thuốc (`ET = 4`) thành mặt nạ nhị phân **Whole Tumor**: $\text{WT} = (\text{seg} > 0)$.
   - Tự động trích xuất tọa độ hộp giới hạn chuẩn `[x_min, y_min, x_max, y_max]` bao quanh vùng WT.
5. **Đóng gói ảnh 3 kênh & Chia tập theo ca bệnh (Patient-level Split)**:
   - Ghép 3 kênh `[FLAIR, T1ce, T2]` thành ảnh `.png` 3 kênh (`dataset/brats2020_png/images/`), mặt nạ nhị phân `.png` (`dataset/brats2020_png/masks/`), tệp `.npz` (`float16`) và xuất tệp chú thích chuẩn `dataset/brats2020_annotations.json` tương thích trực tiếp với `UnifiedBrainMRIDataset`.
   - Chia tập **Train / Val / Test (70% / 15% / 15%) theo từng ca bệnh (`case_id`)**, đảm bảo các lát cắt của cùng một bệnh nhân không bao giờ bị rò rỉ giữa tập huấn luyện và tập kiểm thử.

---

## 4. Hướng dẫn cài đặt và chạy lệnh

### Cài đặt thư viện
```bash
pip install nibabel h5py opencv-python numpy pandas matplotlib tqdm
```

### Bước 1: Chạy khảo sát thống kê (EDA)
```bash
python src/data/eda_brats.py --data_dir dataset/brats2020 --out_dir reports
```

### Bước 2: Chạy pipeline cắt lát 2D, tạo mặt nạ WT, Bounding Box & xuất JSON
```bash
python src/data/build_dataset.py \
    --data_dir dataset/brats2020 \
    --out_dir dataset/brats2020_png \
    --out_json dataset/brats2020_annotations.json \
    --min_tumor_area 50 \
    --neg_ratio 1.0
```

### Bước 3: Kiểm định chất lượng dữ liệu đầu ra
```bash
python src/data/check_dataset.py --out_dir dataset/brats2020_png --min_tumor_area 50
```
