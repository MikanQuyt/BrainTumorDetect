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
