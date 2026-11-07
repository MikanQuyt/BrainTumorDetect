# Báo Cáo Kiểm Tra & Chỉnh Sửa Nhánh `thai_van_huy_tuan1`

**Sinh viên phụ trách:** Thái Văn Huy (MSSV: 23130068)  
**Nhiệm vụ Tuần 1:**
1. Tải và phân tích thống kê (EDA) cho **Dataset 1 (Navoneel – 253 ảnh)** và **Dataset 2 (indk214 – Segmentation & Classification)**.
2. Lập trình và chạy script **Cắt sát hộp sọ tự động (Automated Extreme-Point Skull Cropping bằng OpenCV + CLAHE)** trên toàn bộ Dataset 1 & 2; viết hàm tự động trích xuất tọa độ hộp chuẩn `[x_min, y_min, x_max, y_max]` bao quanh Ground-Truth Mask của Dataset 2 xuất ra tệp **`ds2_annotations.json`**.

---

## 1. Đánh giá hiện trạng nhánh `thai_van_huy_tuan1` trước khi chỉnh sửa

### 1.1. Những phần đã hoàn thành tốt
- **Phân tích thống kê (EDA) số liệu trên Dataset 1 & Dataset 2** ([output_lan2/BAO_CAO_TUAN1.md](output_lan2/BAO_CAO_TUAN1.md) và [output_lan2/dataset_statistics.json](output_lan2/dataset_statistics.json)):
  - **Dataset 1 (`navoneel`)**: Đã kiểm chứng và loại bỏ 253 tệp bản sao trong thư mục con `brain_tumor_dataset/`, giữ đúng **253 ảnh gốc** (`yes`: 155 ảnh – 61.265%, `no`: 98 ảnh – 38.735%), thống kê đầy đủ phân bố kích thước (`150×168` đến `1920×1427`), định dạng (`.jpg`, `.jpeg`, `.png`), hệ màu (`RGB`: 217, `L`: 30, `RGBA`: 5, `P`: 1) và phát hiện 22 cụm ảnh trùng mức xám.
  - **Dataset 2 (`indk214` – Classification)**: Thống kê đầy đủ **7.023 ảnh** (`Training`: 5.712, `Testing`: 1.311) trên 4 lớp (`glioma`: 1.621, `meningioma`: 1.645, `no_tumor`: 2.000, `pituitary`: 1.757) và phát hiện **189 nội dung ảnh trùng lặp giữa tập Train và Test gốc** (chiếm 16.48% tập Test).
  - **Dataset 2 (`indk214` – Segmentation)**: Thống kê đầy đủ **2.192 cặp ảnh – mask** (`glioma`: 554, `meningioma`: 708, `pituitary`: 930), phát hiện nền mask gốc có mức xám `3` (không phải `0`) nên đã dùng ngưỡng nhị phân hóa `>= 128` thay vì `> 0`.
- **Cắt hộp sọ & bảo toàn vùng u**:
  - Đã xử lý toàn bộ **9.468 ảnh MRI** (`253` Navoneel + `7.023` Indk214 Classification + `2.192` Indk214 Segmentation) và **2.192 mask**, có cơ chế kiểm tra `contains(box, tumor_box)` đảm bảo không làm mất bất kỳ pixel u nào sau khi cắt.

---

### 1.2. Các điểm còn thiếu hoặc sai lệch so với yêu cầu đề bài (Đã phát hiện)

| STT | Hạng mục kiểm tra | Hiện trạng ban đầu trên nhánh `thai_van_huy_tuan1` | Mức độ |
| :-: | :--- | :--- | :---: |
| **1** | **Bộ lọc tăng cường tương phản CLAHE (`OpenCV + CLAHE`)** | Trong [preprocess.py](preprocess.py), hàm `process_sample()` chỉ gọi `image.crop(box)` của thư viện PIL rồi lưu thẳng ra PNG, **hoàn toàn thiếu bước áp dụng bộ lọc CLAHE (`cv2.createCLAHE`)** lên ảnh Dataset 1 & Dataset 2. | **Thiếu bắt buộc** |
| **2** | **Thuật toán Extreme-Point Skull Cropping** | Hàm `head_box()` trong [preprocess.py](preprocess.py) chưa thực hiện bước khử nhiễu hình thái học `cv2.erode` / `cv2.dilate` và chưa trích xuất tường minh 4 điểm cực trị (Tây $x_{\min}$, Đông $x_{\max}$, Bắc $y_{\min}$, Nam $y_{\max}$) trên contour hộp sọ lớn nhất. | **Cần chuẩn hóa** |
| **3** | **Tên tệp chú thích xuất ra (`ds2_annotations.json`) & Định dạng chuẩn** | Đề bài yêu cầu xuất ra tệp **`ds2_annotations.json`**, nhưng [preprocess.py](preprocess.py) chỉ xuất ra `output_lan2/ds_annotations.json` (thiếu chữ `2`) và chưa xuất tệp `dataset/ds2_annotations.json` theo chuẩn `UnifiedBrainMRIDataset` (có trường `gt_box = [x_min, y_min, x_max, y_max]`). | **Sai tên tệp** |
| **4** | **Biểu đồ trực quan hóa EDA (`.png`)** | Mới chỉ có bảng số liệu trong `dataset_statistics.json` và `BAO_CAO_TUAN1.md`, chưa có các biểu đồ trực quan (`.png`) về phân bố lớp, kích thước ảnh, diện tích khối u và lưới ảnh mẫu kiểm chứng sau khi cắt + CLAHE. | **Thiếu biểu đồ** |
| **5** | **Nhầm lẫn tệp `README.md` từ nhiệm vụ BRISC 2025 (Dataset 3)** | Tệp [README.md](README.md) ở thư mục gốc của nhánh đang mô tả **BRISC 2025 (Dataset 3)** (do gộp nhầm từ commit `a365d7d`), không khớp với nhiệm vụ Dataset 1 & Dataset 2 của Thái Văn Huy. | **Sai tài liệu gốc** |

---

## 2. Chi tiết các nội dung đã chỉnh sửa & bổ sung

1. **Cập nhật [preprocess.py](preprocess.py)**:
   - Bổ sung hàm [`apply_clahe`](preprocess.py#L25-L34) sử dụng `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))` và áp dụng trực tiếp lên toàn bộ ảnh MRI của Dataset 1 & Dataset 2 sau khi cắt sát hộp sọ trong [`process_sample`](preprocess.py#L285-L299).
   - Cập nhật hàm [`head_box`](preprocess.py#L44-L115) bổ sung khử nhiễu hình thái học `cv2.erode` / `cv2.dilate` (2 vòng) và trích xuất 4 điểm cực trị (`ext_west`, `ext_east`, `ext_north`, `ext_south`) của contour hộp sọ lớn nhất.
   - Bổ sung trường chuẩn `"gt_box": [x_min, y_min, x_max, y_max]` vào mỗi bản ghi chú thích và xuất đồng thời ra **`output_lan2/ds2_annotations.json`** (cùng `output_lan2/ds_annotations.json` để tương thích ngược).
   - Tự động nhận diện đường dẫn dữ liệu trong kho mã nguồn (`dataset/navoneel`, `dataset/indk214`) bên cạnh đường dẫn `~/Downloads`.

2. **Bổ sung bộ mô-đun chuẩn hóa trong `data/` và tệp đích `dataset/ds2_annotations.json`**:
   - [data/skull_crop.py](data/skull_crop.py): Cài đặt chuẩn hóa `skull_crop_box`, `crop_with_mask`, `apply_clahe`, `binarize_mask(thresh=127)` và `mask_to_box` trả về `[x_min, y_min, x_max, y_max]`.
   - [data/ds12_index.py](data/ds12_index.py), [data/eda_ds1_ds2.py](data/eda_ds1_ds2.py), [data/build_ds1_ds2.py](data/build_ds1_ds2.py), [data/check_ds1_ds2.py](data/check_ds1_ds2.py): Bộ script EDA, cắt hộp sọ + CLAHE và kiểm chứng tự động trên toàn bộ **253 ảnh DS1** và **9.215 ảnh DS2** (`2.192` cặp Segmentation + `7.023` ảnh Classification).
   - Xuất đầy đủ tệp **[dataset/ds2_annotations.json](dataset/ds2_annotations.json)**, **[output_lan2/ds2_annotations.json](output_lan2/ds2_annotations.json)** và **[dataset/ds1_annotations.json](dataset/ds1_annotations.json)**.

3. **Bổ sung trọn bộ biểu đồ EDA & ảnh kiểm chứng trực quan trong `reports/`**:
   - Biểu đồ EDA ([reports/eda_ds1_ds2/](reports/eda_ds1_ds2/)):
     - `fig_class_counts.png`: Biểu đồ phân bố lớp của Dataset 1 & Dataset 2 (Classification & Segmentation).
     - `fig_image_sizes.png`: Biểu đồ phân tán kích thước `(W, H)` của các bộ dữ liệu.
     - `fig_ds2_tumor_area.png`: Biểu đồ phân bố tỷ lệ diện tích khối u (% ảnh) theo 3 loại u (`glioma`, `meningioma`, `pituitary`).
     - Bảng thống kê CSV: `class_counts.csv`, `image_stats.csv`, `ds2_mask_stats.csv`, `eda_summary.txt`.
   - Kết quả kiểm chứng Skull Cropping + CLAHE + Bounding Box ([reports/build_ds1_ds2/](reports/build_ds1_ds2/)):
     - `check_samples.png`: Lưới ảnh MRI sau khi cắt sát hộp sọ + lọc CLAHE, phủ Ground-Truth Mask (đỏ) và hộp chuẩn `gt_box = [x_min, y_min, x_max, y_max]` (xanh lá).
     - `crop_log.csv`: Nhật ký tọa độ khung cắt và tỷ lệ diện tích giữ lại trên từng ảnh.

4. **Cập nhật [README.md](README.md) gốc và [output_lan2/BAO_CAO_TUAN1.md](output_lan2/BAO_CAO_TUAN1.md)**:
   - Thay thế nội dung BRISC 2025 bị lẫn trong [README.md](README.md) bằng báo cáo chuẩn của **Dataset 1 (Navoneel) & Dataset 2 (indk214)** kèm hướng dẫn chạy lệnh và cấu trúc tệp đầu ra `ds2_annotations.json`.
   - Cập nhật [output_lan2/BAO_CAO_TUAN1.md](output_lan2/BAO_CAO_TUAN1.md) bổ sung mô tả bước cân bằng tương phản cục bộ **CLAHE** và tên tệp xuất chuẩn `ds2_annotations.json`.
